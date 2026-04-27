"""
Finetune a base model to be a prover model.
Run on one GPU e.g. for debugging:

python -m nanoproof.sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m nanoproof.sft
"""

import os
import random
import argparse
from dataclasses import asdict

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import wandb
import torch
import torch.distributed as dist
import leantree.augmentations

from nanoproof.common import (
    compute_init,
    compute_cleanup,
    print0,
    create_metrics_logger,
    add_logging_args,
    autodetect_device_type,
    create_run_dirs,
    get_lr_multiplier,
)
from nanoproof.checkpoints import load_model, save_checkpoint
from nanoproof.engine import Engine
from nanoproof.data.sft.leantree import leantree_transitions
from nanoproof.data.sft.leantree_dataloader import sft_data_generator
from scripts.policy_eval import eval_tactic_accuracy, eval_critic_errors


# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(
    description="Finetune a base model to be a prover model", allow_abbrev=False
)
# Logging
add_logging_args(parser)
parser.add_argument("--seed", type=int, default=0, help="random seed")
# Model source
parser.add_argument(
    "--model-path",
    type=str,
    required=True,
    help="path to model_NNNNNN.pt to load from (relative to models/ or absolute)",
)
parser.add_argument(
    "--resume-from",
    type=str,
    default=None,
    help="path to model_NNNNNN.pt to resume SFT from (overrides --model-path)",
)
# Runtime
parser.add_argument(
    "--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)"
)
parser.add_argument(
    "--dtype", type=str, default="bfloat16", help="data type for training"
)
parser.add_argument(
    "--device-batch-size", type=int, default=8, help="per-device batch size"
)
# Optimization
parser.add_argument(
    "--num-epochs", type=int, default=5, help="number of training epochs"
)
parser.add_argument(
    "--num-iterations",
    type=int,
    default=-1,
    help="override number of iterations (-1 = use num_epochs)",
)
parser.add_argument(
    "--target-examples-per-step",
    type=int,
    default=512,
    help="target examples per optimization step",
)
parser.add_argument(
    "--unembedding-lr",
    type=float,
    default=0.004,
    help="learning rate for unembedding parameters",
)
parser.add_argument(
    "--embedding-lr",
    type=float,
    default=0.3,
    help="learning rate for embedding parameters",
)
parser.add_argument(
    "--matrix-lr",
    type=float,
    default=0.02,
    help="learning rate for matrix parameters (Muon)",
)
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay")
parser.add_argument(
    "--init-lr-frac", type=float, default=0.8, help="initial learning rate fraction"
)
parser.add_argument(
    "--warmup-ratio", type=float, default=0.0, help="ratio of progress for LR warmup"
)
parser.add_argument(
    "--warmdown-ratio",
    type=float,
    default=0.5,
    help="ratio of progress for LR warmdown",
)
parser.add_argument(
    "--final-lr-frac",
    type=float,
    default=0.0,
    help="final LR as fraction of initial LR",
)
# Evaluation
parser.add_argument(
    "--eval-every", type=int, default=200, help="evaluate every N steps"
)
parser.add_argument("--eval-steps", type=int, default=200, help="number of eval steps")
parser.add_argument(
    "--sample-every", type=int, default=100, help="sample from model every N steps"
)
parser.add_argument(
    "--eval-metrics-max-problems",
    type=int,
    default=1024,
    help="max problems for eval metrics",
)
parser.add_argument(
    "--save-every-epoch",
    type=int,
    default=1,
    help="save a checkpoint every N epochs (-1 disables intermediate saves; final model is always saved)",
)
# Loss weighting
parser.add_argument(
    "--value-weight",
    type=float,
    default=0.01,
    help="weight for value (critic) samples relative to policy samples",
)
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

# Run directories
log_dir, model_dir = create_run_dirs("sft", args.run, args_dict=user_config)

# metrics logging init
run_log = create_metrics_logger(
    "nanoproof-sft",
    args,
    master_process,
    {**user_config, "log_dir": log_dir, "model_dir": model_dir},
    log_dir=log_dir,
    save_code=True,
)

# Load the model and tokenizer
if args.resume_from is not None:
    # Resume from an SFT checkpoint
    model, tokenizer, meta = load_model(args.resume_from, device, phase="train")
    start_step = meta.get("step", 0)
    print0(f"Resuming from SFT checkpoint at step {start_step}")
else:
    # Start fresh from base/mid checkpoint
    model, tokenizer, meta = load_model(args.model_path, device, phase="train")
    start_step = 0
orig_model = model  # original, uncompiled model
# model = torch.compile(model, dynamic=True) # doesn't work super well because of variable lengths of inputs
engine = Engine(model, tokenizer)  # will be used for inline model evaluation only
bos_token = tokenizer.get_bos_token_id()
value_delim_tok = tokenizer.encode_special(
    "<|value|>"
)  # for distinguishing policy vs value samples

# -----------------------------------------------------------------------------
# DataLoader

examples_per_step = args.device_batch_size * ddp_world_size
print0(f"Target examples per step: {args.target_examples_per_step}")
print0(f"Device batch size: {args.device_batch_size}")
print0(f"Examples per step is device_batch_size * ddp_world_size: {examples_per_step}")
assert args.target_examples_per_step % examples_per_step == 0, (
    "Target examples per step must be divisible by examples per step"
)
grad_accum_steps = args.target_examples_per_step // examples_per_step
print0(f"=> Setting grad accum steps: {grad_accum_steps}")

augmentations = [
    leantree.augmentations.ShuffleGoalsAndHypotheses(seed=args.seed),
    leantree.augmentations.RandomRename(seed=args.seed),
]
train_ds = list(leantree_transitions(split="train", augmentations=augmentations))
random.Random(args.seed).shuffle(train_ds)
val_ds = list(leantree_transitions(split="valid"))
print0(f"Train rows count: {len(train_ds)} | Val rows count: {len(val_ds)}")

train_loader = sft_data_generator(train_ds, batch_size=args.device_batch_size)
build_val_loader = lambda: sft_data_generator(val_ds, batch_size=args.device_batch_size)

# -----------------------------------------------------------------------------
# Initialize the Optimizer

optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)
# Set the initial learning rate as a fraction of the base learning rate
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group[
        "lr"
    ]  # save the initial learning so we can decay easily later

# -----------------------------------------------------------------------------
# Training loop

# Go!
progress = 0  # will go from 0 to 1 over the course of the epoch
step = start_step
epoch = 0
x, y, approx_progress, last_step = next(
    train_loader
)  # prefetch the very first batch of data
while True:
    # Synchronize last_step across all ranks to avoid hangs in the distributed setting
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    if last_step or step % args.eval_every == 0:
        model.eval()

        # evaluate the validation loss
        val_iter = iter(build_val_loader())
        losses = []
        for _ in range(args.eval_steps):
            val_inputs, val_targets, _, _ = next(val_iter)
            with torch.no_grad():
                loss = model(val_inputs, val_targets)
            losses.append(loss)
        val_loss = torch.stack(losses).mean()  # average over eval_steps
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)  # average over ranks
        val_loss = val_loss.item()

        tactic_results = eval_tactic_accuracy(
            model, tokenizer, build_val_loader(), eval_steps=args.eval_steps
        )
        critic_results = eval_critic_errors(
            model, tokenizer, build_val_loader(), eval_steps=args.eval_steps
        )

        print0(
            f"Step {step:05d} | Validation loss: {val_loss:.6f} | Tactic full accuracy: {tactic_results['full_acc']:.4%} | Tactic first token accuracy: {tactic_results['first_token_acc']:.4%} | Critic argmax MSE: {critic_results['argmax_mse']:.4f} | Critic soft MSE: {critic_results['soft_mse']:.4f}"
        )
        print0(
            f"  Entropy - Tactic first: {tactic_results['first_token_entropy']:.4f} | Tactic all: {tactic_results['all_tokens_entropy']:.4f} | Critic: {critic_results['entropy']:.4f}"
        )

        # Create confusion matrix for wandb
        bin_labels = [str(i) for i in range(1, 65)]

        run_log.log(
            {
                "step": step,
                "val_loss": val_loss,
                "val_full_acc": tactic_results["full_acc"],
                "val_first_token_acc": tactic_results["first_token_acc"],
                "val_first_token_entropy": tactic_results["first_token_entropy"],
                "val_all_tokens_entropy": tactic_results["all_tokens_entropy"],
                "val_tactic_samples": tactic_results["total_samples"],
                "val_critic_argmax_mse": critic_results["argmax_mse"],
                "val_critic_soft_mse": critic_results["soft_mse"],
                "val_critic_entropy": critic_results["entropy"],
                "val_critic_samples": critic_results["total_samples"],
                "val_critic_confusion": wandb.plot.confusion_matrix(
                    y_true=critic_results["y_true"],
                    preds=critic_results["y_pred"],
                    class_names=bin_labels,
                ),
            }
        )

        model.train()

    # evaluate accuracy of the multiple choice tasks (which are quick to run)
    if last_step or (step > 0 and step % args.sample_every == 0):
        model.eval()
        prompts = [
            "The capital of France is",
            "If 5*x + 3 = 13, then x is",
            # gold from mathlib: 'exact LipschitzWith.comp_locallyBoundedVariationOn (A i) h'
            """case h
\u03b9 : Type u_4
inst\u271d : Fintype \u03b9
f : \u211d \u2192 \u03b9 \u2192 \u211d
s : Set \u211d
h : LocallyBoundedVariationOn f s
A : \u2200 (i : \u03b9), LipschitzWith 1 fun x => x i
i : \u03b9
\u22a2 LocallyBoundedVariationOn (fun x => f x i) s
<|tactic|>""",
            # sensible tactic: 'intro h'
            """p q : Prop
\u22a2 p \u2227 q \u2192 p
<|tactic|>""",
            # sensible tactic: 'rfl'
            """\u22a2 2 + 3 = 5
<|tactic|>""",
            # sensible tactic: 'exact Or.inl \u27e8hp, hq\u27e9'
            """case mp.inl
p q r : Prop
hp : p
hq : q
\u22a2 p \u2227 q \u2228 p \u2227 r
<|tactic|>""",
            # sensible tactic: 'exact Exists.intro x0 hx0'
            """\u03b1 : Type
P : \u03b1 \u2192 Prop
inst\u271d : Inhabited \u03b1
h : \u2200 (x : \u03b1), P x
x0 : \u03b1 := default
hx0 : P x0
\u22a2 \u2203 x, P x
<|tactic|>""",
            """p q : Prop
\u22a2 p \u2227 q \u2192 p
<|value|>""",
            """\u03b1 : Type
P : \u03b1 \u2192 Prop
inst\u271d : Inhabited \u03b1
h : \u2200 (x : \u03b1), P x
x0 : \u03b1 := default
hx0 : P x0
\u22a2 \u2203 x, P x
<|value|>""",
        ]
        engine = Engine(orig_model, tokenizer)  # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend=bos_token)
            sample, _ = engine.generate_batch(
                tokens, num_samples=1, max_tokens=16, temperature=0
            )
            print0(tokenizer.decode(sample[0]) + "\n---")
        model.train()

    if last_step:
        completed_epoch = epoch + 1  # 1-indexed count of fully completed epochs
        is_final_epoch = epoch >= args.num_epochs - 1
        # Save a checkpoint every N epochs; the final epoch is saved unconditionally below.
        if (
            master_process
            and not is_final_epoch
            and args.save_every_epoch > 0
            and completed_epoch % args.save_every_epoch == 0
        ):
            save_checkpoint(
                model_dir,
                step,
                model.state_dict(),
                None,
                {
                    "step": step,
                    "val_loss": val_loss,
                    "model_config": asdict(orig_model.config),
                },
            )
            print(f"Saved epoch {completed_epoch} checkpoint to {model_dir}")
        if not is_final_epoch:
            print0(f"Epoch {epoch} done, starting next one.")
            epoch += 1
            train_loader = sft_data_generator(
                train_ds, batch_size=args.device_batch_size
            )
            progress = 0
        else:
            print0(f"Epoch {epoch} done, terminating.")
            break

    # evaluate the gradient
    num_tokens = torch.tensor(
        0, device=device
    )  # the number of "active" tokens of supervision seen
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets, approx_progress, last_step = next(
            train_loader
        )  # prefetch the next batch while the GPU is busy with forward/backward
        progress = max(
            progress, approx_progress
        )  # only increase progress monotonically

        # Compute per-token losses to apply different weights to value vs policy samples
        per_token_loss = model(
            train_inputs, train_targets, loss_reduction="none"
        )  # (B*T,)
        per_token_loss = per_token_loss.view(train_inputs.shape)  # (B, T)

        # Identify value samples: those where input contains the value delimiter token
        is_value_sample = (train_inputs == value_delim_tok).any(dim=1)  # (B,)

        # Create per-sample weights: value_weight for value samples, 1.0 for policy samples
        sample_weights = torch.where(is_value_sample, args.value_weight, 1.0)  # (B,)

        # Compute weighted loss: weight each token by its sample's weight
        token_mask = train_targets >= 0  # (B, T)
        weighted_token_loss = per_token_loss * sample_weights.unsqueeze(1)  # (B, T)

        # Mean over all valid tokens (weighted)
        loss = (weighted_token_loss * token_mask).sum() / token_mask.sum()
        train_loss = loss.detach()  # for logging
        loss = (
            loss / grad_accum_steps
        )  # each .backward() is a grad sum => normalize loss here
        loss.backward()  # accumulate the gradient
        num_tokens += (train_targets >= 0).sum()
    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)  # sum over ranks

    # learning rate scheduler (uses global progress across all epochs)
    global_progress = (epoch + progress) / args.num_epochs
    lrm = get_lr_multiplier(global_progress, args)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    # step the optimizer
    optimizer.step()
    model.zero_grad(set_to_none=True)

    pct_done = 100 * progress
    if ddp:
        pct_done_tensor = torch.tensor([pct_done], dtype=torch.float32, device=device)
        gathered_pct_done = [
            torch.zeros_like(pct_done_tensor) for _ in range(ddp_world_size)
        ]
        dist.all_gather(gathered_pct_done, pct_done_tensor)
        pct_dones = [t.item() for t in gathered_pct_done]
        pct_done_str = "[" + ", ".join(f"{p:.2f}" for p in pct_dones) + "]%"
    else:
        pct_done_str = f"{pct_done:.2f}%"

    # logging
    train_loss_item = train_loss.item()
    num_tokens_item = num_tokens.item()
    print0(
        f"Step {step:05d} ({pct_done_str}, ep {epoch:02d}/{args.num_epochs:02d}) | Training loss: {train_loss_item:.6f}| lrm: {lrm:.6f}| num_tokens: {num_tokens_item:,}"
    )
    run_log.log(
        {
            "step": step,
            "lrm": lrm,
            "train_loss": train_loss_item,
            "num_tokens": num_tokens_item,
        }
    )

    step += 1

# Save the model at the end of the run
if master_process:
    save_checkpoint(
        model_dir,
        step,
        model.state_dict(),
        None,  # note: we don't bother to save the optimizer state
        {
            "step": step,
            "val_loss": val_loss,
            "model_config": asdict(orig_model.config),
        },
    )
    print(f"Saved model checkpoint to {model_dir}")

# Cleanup
run_log.finish()
compute_cleanup()
