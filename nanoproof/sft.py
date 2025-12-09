"""
Finetune a base model to be a prover model.
Run on one GPU e.g. for debugging:

python -m scripts.sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.sft
"""

import os

import leantree.augmentations

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random

import wandb
import torch
import torch.distributed as dist
from contextlib import nullcontext

from nanoproof.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb, autodetect_device_type
from nanoproof.checkpoints import load_model, save_checkpoint
from nanoproof.engine import Engine
from nanoproof.data.leantree import iter_data
from nanoproof.data.leantree_dataloader import sft_data_generator
from scripts.policy_eval import eval_tactic_accuracy

# -----------------------------------------------------------------------------
# SFT Hyperparameters
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
seed = 0
# input model options
source = "mid" # base|mid , which checkpoint to load the model from (base model or midtrained model)
model_tag = "d26" # model tag to load the model from (base model or midtrained model)
step = None # step to load the model from (base model or midtrained model)
# compute/precision
device_type = "" # cuda|cpu|mps (empty => autodetect)
dtype = "bfloat16"
device_batch_size = 8 # (maybe) max to avoid OOM (on A100 40GB)
# optimization
num_epochs = 1
num_iterations = -1 # override number of iterations (-1 = disable, use num_epochs to derive it)
target_examples_per_step = 512
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.02
# evaluation and logging there of
eval_every = 100
eval_steps = 100
# eval_metrics_every = 200
sample_every = 100
eval_metrics_max_problems = 1024
# now allow CLI to override the settings via the configurator lol
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanoproof', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # possibly useful for logging
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanoproof-sft", name=run, config=user_config, save_code=True)

# Load the model and tokenizer
model, tokenizer, meta = load_model(source, device, phase="train", model_tag=model_tag, step=step)
orig_model = model # original, uncompiled model
# model = torch.compile(model, dynamic=True) # doesn't work super well because of variable lengths of inputs
engine = Engine(model, tokenizer) # will be used for inline model evaluation only
bos_token = tokenizer.get_bos_token_id()

# -----------------------------------------------------------------------------
# DataLoader

examples_per_step = device_batch_size * ddp_world_size
print0(f"Target examples per step: {target_examples_per_step}")
print0(f"Device batch size: {device_batch_size}")
print0(f"Examples per step is device_batch_size * ddp_world_size: {examples_per_step}")
assert target_examples_per_step % examples_per_step == 0, "Target examples per step must be divisible by examples per step"
grad_accum_steps = target_examples_per_step // examples_per_step
print0(f"=> Setting grad accum steps: {grad_accum_steps}")

augmentations = [
    leantree.augmentations.ShuffleGoalsAndHypotheses(seed=seed),
    leantree.augmentations.RandomRename(seed=seed),
]

train_ds = list(iter_data(split="train", augmentations=augmentations))
random.Random(seed).shuffle(train_ds)
val_ds = list(iter_data(split="val"))
print0(f"Train dataset size: {len(train_ds)} | Val dataset size: {len(val_ds)}")

# if num_iterations == -1:
#     # derive num_iterations from num_epochs and the size of the dataset
#     assert num_epochs > 0, "num_epochs must be positive if num_iterations is -1"
#     num_iterations = (len(train_ds) // target_examples_per_step) * num_epochs
#     print0(f"=> Setting number of iterations: {num_iterations}")
train_loader = sft_data_generator(train_ds, batch_size=device_batch_size)
build_val_loader = lambda: sft_data_generator(val_ds, batch_size=device_batch_size)

# -----------------------------------------------------------------------------
# Initialize the Optimizer

optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)
# Set the initial learning rate as a fraction of the base learning rate
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"] # save the initial learning so we can decay easily later

# -----------------------------------------------------------------------------
# Training loop

# Learning rate scheduler
# def get_lr_multiplier(it):
#     lrm = 1.0 - it / num_iterations
#     return lrm


# Learning rate scheduler
def get_lr_multiplier(progress):
    # return max(0.0, 1.0 - progress)
    global_progress = (epoch + progress) / num_epochs
    return max(0.0, 1.0 - global_progress)

# Go!
progress = 0 # will go from 0 to 1 over the course of the epoch
step = 0
epoch = 0
x, y, approx_progress, last_step = next(train_loader) # prefetch the very first batch of data
while True:
    # Synchronize last_step across all ranks to avoid hangs in the distributed setting
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    if last_step or step % eval_every == 0:
        model.eval()

        # evaluate the validation loss
        val_iter = iter(build_val_loader())
        losses = []
        for _ in range(eval_steps):
            val_inputs, val_targets, _, _ = next(val_iter)
            with torch.no_grad(), autocast_ctx:
                loss = model(val_inputs, val_targets)
            losses.append(loss)
        val_loss = torch.stack(losses).mean() # average over eval_steps
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG) # average over ranks
        val_loss = val_loss.item()

        with autocast_ctx:
            results = eval_tactic_accuracy(model, build_val_loader(), max_steps=eval_steps)

        print0(f"Step {step:05d} | Validation loss: {val_loss:.6f} | Tactic full accuracy: {results['full_acc']:.4%} | Tactic first token accuracy: {results['first_token_acc']:.4%}")

        wandb_run.log({
            "step": step,
            "val_loss": val_loss,
            "val_full_acc": results["full_acc"],
            "val_first_token_acc": results["first_token_acc"],
        })

        model.train()

    # TODO: eval tactic accuracy
    # TODO: eval value MSE

    # evaluate accuracy of the multiple choice tasks (which are quick to run)
    if last_step or (step > 0 and step % sample_every == 0):
        model.eval()
        prompts = [
            "The capital of France is",
            "If 5*x + 3 = 13, then x is",
            # gold from mathlib: 'exact LipschitzWith.comp_locallyBoundedVariationOn (A i) h'
            """case h
ι : Type u_4
inst✝ : Fintype ι
f : ℝ → ι → ℝ
s : Set ℝ
h : LocallyBoundedVariationOn f s
A : ∀ (i : ι), LipschitzWith 1 fun x => x i
i : ι
⊢ LocallyBoundedVariationOn (fun x => f x i) s
<|tactic|>""",
            # sensible tactic: 'intro h'
            """p q : Prop
⊢ p ∧ q → p
<|tactic|>""",
            # sensible tactic: 'rfl'
            """⊢ 2 + 3 = 5
<|tactic|>""",
            # sensible tactic: 'exact Or.inl ⟨hp, hq⟩'
            """case mp.inl
p q r : Prop
hp : p
hq : q
⊢ p ∧ q ∨ p ∧ r
<|tactic|>""",
            # sensible tactic: 'exact Exists.intro x0 hx0'
            """α : Type
P : α → Prop
inst✝ : Inhabited α
h : ∀ (x : α), P x
x0 : α := default
hx0 : P x0
⊢ ∃ x, P x
<|tactic|> """,

            """p q : Prop
⊢ p ∧ q → p
<|value|> """,
            """α : Type
P : α → Prop
inst✝ : Inhabited α
h : ∀ (x : α), P x
x0 : α := default
hx0 : P x0
⊢ ∃ x, P x
<|value|> """,
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend=bos_token)
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]) + "\n---")
        model.train()

    if last_step:
        if epoch < num_epochs - 1:
            print0(f"Epoch {epoch} done, starting next one.")
            epoch += 1
            train_loader = sft_data_generator(train_ds, batch_size=device_batch_size)
            progress = 0
        else:
            print0(f"Epoch {epoch} done, terminating.")
            break

    # evaluate the gradient
    num_tokens = torch.tensor(0, device=device) # the number of "active" tokens of supervision seen
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets, approx_progress, last_step = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
        progress = max(progress, approx_progress) # only increase progress monotonically
        with autocast_ctx:
            loss = model(train_inputs, train_targets)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        loss.backward() # accumulate the gradient
        num_tokens += (train_targets >= 0).sum()
    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM) # sum over ranks

    # learning rate scheduler
    lrm = get_lr_multiplier(progress)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    # step the optimizers
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    pct_done = 100 * progress
    if ddp:
        pct_done_tensor = torch.tensor([pct_done], dtype=torch.float32, device=device)
        gathered_pct_done = [torch.zeros_like(pct_done_tensor) for _ in range(ddp_world_size)]
        dist.all_gather(gathered_pct_done, pct_done_tensor)
        pct_dones = [t.item() for t in gathered_pct_done]
        pct_done_str = "[" + ", ".join(f"{p:.2f}" for p in pct_dones) + "]%"
    else:
        pct_done_str = f"{pct_done:.2f}%"

    # logging
    train_loss_item = train_loss.item()
    num_tokens_item = num_tokens.item()
    print0(f"Step {step:05d} ({pct_done_str}, ep {epoch:02d}/{num_epochs:02d}) | Training loss: {train_loss_item:.6f}| lrm: {lrm:.6f}| num_tokens: {num_tokens_item:,}")
    wandb_run.log({
        "step": step,
        "lrm": lrm,
        "train_loss": train_loss_item,
        "num_tokens": num_tokens_item,
    })

    step += 1

# Save the model at the end of the run
if master_process:
    base_dir = get_base_dir()
    depth = model.config.n_layer
    model_tag = f"d{depth}" # base the model tag on the depth of the base model
    checkpoint_dir = os.path.join(base_dir, "sft_checkpoints", model_tag)
    model_config_kwargs = model.config.__dict__ # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),
        None, # note: we don't bother to save the optimizer state
        {
            "step": step,
            "val_loss": val_loss,
            "model_config": model_config_kwargs,
        }
    )
    print(f"✅ Saved model checkpoint to {checkpoint_dir}")

# Log to report
from nanoproof.report import get_report
get_report().log(section="SFT", data=[
    user_config, # CLI args
    {
        "Training rows": len(train_ds),
        "Number of iterations": step,
        "Training loss": train_loss_item,
        "Validation loss": val_loss,
    },
])

# Cleanup
wandb_run.finish()
compute_cleanup()