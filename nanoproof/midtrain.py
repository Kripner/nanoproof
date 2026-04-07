"""
Midtrain the model. Same as pretraining but simpler.
Run as:

python -m nanoproof.midtrain

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m nanoproof.midtrain -- --device-batch-size=16
"""

from collections import deque
from dataclasses import asdict
import os
import time
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import wandb
import torch
import torch.distributed as dist

from nanoproof.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type, is_ddp_initialized, create_run_dirs, GLOBAL_CONFIG, get_lr_multiplier
from nanoproof.tokenizer import get_token_bytes
from nanoproof.checkpoints import save_checkpoint
from nanoproof.loss_eval import evaluate_bpb
from nanoproof.checkpoints import load_model
from nanoproof.data.midtrain.leangithubraw import iter_data

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Midtrain the model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model source
parser.add_argument("--model-path", type=str, required=True, help="model path to load from (relative to models/ or absolute)")
parser.add_argument("--step", type=int, default=None, help="step to load the model from (default: latest)")
# Training
parser.add_argument("--dtype", type=str, default="bfloat16", help="data type for training")
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--max-seq-len", type=int, default=GLOBAL_CONFIG.max_seq_len, help="max context length")
parser.add_argument("--device-batch-size", type=int, default=16, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=491520, help="total batch size in tokens")
parser.add_argument("--eval-tokens", type=int, default=-1, help="number of tokens to evaluate val loss on (-1 = 20*total_batch_size)")
# Optimization
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--init-lr-frac", type=float, default=0.8, help="initial learning rate fraction")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of progress for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of progress for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay")
# Evaluation
parser.add_argument("--eval-every", type=int, default=150, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--dry-run", type=int, default=0, help="dry_run=1 logs to wandb but skips checkpoints/report")
args = parser.parse_args()
user_config = vars(args).copy()

# Derived defaults
if args.eval_tokens == -1:
    args.eval_tokens = 20 * args.total_batch_size
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# Run directories
log_dir, model_dir = create_run_dirs("midtrain", args.run, args_dict=user_config)

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanoproof-mid", name=args.run, dir=log_dir, config={**user_config, "log_dir": log_dir, "model_dir": model_dir})

# Load the model and tokenizer
model, tokenizer, meta = load_model(args.model_path, device, phase="train", step=args.step)
pretrain_batch_size = meta.get("device_batch_size", None)
if pretrain_batch_size is not None and args.device_batch_size > pretrain_batch_size:
    print0(f"FOOTGUN WARNING: base model training used device_batch_size {pretrain_batch_size}, did you pass in a good --device-batch-size to this script?")
orig_model = model
model = torch.compile(model, dynamic=False)
depth = model.config.n_layer
num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
token_bytes = get_token_bytes(device=device)

# Initialize the Optimizer
optimizer = model.setup_optimizer(unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr, matrix_lr=args.matrix_lr, weight_decay=args.weight_decay)
# Override the initial learning rate as a fraction of the base learning rate
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"] # save the initial learning so we can decay easily later

# Midtraining data mixture and DataLoader
base_dir = get_base_dir()
train_loader = iter_data(args.device_batch_size, args.max_seq_len, "train")
build_val_loader = lambda: iter_data(args.device_batch_size, args.max_seq_len, "val")

progress = 0 # will go from 0 to 1 over the course of the epoch

# -----------------------------------------------------------------------------
# Training loop
x, y, approx_progress, last_step = next(train_loader) # prefetch the very first batch of data
min_val_bpb = float("inf")
smooth_train_loss = 0 # EMA of training loss
ema_beta = 0.9 # EMA decay factor
total_training_time = 0 # total wall-clock time of training
step = 0
while True:
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # Synchronize last_step across all ranks to avoid hangs in the distributed setting
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # save checkpoint at the end of the run (only on master process)
    if master_process and last_step and not args.dry_run:
        save_checkpoint(
            model_dir,
            step,
            orig_model.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": asdict(orig_model.config),
                "user_config": user_config, # inputs to the training script
            }
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        loss.backward()
        x, y, approx_progress, last_step = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
        progress = max(progress, approx_progress) # only increase progress monotonically
    # step the optimizer
    lrm = get_lr_multiplier(progress, args)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # State
    step += 1

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * progress
    if ddp:
        pct_done_tensor = torch.tensor([pct_done], dtype=torch.float32, device=device)
        gathered_pct_done = [torch.zeros_like(pct_done_tensor) for _ in range(ddp_world_size)]
        dist.all_gather(gathered_pct_done, pct_done_tensor)
        pct_dones = [t.item() for t in gathered_pct_done]
        pct_done_str = "[" + ", ".join(f"{p:.2f}" for p in pct_dones) + "]%"
    else:
        pct_done_str = f"{pct_done:.2f}%"

    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # in %
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    print0(f"step {step:05d} ({pct_done_str}) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        })

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
if not args.dry_run:
    from nanoproof.report import get_report
    get_report().log(section="Midtraining", data=[
        user_config, # CLI args
        { # stats about the training setup
            "Number of iterations": step,
            "DDP world size": ddp_world_size,
        },
        { # stats about training outcomes
            "Minimum validation bpb": min_val_bpb,
        }
    ])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
