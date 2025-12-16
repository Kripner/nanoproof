import os

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
from nanoproof.data.leantree_dataloader import rl_data_generator
from nanoproof.experience_collection import ReplayBuffer, TheoremsSampler, Config, run_actor
from nanoproof.search import TacticModel
from scripts.minif2f_eval import eval_minif2f

# TODO: log times to see where is the bottleneck
# TODO: if tactic application results in a state that already is on the path from root, skip the tactic (otherwise we sometimes get stuck in loop of eg. rw [add_comm])
# TODO: store all transitions in a file

# TODO: (maybe) try removing each tactic and if the proof is still valid, do not add the transition to the replay buffer

# -----------------------------------------------------------------------------
# RL Hyperparameters
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
seed = 0
# compute/precision
device_type = "" # cuda|cpu|mps (empty => autodetect)
dtype = "bfloat16"
device_batch_size = 8 # (maybe) max to avoid OOM (on A100 40GB)
# data
fraction_sft = 0.1  # 10% of data will come from Mathlib (leantree), 90% from replay buffer
collect_every = 100  # how many steps to train between RL data collections
# collect_steps = 100  # how many proofs to collect per actor
collect_steps = 2  # TODO
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
# eval_metrics_every = 200
sample_every = 100
eval_metrics_max_problems = 1024
save_every = 1000
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
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanoproof-rl", name=run, config=user_config, save_code=True)

tactic_model = TacticModel.create()
model = tactic_model.network

# -----------------------------------------------------------------------------
# DataLoader

examples_per_step = device_batch_size * ddp_world_size
print0(f"Target examples per step: {target_examples_per_step}")
print0(f"Device batch size: {device_batch_size}")
print0(f"Examples per step is device_batch_size * ddp_world_size: {examples_per_step}")
assert target_examples_per_step % examples_per_step == 0, "Target examples per step must be divisible by examples per step"
grad_accum_steps = target_examples_per_step // examples_per_step
print0(f"=> Setting grad accum steps: {grad_accum_steps}")

rank_seed = seed + ddp_rank
mathlib_train = list(iter_data(split="train"))
random.Random(rank_seed).shuffle(mathlib_train)

config = Config()
replay_buffer = ReplayBuffer(config, seed=rank_seed)
theorems_sampler = TheoremsSampler(seed=rank_seed)

def train_generator():
    rng = random.Random(rank_seed)
    mathlib_iter = iter(mathlib_train)
    while True:
        # assert len(replay_buffer.buffer) > 100  # TODO
        if rng.random() < fraction_sft:
            try:
                yield next(mathlib_iter)
            except StopIteration:
                mathlib_iter = iter(mathlib_train)
                yield next(mathlib_iter)
        else:
            yield replay_buffer.sample_transition()

train_loader = rl_data_generator(train_generator(), batch_size=device_batch_size)

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

# -----------------------------------------------------------------------------
# Training loop

# Go!
step = 0
while True:
    if step % collect_every == 0:
        # collect proofs
        model.eval()
        run_actor(collect_steps, config, tactic_model, replay_buffer, theorems_sampler)
        model.train()
        replay_buffer.synchronize()

    if step % eval_every == 0:
        # TODO: also evaluate on a val split of LeanWorkBook
        model.eval()
        minif2f_results = eval_minif2f(tactic_model, max_theorems=64)
        # minif2f_results = eval_minif2f(tactic_model, max_theorems=2)  # TODO
        print0(f"Step {step:05d} | minif2f success rate: {minif2f_results['success_rate']:.4%} ({minif2f_results['solved']}/{minif2f_results['total']}) | error rate: {minif2f_results['error_rate']:.4%}")
        wandb_run.log({
            "step": step,
            "minif2f": minif2f_results['success_rate'],
        })
        model.train()

    if step > 0 and step % save_every == 0 and master_process:
        base_dir = get_base_dir()
        depth = model.config.n_layer
        model_tag = f"d{depth}" # base the model tag on the depth of the base model
        checkpoint_dir = os.path.join(base_dir, "rl_checkpoints", model_tag)
        model_config_kwargs = model.config.__dict__ # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None, # note: we don't bother to save the optimizer state
            {
                "step": step,
                "model_config": model_config_kwargs,
            }
        )
        print(f"Saved model checkpoint to {checkpoint_dir}")

    # evaluate the gradient
    num_tokens = torch.tensor(0, device=device) # the number of "active" tokens of supervision seen
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
        with autocast_ctx:
            loss = model(train_inputs, train_targets)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        loss.backward() # accumulate the gradient
        num_tokens += (train_targets >= 0).sum()
    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM) # sum over ranks

    # step the optimizers
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    # logging
    train_loss_item = train_loss.item()
    num_tokens_item = num_tokens.item()
    print0(f"Step {step:05d} | Training loss: {train_loss_item:.6f} | num_tokens: {num_tokens_item:,} | replay_buffer_size: {len(replay_buffer.buffer)}")
    wandb_run.log({
        "step": step,
        "train_loss": train_loss_item,
        "num_tokens": num_tokens_item,
        "replay_buffer_size": len(replay_buffer.buffer),
    })

    step += 1