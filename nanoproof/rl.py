import os
from datetime import datetime
import json
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random

import wandb
import torch
import torch.distributed as dist
from contextlib import nullcontext

from nanoproof.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb, autodetect_device_type, \
    SimpleTimer
from nanoproof.checkpoints import load_model, save_checkpoint
from nanoproof.engine import Engine
from nanoproof.data.leantree import iter_data
from nanoproof.data.leantree_dataloader import rl_data_generator
from nanoproof.experience_collection import ReplayBuffer, TheoremsSampler, Config, run_actor
from nanoproof.search import TacticModel, BatchedTacticModel
from nanoproof.data import minif2f
from nanoproof.data import leanworkbook
from nanoproof.cli import create_monitor, configure_logging, log
from scripts.prover_eval import eval_success_rate

# TODO: if tactic application results in a state that already is on the path from root, skip the tactic (otherwise we sometimes get stuck in loop of eg. rw [add_comm])

"""
Timer results:                                                                                                                                                                   
  expand : 5297.9417s (67.0%)                                                                                                                                                    
  sample : 2612.2757s (33.0%)
"""

# TODO: save all proofs found during evaluation
# TODO: (maybe) try removing each tactic and if the proof is still valid, do not add the transition to the replay buffer
#   ... however, then we need to be sure to update the proof states
# TODO: matchmaker

# -----------------------------------------------------------------------------
# RL Hyperparameters
run = "dummy"  # wandb run name default ("dummy" is special - we won't log to wandb)
seed = 0
# compute/precision
device_type = ""  # cuda|cpu|mps (empty => autodetect)
dtype = "bfloat16"
device_batch_size = 8  # (maybe) max to avoid OOM (on A100 40GB)
# data
fraction_sft = 0.1  # 10% of data will come from Mathlib (leantree), 90% from replay buffer
collect_every = 1  # how many steps to train between RL data collections
collect_transitions = 100  # how many proof transitions to collect in one collection
# parallel experience collection
num_actors = 32  # number of parallel actor threads for experience collection
num_sampled_tactics = 6  # number of tactics to sample per state in MCTS
batch_timeout = 0.1  # timeout in seconds for batching LLM calls
# optimization
num_epochs = 1
num_iterations = -1  # override number of iterations (-1 = disable, use num_epochs to derive it)
target_examples_per_step = 512
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.02
# evaluation and logging there of
eval_every = 5  # TODO: when eval_every>1, we need some warmup (collect eval_every*collect_transitions)
# eval_metrics_every = 200
sample_every = 100
eval_metrics_max_problems = 1024
save_every = 1000
# now allow CLI to override the settings via the configurator lol
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanoproof', 'configurator.py')).read())  # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys}  # possibly useful for logging
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# Output directory init

if master_process:
    base_dir = get_base_dir()
    timestamp = datetime.now().strftime("%y-%m-%d_%H-%M")
    output_dirname = f"{timestamp}-{run}"
    output_dir = os.path.join(base_dir, "rl", output_dirname)
    if os.path.exists(output_dir):
        print0(f"Error: Output directory {output_dir} already exists")
        if ddp:
            dist.destroy_process_group()
        sys.exit(1)
    os.makedirs(output_dir)
    print0(f"Output directory: {output_dir}")
else:
    output_dir = None

if ddp:
    output_dir_list = [output_dir]
    dist.broadcast_object_list(output_dir_list, src=0)
    output_dir = output_dir_list[0]

# Configure file-based logging (only on master process to avoid duplicate writes)
if master_process:
    configure_logging(output_dir)

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanoproof-rl", name=run, config=user_config, save_code=True)

# Create the tactic model with batching support for parallel actors
inner_tactic_model = TacticModel.create(num_samples=num_sampled_tactics)
tactic_model = BatchedTacticModel(
    inner_model=inner_tactic_model,
    batch_size=num_actors,
    timeout_seconds=batch_timeout
)
model = tactic_model.network

# print0(f"Target examples per step: {target_examples_per_step}")
# print0(f"Collect every: {collect_every}")
# collect_transitions = target_examples_per_step * collect_every
# print0(f"=> Setting collect_transitions: {collect_transitions}")

# -----------------------------------------------------------------------------
# DataLoader

examples_per_step = device_batch_size * ddp_world_size
log(f"Target examples per step: {target_examples_per_step}", component="Config")
log(f"Device batch size: {device_batch_size}", component="Config")
log(f"Examples per step is device_batch_size * ddp_world_size: {examples_per_step}", component="Config")
assert target_examples_per_step % examples_per_step == 0, "Target examples per step must be divisible by examples per step"
grad_accum_steps = target_examples_per_step // examples_per_step
log(f"=> Setting grad accum steps: {grad_accum_steps}", component="Config")

rank_seed = seed + ddp_rank
mathlib_train = list(iter_data(split="train"))
random.Random(rank_seed).shuffle(mathlib_train)

config = Config(num_actors=num_actors, num_sampled_tactics=num_sampled_tactics)
replay_buffer = ReplayBuffer(config, seed=rank_seed)
theorems_sampler = TheoremsSampler(seed=rank_seed)

# Create the RL monitor (only show on master process)
rl_monitor = create_monitor(num_actors=num_actors, enabled=master_process)


def train_generator():
    rng = random.Random(rank_seed)
    mathlib_iter = iter(mathlib_train)
    while True:
        assert len(replay_buffer.buffer) > 100
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
timer = SimpleTimer()
while True:
    if step % collect_every == 0:
        # collect proofs
        timer.start("collect")
        model.eval()
        rl_monitor.start_collection(target_samples=collect_transitions, num_actors=num_actors)
        rl_monitor.set_step(step)
        run_actor(collect_transitions, config, tactic_model, replay_buffer, theorems_sampler)
        model.train()
        timer.end("collect")
        replay_buffer.synchronize()
        rl_monitor.set_replay_buffer_size(len(replay_buffer.buffer))
        rl_monitor.display()
        with open(os.path.join(output_dir, f"replay_buffer_{step:05d}.json"), "w") as f:
            json.dump(replay_buffer.buffer, f)

    if step % eval_every == 0:
        timer.start("eval")
        model.eval()
        rl_monitor.set_phase("evaluating")

        minif2f_theorems = minif2f.list_theorems(split="Valid")
        minif2f_theorems = minif2f_theorems[:64]
        log(f"Evaluating on {len(minif2f_theorems)} theorems from MiniF2F", component="Eval")
        # Use inner model for evaluation (sequential, no batching needed)
        minif2f_results = eval_success_rate(inner_tactic_model, minif2f_theorems)
        rl_monitor.record_eval(step, "MiniF2F", minif2f_results['success_rate'],
                               minif2f_results['solved'], minif2f_results['total'], minif2f_results['errors'])

        leanworkbook_theorems = leanworkbook.list_theorems(split="val")
        leanworkbook_theorems = leanworkbook_theorems[:64]
        log(f"Evaluating on {len(leanworkbook_theorems)} theorems from LeanWorkBook", component="Eval")
        leanworkbook_results = eval_success_rate(inner_tactic_model, leanworkbook_theorems)
        rl_monitor.record_eval(step, "LeanWorkBook", leanworkbook_results['success_rate'],
                               leanworkbook_results['solved'], leanworkbook_results['total'],
                               leanworkbook_results['errors'])

        log(f"Step {step:05d} | minif2f: {minif2f_results['success_rate']:.4%} ({minif2f_results['solved']}/{minif2f_results['total']}, errors={minif2f_results['errors']}) | leanworkbook: {leanworkbook_results['success_rate']:.4%} ({leanworkbook_results['solved']}/{leanworkbook_results['total']}, errors={leanworkbook_results['errors']})", component="Eval")
        wandb_run.log({
            "step": step,
            "minif2f_val": minif2f_results['success_rate'],
            "leanworkbook_val": leanworkbook_results['success_rate'],
        })
        model.train()
        timer.end("eval")
        rl_monitor.display()

    if step > 0 and step % save_every == 0 and master_process:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        model_config_kwargs = model.config.__dict__  # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            [opt.state_dict() for opt in optimizers],  # optimizer states
            {
                "step": step,
                "model_config": model_config_kwargs,
                "minif2f_val": minif2f_results['success_rate'],
                "leanworkbook_val": leanworkbook_results['success_rate'],
            },
            rank=ddp_rank,
        )

    timer.start("train")
    rl_monitor.set_phase("training")
    # evaluate the gradient
    num_tokens = torch.tensor(0, device=device)  # the number of "active" tokens of supervision seen
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(
            train_loader)  # prefetch the next batch while the GPU is busy with forward/backward
        with autocast_ctx:
            loss = model(train_inputs, train_targets)
        train_loss = loss.detach()  # for logging
        loss = loss / grad_accum_steps  # each .backward() is a grad sum => normalize loss here
        loss.backward()  # accumulate the gradient
        num_tokens += (train_targets >= 0).sum()
    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)  # sum over ranks

    # step the optimizers
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    timer.end("train")

    # logging
    train_loss_item = train_loss.item()
    num_tokens_item = num_tokens.item()
    rl_monitor.update_training(step, train_loss_item, num_tokens_item)
    rl_monitor.display()
    log(f"Step {step:05d} | Training loss: {train_loss_item:.6f} | num_tokens: {num_tokens_item:,} | replay_buffer_size: {len(replay_buffer.buffer)}", component="Train")
    wandb_run.log({
        "step": step,
        "train_loss": train_loss_item,
        "num_tokens": num_tokens_item,
        "replay_buffer_size": len(replay_buffer.buffer),
        **{f"time/{k}": v for k, v in timer.get_times().items()}
    })

    step += 1
