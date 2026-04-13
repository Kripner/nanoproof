import atexit
import os
import json
import sys
import argparse
import random
import time
from dataclasses import asdict

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import wandb
import torch
import torch.distributed as dist
import leantree.augmentations
from leantree.core.lean import LeanGoal

from nanoproof.common import compute_init, compute_cleanup, get_base_dir, DummyWandb, autodetect_device_type, SimpleTimer, flush, create_run_dirs, active_barrier_master, active_barrier_wait
from nanoproof.checkpoints import load_model, save_checkpoint, save_eval_results_to_run_dir
from nanoproof.engine import Engine
from nanoproof.data.sft.leantree import leantree_transitions
from nanoproof.data.sft.leantree_dataloader import rl_data_generator
from nanoproof.replay_buffer import ReplayBuffer
from nanoproof.prover import TheoremsSampler, Prover, setup_distributed_inference
from nanoproof.inference import TacticModel, BlockingTacticModel
from nanoproof.optim import optimizer_to_cpu, optimizer_to_gpu
from nanoproof.data.bench import minif2f
from nanoproof.data.rl import leanworkbook
from nanoproof.cli import create_monitor, configure_logging, log, log0, set_ddp_info
from scripts.policy_eval import eval_tactic_accuracy, eval_critic_errors
from nanoproof.data.sft.leantree_dataloader import sft_data_generator

# TODO: in each episode, save a sample of training data (right before it goes into the model)

# TODO: matchmaker

# TODO: maybe log numbers of OR and AND nodes in the proof searches

# TODO: maybe migrate to newer Lean?

# TODO: the eval is now a bit unfair, since when the prover finds an invalid (e.g. self-referential) proof, it's not allowed to continue


# -----------------------------------------------------------------------------
# RL Hyperparameters
parser = argparse.ArgumentParser(description="RL training for nanoproof")

# General
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model-path", type=str, required=True, help="path to model_NNNNNN.pt to load from (relative to models/ or absolute)")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--resume-from", type=str, default="")

# Infrastructure
parser.add_argument("--lean-servers", type=str, nargs="+", required=True, help="Lean server addresses (e.g., 10.10.25.33:8000 10.10.25.34); port defaults to 8000")
parser.add_argument("--inference-server-port", type=int, default=5000, help="base port for per-rank inference servers (rank N uses port base+1+N)")

# Search / collection
ALL_DATASETS = ["leanworkbook", "deepseek_prover", "numinamath"]
parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS, help="which theorem datasets to sample from (default: all three)")
parser.add_argument("--num-sampled-tactics", type=int, default=6)
parser.add_argument("--num-simulations-collect", type=int, default=50)
parser.add_argument("--num-simulations-eval", type=int, default=50)
parser.add_argument("--collect-every", type=int, default=1)
parser.add_argument("--collect-transitions", type=int, default=100)
parser.add_argument("--replay-buffer-window-size", type=int, default=60_000_000)
parser.add_argument("--batch-timeout", type=float, default=0.2)
parser.add_argument("--max-batch-tokens", type=int, default=8000)

# Training
parser.add_argument("--device-batch-size", type=int, default=8)
parser.add_argument("--target-examples-per-step", type=int, default=512)
parser.add_argument("--fraction-sft", type=float, default=0.2)
parser.add_argument("--augment-data", type=bool, default=True)
parser.add_argument("--value-weight", type=float, default=0.01)

# Optimizer
parser.add_argument("--unembedding-lr", type=float, default=0.004)
parser.add_argument("--embedding-lr", type=float, default=0.2)
parser.add_argument("--matrix-lr", type=float, default=0.02)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--init-lr-frac", type=float, default=0.02)

# Evaluation / checkpointing
parser.add_argument("--eval-every", type=int, default=100)
parser.add_argument("--eval-start", type=int, default=0)
parser.add_argument("--save-every", type=int, default=500)
args = parser.parse_args()
user_config = vars(args).copy()


# -----------------------------------------------------------------------------
# Compute init

args.device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(args.device_type)
master_process = ddp_rank == 0
set_ddp_info(is_master=master_process, rank=ddp_rank)

# Output directory init
log_dir, model_dir = create_run_dirs("rl", args.run, args_dict=user_config)
output_dir = log_dir
user_config["log_dir"] = log_dir
user_config["model_dir"] = model_dir

if master_process:
    configure_logging(output_dir)

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanoproof-rl", name=args.run, dir=log_dir, config=user_config, save_code=True)

# Create the policy/critic model.
inner_tactic_model = TacticModel.create(num_samples=args.num_sampled_tactics, model_path=args.model_path)
tactic_model = BlockingTacticModel(
    inner_model=inner_tactic_model,
    timeout_seconds=args.batch_timeout,
    max_batch_tokens=args.max_batch_tokens,
)
model = tactic_model.network

# -----------------------------------------------------------------------------
# DataLoader

examples_per_step = args.device_batch_size * ddp_world_size
log0(f"Target examples per step: {args.target_examples_per_step}", component="Config")
log0(f"Device batch size: {args.device_batch_size}", component="Config")
log0(f"Examples per step is device_batch_size * ddp_world_size: {examples_per_step}", component="Config")
assert args.target_examples_per_step % examples_per_step == 0, "Target examples per step must be divisible by examples per step"
grad_accum_steps = args.target_examples_per_step // examples_per_step
log0(f"=> Setting grad accum steps: {grad_accum_steps}", component="Config")

rank_seed = args.seed + ddp_rank

replay_buffer = ReplayBuffer(window_size=args.replay_buffer_window_size, seed=rank_seed)
theorems_sampler = TheoremsSampler(seed=rank_seed, datasets=args.datasets)

# Set up distributed inference (starts servers on worker ranks, builds balancer on master)
lean_server_addrs = [s if ":" in s else f"{s}:8000" for s in args.lean_servers]
balancer = setup_distributed_inference(tactic_model, args.inference_server_port)
prover = Prover(balancer, lean_server_addrs) if balancer else None

# Create the RL monitor (master only)
rl_monitor = create_monitor(num_actors=0, enabled=master_process)
rl_monitor.set_output_dir(output_dir)
rl_monitor.set_lean_servers(lean_server_addrs)

# Augmentations
shuffle_goals_and_hypotheses = leantree.augmentations.ShuffleGoalsAndHypotheses(seed=args.seed)
random_rename = leantree.augmentations.RandomRename(seed=args.seed)

mathlib_train = list(leantree_transitions(
    split="train",
    augmentations=[shuffle_goals_and_hypotheses, random_rename] if args.augment_data else None,
))
random.Random(rank_seed).shuffle(mathlib_train)
mathlib_val = list(leantree_transitions(split="valid"))

def augment(state_str, tactic_str):
    try:
        goals = [LeanGoal.from_string(goal_str) for goal_str in state_str.split("\n\n")]
    except Exception as e:
        print(f"Error parsing goals: {e}")
        print(f"State: {state_str}")
        raise e
    goals = shuffle_goals_and_hypotheses.run_on_goals(goals)
    goals, tactic = random_rename.run_on_goals(goals, tactic_str)

    state_str = "\n\n".join([str(goal) for goal in goals])
    tactic_str = tactic

    return state_str, tactic_str

# We train on the collected transitions, with some portion of LeanTree Mathlib transitions mixed in.

def train_generator():
    rng = random.Random(rank_seed)
    mathlib_iter = iter(mathlib_train)
    while True:
        assert len(replay_buffer.buffer) >= args.collect_transitions
        if rng.random() < args.fraction_sft:
            try:
                state, tactic, proof_depth = next(mathlib_iter)
            except StopIteration:
                mathlib_iter = iter(mathlib_train)
                state, tactic, proof_depth = next(mathlib_iter)
        else:
            state, tactic, value_target = replay_buffer.sample_transition()
            proof_depth = -value_target
            # Only run augmentations on replay buffer data - Mathlib data is already augmented.
            if args.augment_data:
                state, tactic = augment(state, tactic)

        yield state, tactic, proof_depth


train_loader = rl_data_generator(train_generator(), batch_size=args.device_batch_size)
value_delim_tok = inner_tactic_model.tokenizer.encode_special("<|value|>")

# -----------------------------------------------------------------------------
# Initialize the Optimizer

optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac

# Note: optimizer state is lazy-initialized by PyTorch on the first step().
# optimizer_to_cpu is called after the first step to offload it.

# Wait for all ranks to be ready
if ddp:
    dist.barrier()

# Go!
step = 0
minif2f_results = None
leanworkbook_results = None

def cleanup():
    """Cleanup function to ensure resources are released on shutdown."""
    log0("Shutting down...", component="Main")
    if prover is not None:
        prover.shutdown()
    log0("Shutdown complete", component="Main")

atexit.register(cleanup)

while True:
    timer = SimpleTimer()

    if step % args.eval_every == 0 and step >= args.eval_start:
        timer.start("eval")
        model.eval()
        rl_monitor.set_phase("evaluating")

        # Policy evaluation (all ranks — uses DDP collectives internally)
        eval_steps = 200
        build_val_loader = lambda: sft_data_generator(mathlib_val, batch_size=args.device_batch_size)
        tactic_results = eval_tactic_accuracy(model, inner_tactic_model.tokenizer, build_val_loader(), eval_steps=eval_steps)
        critic_results = eval_critic_errors(model, inner_tactic_model.tokenizer, build_val_loader(), eval_steps=eval_steps)

        if master_process:
            log(f"Step {step:05d} | Tactic full acc: {tactic_results['full_acc']:.4%} | Tactic first acc: {tactic_results['first_token_acc']:.4%} | Critic argmax MSE: {critic_results['argmax_mse']:.4f} | Critic soft MSE: {critic_results['soft_mse']:.4f}", component="Eval")
            log(f"  Entropy - Tactic first: {tactic_results['first_token_entropy']:.4f} | Tactic all: {tactic_results['all_tokens_entropy']:.4f} | Critic: {critic_results['entropy']:.4f}", component="Eval")

        # Prover evaluation + collection (rank 0 only).
        # Worker ranks poll via active_barrier so their inference servers stay responsive.
        if master_process:
            minif2f_theorems = minif2f.list_theorems(split="valid")
            leanworkbook_theorems = leanworkbook.list_theorems(split="valid")[:128]

            log(f"Evaluating on {len(minif2f_theorems)} theorems from MiniF2F", component="Eval")
            minif2f_results = prover.evaluate(minif2f_theorems, dataset_name="MiniF2F", num_simulations=args.num_simulations_eval)
            log(f"Evaluating on {len(leanworkbook_theorems)} theorems from LeanWorkBook", component="Eval")
            leanworkbook_results = prover.evaluate(leanworkbook_theorems, dataset_name="LeanWorkBook", num_simulations=args.num_simulations_eval)

            rl_monitor.record_eval(step, "MiniF2F", minif2f_results['success_rate'],
                                   minif2f_results['solved'], minif2f_results['total'], minif2f_results['errors'])
            rl_monitor.record_eval(step, "LeanWorkBook", leanworkbook_results['success_rate'],
                                   leanworkbook_results['solved'], leanworkbook_results['total'],
                                   leanworkbook_results['errors'])

            minif2f_status = f"minif2f: {minif2f_results['success_rate']:.4%} ({minif2f_results['solved']}/{minif2f_results['total']}, errors={minif2f_results['errors']})"
            leanworkbook_status = f"leanworkbook: {leanworkbook_results['success_rate']:.4%} ({leanworkbook_results['solved']}/{leanworkbook_results['total']}, errors={leanworkbook_results['errors']})"
            log(f"Step {step:05d} | {minif2f_status} | {leanworkbook_status}", component="Eval")

            wandb_data = {
                "step": step,
                "val_full_acc": tactic_results["full_acc"],
                "val_first_token_acc": tactic_results["first_token_acc"],
                "val_first_token_entropy": tactic_results["first_token_entropy"],
                "val_all_tokens_entropy": tactic_results["all_tokens_entropy"],
                "val_critic_argmax_mse": critic_results["argmax_mse"],
                "val_critic_soft_mse": critic_results["soft_mse"],
                "val_critic_entropy": critic_results["entropy"],
                "minif2f_val": minif2f_results['success_rate'],
                "leanworkbook_val": leanworkbook_results['success_rate'],
            }
            if minif2f_results['errors'] > 0:
                wandb_data["minif2f_errors"] = minif2f_results['errors']
            if leanworkbook_results['errors'] > 0:
                wandb_data["leanworkbook_errors"] = leanworkbook_results['errors']
            wandb_run.log(wandb_data)

            save_eval_results_to_run_dir(output_dir, step, "minif2f", minif2f_results)
            save_eval_results_to_run_dir(output_dir, step, "leanworkbook", leanworkbook_results)

        model.train()
        timer.end("eval")
        flush()

    if step % args.collect_every == 0:
        # Check if we can resume from a previous run's replay buffer
        resume_file = os.path.join(args.resume_from, f"replay_buffer_{step:05d}.jsonl") if args.resume_from else None
        if resume_file and os.path.exists(resume_file):
            if master_process:
                log(f"Loading replay buffer at step {step} from {resume_file}", component="Main")
            with open(resume_file, "r") as f:
                replay_buffer.buffer = [
                    (obj["context"], obj["tactic"], obj["value_target"])
                    for line in f if line.strip()
                    for obj in [json.loads(line)]
                ]
            rl_monitor.set_replay_buffer_size(len(replay_buffer.buffer))
            if master_process:
                log(f"Loaded {len(replay_buffer.buffer)} transitions at step {step} from previous run", component="Main")
            flush()
            if ddp:
                dist.barrier()
        else:
            # Collect proofs (rank 0 only — worker ranks serve inference)
            timer.start("collect")
            model.eval()
            rl_monitor.set_step(step)

            if master_process:
                prover.collect(theorems_sampler, args.collect_transitions, replay_buffer, num_simulations=args.num_simulations_collect)

            model.train()
            timer.end("collect")
            flush()

            # Broadcast replay buffer from rank 0 to all ranks
            replay_buffer.synchronize()

            if master_process:
                rl_monitor.set_replay_buffer_size(len(replay_buffer.buffer))
                with open(os.path.join(output_dir, f"replay_buffer_{step:05d}.jsonl"), "w") as f:
                    for context, tactic, value_target in replay_buffer.buffer:
                        f.write(json.dumps({"context": context, "tactic": tactic, "value_target": value_target}) + "\n")

    if step % args.save_every == 0 and step > 0 and master_process:
        checkpoint_meta = {
            "step": step,
            "model_config": asdict(model.config),
        }
        if minif2f_results:
            checkpoint_meta["minif2f_val"] = minif2f_results['success_rate']
        if leanworkbook_results:
            checkpoint_meta["leanworkbook_val"] = leanworkbook_results['success_rate']
        save_checkpoint(
            model_dir,
            step,
            model.state_dict(),
            optimizer.state_dict(),
            checkpoint_meta,
            rank=ddp_rank,
        )

    timer.start("train")
    rl_monitor.set_phase("training")

    # Pause inference during training to prevent concurrent model access.
    if master_process and prover is not None:
        prover.pause()

    # Move optimizer state to GPU for training
    optimizer_to_gpu(optimizer, device)

    # Evaluate the gradient
    num_tokens = torch.tensor(0, device=device)
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_loader)
        per_token_loss = model(train_inputs, train_targets, loss_reduction='none')  # (B*T,)
        per_token_loss = per_token_loss.view(train_inputs.shape)  # (B, T)

        is_value_sample = (train_inputs == value_delim_tok).any(dim=1)  # (B,)
        sample_weights = torch.where(is_value_sample, args.value_weight, 1.0)  # (B,)

        token_mask = (train_targets >= 0)  # (B, T)
        weighted_token_loss = per_token_loss * sample_weights.unsqueeze(1)  # (B, T)

        loss = (weighted_token_loss * token_mask).sum() / token_mask.sum()
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        num_tokens += (train_targets >= 0).sum()
    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)

    optimizer.step()
    model.zero_grad(set_to_none=True)

    # Offload optimizer state back to CPU, resume inference
    optimizer_to_cpu(optimizer)
    flush()
    if master_process and prover is not None:
        prover.resume()

    timer.end("train")

    # Logging
    train_loss_item = train_loss.item()
    num_tokens_item = num_tokens.item()
    rl_monitor.update_training(step, train_loss_item, num_tokens_item)
    if master_process:
        log(f"Step {step:05d} | Training loss: {train_loss_item:.6f} | num_tokens: {num_tokens_item:,} | replay_buffer_size: {len(replay_buffer.buffer)}", component="Train")
    wandb_run.log({
        "step": step,
        "train_loss": train_loss_item,
        "num_tokens": num_tokens_item,
        "replay_buffer_size": len(replay_buffer.buffer),
        **{f"time/{k}": v for k, v in timer.get_times().items()}
    })

    step += 1
