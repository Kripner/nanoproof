import atexit
import os
import logging
import sys
import argparse
import faulthandler
import random
import signal
import time
from dataclasses import asdict

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
import leantree.augmentations
from leantree.core.lean import LeanGoal

from nanoproof.common import compute_init, compute_cleanup, get_base_dir, create_metrics_logger, add_logging_args, autodetect_device_type, SimpleTimer, flush, create_run_dirs, active_barrier, broadcast_value, enable_memory_profiling
from nanoproof.checkpoints import load_model, save_checkpoint, save_eval_results_to_run_dir
from nanoproof.engine import Engine
from nanoproof.data.sft.leantree import leantree_transitions
from nanoproof.data.sft.leantree_dataloader import rl_data_generator
from nanoproof.experience_collection import ReplayBuffer, TheoremsSampler, CollectedExperience, collection_dir, eval_dir
from nanoproof.prover import ProverWorker
from nanoproof.inference import setup_distributed_inference
from nanoproof.inference import TacticModel, BlockingTacticModel, compute_max_batch_prompt_tokens
from nanoproof.optim import optimizer_to_cpu, optimizer_to_gpu
from nanoproof.data.bench import minif2f
from nanoproof.data.check_init import read_lean_version
from nanoproof.data.bench import proofnet
from nanoproof.cli import create_monitor, configure_logging, log, log0, set_ddp_info, set_tactic_sink
from scripts.policy_eval import eval_tactic_accuracy, eval_critic_errors
from nanoproof.data.sft.leantree_dataloader import sft_data_generator

# TODO: in each episode, save a sample of training data (right before it goes into the model)

# TODO: matchmaker

# TODO: maybe log numbers of OR and AND nodes in the proof searches

# TODO: maybe migrate to newer Lean?

# TODO: the eval is now a bit unfair, since when the prover finds an invalid (e.g. self-referential) proof, it's not allowed to continue


# -----------------------------------------------------------------------------
# RL Hyperparameters
parser = argparse.ArgumentParser(description="RL training for nanoproof", allow_abbrev=False)

# General
add_logging_args(parser)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model-path", type=str, required=True, help="path to model_NNNNNN.pt to load from (relative to models/ or absolute)")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--load-buffer", type=str, default="", help="path to a previous RL run dir; at startup, seed the replay buffer with every transition found in its collection_*/collected.jsonl shards (FIFO-truncated to --replay-buffer-window-size). Independent of training resumption - does not affect the step counter or model checkpoint.")

# Infrastructure
parser.add_argument("--lean-servers", type=str, nargs="+", required=True, help="Lean server addresses (e.g., 10.10.25.33:8000 10.10.25.34); port defaults to 8000")
parser.add_argument("--lean-project", type=str, required=True, help="Path to the Lean project directory (contains lean-toolchain). The Lean version is read from this file and used to select per-dataset whitelists.")
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
parser.add_argument("--batch-time-limit", type=float, default=0.5)
parser.add_argument("--batch-max-gen-samples", type=int, default=None,
                    help="max generation samples per batch (default: num_actors * num_sampled_tactics)")
parser.add_argument("--batch-max-prompt-tokens", type=int, default=None,
                    help="max estimated prompt tokens per batch (default: auto from VRAM)")
parser.add_argument("--memory-profile", type=str, default=None,
                    help="if set, record CUDA memory history and dump snapshot to this dir on first OOM")

# Training
parser.add_argument("--device-batch-size", type=int, default=8)
parser.add_argument("--target-examples-per-step", type=int, default=512)
parser.add_argument("--num-updates-per-step", type=int, default=1, help="number of optimizer updates per training step (i.e. per collection cycle)")
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
parser.add_argument("--verbose", action="store_true", help="enable debug logging for inference and proving")
args = parser.parse_args()
user_config = vars(args).copy()

if args.verbose:
    logging.getLogger("nanoproof").setLevel(logging.DEBUG)


# -----------------------------------------------------------------------------
# Compute init

args.device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(args.device_type)
master_process = ddp_rank == 0
set_ddp_info(is_master=master_process, rank=ddp_rank)

# `kill -USR1 <pid>` on any rank dumps all-thread Python tracebacks to stderr.
faulthandler.register(signal.SIGUSR1, all_threads=True)

# Output directory init
log_dir, model_dir = create_run_dirs("rl", args.run, args_dict=user_config)
output_dir = log_dir
user_config["log_dir"] = log_dir
user_config["model_dir"] = model_dir

configure_logging(output_dir)

# metrics logging init
wandb_run = create_metrics_logger("nanoproof-rl", args, master_process, user_config, log_dir=log_dir, save_code=True)

# Enable memory profiling before model load so model weight allocations are captured.
if args.memory_profile:
    enable_memory_profiling(args.memory_profile)

# Create the policy/critic model.
inner_tactic_model = TacticModel.create(num_samples=args.num_sampled_tactics, model_path=args.model_path)
tactic_model = BlockingTacticModel(
    inner_model=inner_tactic_model,
    timeout_seconds=args.batch_time_limit,
    max_gen_samples=None,  # resolved after ProverWorker init
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

lean_version = read_lean_version(args.lean_project)
log0(f"Lean version: {lean_version} (from {args.lean_project}/lean-toolchain)", component="Config")

replay_buffer = ReplayBuffer(window_size=args.replay_buffer_window_size, seed=rank_seed)
if args.load_buffer:
    replay_buffer.load_from(args.load_buffer)
theorems_sampler = TheoremsSampler(seed=rank_seed, datasets=args.datasets, lean_version=lean_version)

# Set up distributed inference (starts servers on worker ranks, builds balancer on master)
balancer = setup_distributed_inference(tactic_model, args.inference_server_port)
if balancer:
    prover = ProverWorker(balancer, args.lean_servers)
    max_gen_samples = args.batch_max_gen_samples or prover.num_actors * args.num_sampled_tactics
    tactic_model.max_gen_samples = max_gen_samples
    log0(f"Batch max gen samples: {max_gen_samples} ({prover.num_actors} actors * {args.num_sampled_tactics} samples)", component="Config")
else:
    prover = None

# Prompt token limit for inference batches (prevents OOM on long prompts).
# Each rank computes its own limit based on its GPU's free VRAM rather than
# broadcasting from rank 0.  This is important because NCCL lazily allocates
# ~414 MiB per peer on some GPUs (topology-dependent), so different ranks
# can have different amounts of usable memory.
max_prompt_tokens = args.batch_max_prompt_tokens
if max_prompt_tokens is None:
    max_prompt_tokens = compute_max_batch_prompt_tokens(model.config, args.num_sampled_tactics, device)
    free_driver, _ = torch.cuda.mem_get_info(device)
    log0(f"Batch max prompt tokens: {max_prompt_tokens} (auto from {free_driver / 1024**3:.1f} GiB free, {torch.cuda.memory_allocated(device) / 1024**3:.1f} GiB allocated)", component="Config")
else:
    log0(f"Batch max prompt tokens: {max_prompt_tokens} (manual)", component="Config")
tactic_model.max_batch_prompt_tokens = max_prompt_tokens

# Broadcast max_gen_samples from master to worker ranks (workers don't have
# a ProverWorker to compute it from).
if ddp:
    tactic_model.max_gen_samples = broadcast_value(tactic_model.max_gen_samples)

# Create the RL monitor (master only)
rl_monitor = create_monitor(num_actors=0, enabled=master_process)
rl_monitor.set_output_dir(output_dir)
rl_monitor.set_lean_servers(args.lean_servers)
rl_monitor.set_replay_buffer_size(len(replay_buffer.buffer))

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
proofnet_results = None

def cleanup():
    """Cleanup function to ensure resources are released on shutdown."""
    log0("Shutting down...", component="Main")
    tactic_model.shutdown()
    log0("Shutdown complete", component="Main")

atexit.register(cleanup)

while True:
    timer = SimpleTimer()

    if step % args.eval_every == 0 and step >= args.eval_start:
        timer.start("eval")
        rl_monitor.record_phase_event("eval", "start")
        model.eval()
        rl_monitor.set_phase("evaluating")
        eval_experience = CollectedExperience()
        set_tactic_sink(eval_experience.record_tactic)

        # Policy evaluation (all ranks, uses DDP collectives internally)
        eval_steps = 200
        build_val_loader = lambda: sft_data_generator(mathlib_val, batch_size=args.device_batch_size)
        tactic_results = eval_tactic_accuracy(model, inner_tactic_model.tokenizer, build_val_loader(), eval_steps=eval_steps)
        critic_results = eval_critic_errors(model, inner_tactic_model.tokenizer, build_val_loader(), eval_steps=eval_steps)

        if master_process:
            log(f"Step {step:05d} | Tactic full acc: {tactic_results['full_acc']:.4%} | Tactic first acc: {tactic_results['first_token_acc']:.4%} | Critic argmax MSE: {critic_results['argmax_mse']:.4f} | Critic soft MSE: {critic_results['soft_mse']:.4f}", component="Eval")
            log(f"  Entropy - Tactic first: {tactic_results['first_token_entropy']:.4f} | Tactic all: {tactic_results['all_tokens_entropy']:.4f} | Critic: {critic_results['entropy']:.4f}", component="Eval")

        # Prover evaluation (rank 0 only).
        # Worker ranks poll via active_barrier so their inference servers stay responsive.
        if master_process:
            minif2f_theorems = minif2f.list_theorems(split="valid")
            proofnet_theorems = proofnet.list_theorems(split="valid")

            log(f"Evaluating on {len(minif2f_theorems)} theorems from MiniF2F", component="Eval")
            minif2f_results = prover.evaluate(minif2f_theorems, dataset_name="MiniF2F", num_simulations=args.num_simulations_eval)
            log(f"Evaluating on {len(proofnet_theorems)} theorems from ProofNet", component="Eval")
            proofnet_results = prover.evaluate(proofnet_theorems, dataset_name="ProofNet", num_simulations=args.num_simulations_eval)

            rl_monitor.record_eval(step, "MiniF2F", minif2f_results['success_rate'],
                                   minif2f_results['solved'], minif2f_results['total'], minif2f_results['errors'])
            rl_monitor.record_eval(step, "ProofNet", proofnet_results['success_rate'],
                                   proofnet_results['solved'], proofnet_results['total'],
                                   proofnet_results['errors'])

            minif2f_status = f"minif2f: {minif2f_results['success_rate']:.4%} ({minif2f_results['solved']}/{minif2f_results['total']}, errors={minif2f_results['errors']})"
            proofnet_status = f"proofnet: {proofnet_results['success_rate']:.4%} ({proofnet_results['solved']}/{proofnet_results['total']}, errors={proofnet_results['errors']})"
            log(f"Step {step:05d} | {minif2f_status} | {proofnet_status}", component="Eval")

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
                "proofnet_val": proofnet_results['success_rate'],
            }
            if minif2f_results['errors'] > 0:
                wandb_data["minif2f_errors"] = minif2f_results['errors']
            if proofnet_results['errors'] > 0:
                wandb_data["proofnet_errors"] = proofnet_results['errors']
            wandb_run.log(wandb_data)

            save_eval_results_to_run_dir(output_dir, step, "minif2f", minif2f_results)
            save_eval_results_to_run_dir(output_dir, step, "proofnet", proofnet_results)

        # Prover eval can take many minutes; no timeout (use SIGUSR1 to debug).
        active_barrier(f"prover_eval_{step}", timeout=None)

        set_tactic_sink(None)
        if master_process:
            eval_experience.save(eval_dir(output_dir, step))

        model.train()
        timer.end("eval")
        rl_monitor.record_phase_event("eval", "end")
        flush()

    if step % args.collect_every == 0:
        # Collect proofs (rank 0 only, worker ranks serve inference)
        timer.start("collect")
        rl_monitor.record_phase_event("collect", "start")
        experience = CollectedExperience()
        set_tactic_sink(experience.record_tactic)
        model.eval()
        rl_monitor.set_step(step)

        if master_process:
            prover.collect(theorems_sampler, args.collect_transitions, experience, num_simulations=args.num_simulations_collect)

        model.train()
        timer.end("collect")
        rl_monitor.record_phase_event("collect", "end")
        flush()

        # Park workers in a Python-level wait while master collects, so they
        # do not enter the NCCL broadcast below until master is also there.
        # Without this, workers block in NCCL for the entire collect duration
        # and trip the 10-min watchdog if collect ever takes longer.
        active_barrier(f"collect_{step}", timeout=None)

        # Rank 0 contributes its experience's transitions; workers pass [].
        replay_buffer.extend_and_sync(experience.transitions() if master_process else [])
        set_tactic_sink(None)

        if master_process:
            rl_monitor.set_replay_buffer_size(len(replay_buffer.buffer))
            experience.save(collection_dir(output_dir, step))

    if step % args.save_every == 0 and step > 0 and master_process:
        checkpoint_meta = {
            "step": step,
            "model_config": asdict(model.config),
        }
        if minif2f_results:
            checkpoint_meta["minif2f_val"] = minif2f_results['success_rate']
        if proofnet_results:
            checkpoint_meta["proofnet_val"] = proofnet_results['success_rate']
        save_checkpoint(
            model_dir,
            step,
            model.state_dict(),
            optimizer.state_dict(),
            checkpoint_meta,
            rank=ddp_rank,
        )

    timer.start("train")
    rl_monitor.record_phase_event("train", "start")
    rl_monitor.set_phase("training")

    # Pause inference across all ranks before touching the model for training.
    # The store-based barrier turns rank desyncs at this transition into a
    # diagnosable TimeoutError + traceback instead of a cryptic NCCL watchdog.
    tactic_model.pause()
    active_barrier(f"train_{step}/enter")

    optimizer_to_gpu(optimizer, device)

    total_loss = 0.0
    total_tokens = 0
    for _ in range(args.num_updates_per_step):
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

        total_loss += train_loss.item()
        total_tokens += num_tokens.item()

    optimizer_to_cpu(optimizer)
    flush()

    active_barrier(f"train_{step}/exit")
    tactic_model.resume()

    timer.end("train")
    rl_monitor.record_phase_event("train", "end")

    mean_loss = total_loss / args.num_updates_per_step
    rl_monitor.update_training(step, mean_loss, total_tokens)
    if master_process:
        log(f"Step {step:05d} | Training loss: {mean_loss:.6f} | num_tokens: {total_tokens:,} | replay_buffer_size: {len(replay_buffer.buffer)}", component="Train")
    wandb_run.log({
        "step": step,
        "train_loss": mean_loss,
        "num_tokens": total_tokens,
        "replay_buffer_size": len(replay_buffer.buffer),
        **{f"time/{k}": v for k, v in timer.get_times().items()},
        **rl_monitor.lean_server_metrics(),
    })

    step += 1
