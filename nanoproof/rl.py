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

from nanoproof.common import compute_init, compute_cleanup, get_base_dir, DummyWandb, autodetect_device_type, SimpleTimer, flush, create_run_dirs
from nanoproof.checkpoints import load_model, save_checkpoint
from nanoproof.engine import Engine
from nanoproof.search import SearchConfig
from nanoproof.data.sft.leantree import leantree_transitions
from nanoproof.data.sft.leantree_dataloader import rl_data_generator
from nanoproof.replay_buffer import ReplayBuffer
from nanoproof.prover import TheoremsSampler, LocalProver, DistributedProver, Prover
from nanoproof.inference import TacticModel, BlockingTacticModel, start_inference_server
from nanoproof.data.bench import minif2f
from nanoproof.data.rl import leanworkbook
from nanoproof.cli import create_monitor, configure_logging, log, log0, set_ddp_info
from nanoproof.rl_server import start_coordinator
from nanoproof.infra import load_infra_config, InfraConfig, parse_lean_server
from scripts.prover_eval import save_eval_results_to_run_dir
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
parser.add_argument("--infra-file", type=str, default="infra-ms.toml", help="path to infra.toml (empty = local mode)")
parser.add_argument("--lean-server", type=str, default="10.10.25.31:8000")
parser.add_argument("--inference-server-port", type=int, default=5000)
parser.add_argument("--poll-interval", type=float, default=3.0)

# Search / collection
parser.add_argument("--num-actors", type=int, default=32)
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


distributed = bool(args.infra_file)
infra_config: InfraConfig | None = None

if distributed:
    if not os.path.exists(args.infra_file):
        log(f"Error: Infrastructure file {args.infra_file} does not exist", component="Main")
        sys.exit(1)
    infra_config = load_infra_config(args.infra_file)
    # Override inference_server_port from infra config
    args.inference_server_port = infra_config.rl_server_port
    # Get lean servers for monitoring
    lean_servers = infra_config.get_lean_server_list()
    # Lean server for local actors not used in distributed mode
    local_lean_server = parse_lean_server("localhost:8000")
    # Warn if num_simulations is changed from default - must be set in prover_server.py
    if args.num_simulations_collect != 50:
        print(f"WARNING: num_simulations_collect={args.num_simulations_collect} is ignored in distributed mode. "
              f"Set --num-simulations in prover_server.py instead.")
    if args.num_simulations_eval != 50:
        print(f"WARNING: num_simulations_eval={args.num_simulations_eval} is ignored in distributed mode. "
              f"Set --num-simulations in prover_server.py instead.")
else:
    # Local mode: parse lean_server and use it for both actors and monitoring
    local_lean_server = parse_lean_server(args.lean_server)
    lean_servers = [str(local_lean_server)]

# -----------------------------------------------------------------------------

# Compute init
args.device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(args.device_type)
master_process = ddp_rank == 0
set_ddp_info(is_master=master_process, rank=ddp_rank)
# Model handles dtype internally via Linear class, no autocast needed

# Output directory init
log_dir, model_dir = create_run_dirs("rl", args.run, args_dict=user_config)
output_dir = log_dir  # RL logs, replay buffers, eval results go in log_dir
user_config["log_dir"] = log_dir
user_config["model_dir"] = model_dir

# Configure file-based logging (only on master process to avoid duplicate writes)
if master_process:
    configure_logging(output_dir)

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanoproof-rl", name=args.run, dir=log_dir, config=user_config, save_code=True)

log0(f"Distributed mode: {distributed}", component="Config")
# Create the policy/critic model.
inner_tactic_model = TacticModel.create(num_samples=args.num_sampled_tactics, model_path=args.model_path)
# BlockingTacticModel blocks asynchronous requests to create batches for inference.
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

search_config = SearchConfig(
    num_actors=args.num_actors,
    num_simulations=args.num_simulations_collect,
)
replay_buffer = ReplayBuffer(window_size=args.replay_buffer_window_size, seed=rank_seed)
theorems_sampler = TheoremsSampler(seed=rank_seed)

# Create the RL monitor (enabled only on master process - no-op for others)
rl_monitor = create_monitor(num_actors=args.num_actors, enabled=master_process)
rl_monitor.set_output_dir(output_dir)
rl_monitor.set_lean_servers(lean_servers)

# Augmentations. Run directly on LeanTree Mathlib theorems where we have the structure. Collected transitions
# have to be parsed first (the augment method).
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
value_delim_tok = inner_tactic_model.tokenizer.encode_special("<|value|>")  # for distinguishing policy vs value samples

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


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

# Construct the Prover. In distributed mode, also bring up the rank-local
# inference server and (on master) the coordinator before constructing it.
prover: Prover
if distributed:
    # Each rank starts an inference server on its own port
    inference_port = args.inference_server_port + 1 + ddp_rank  # ports 5001, 5002, ...
    inference_server_thread = start_inference_server(tactic_model, inference_port)

    # Sync to ensure all inference servers are up before coordinator starts
    if ddp:
        dist.barrier()

    # Master also starts the coordinator (proxies inference + handles registration)
    if master_process:
        inference_endpoints = [f"http://127.0.0.1:{args.inference_server_port + 1 + r}" for r in range(ddp_world_size)]
        coordinator_thread, inference_router = start_coordinator(
            args.inference_server_port, inference_endpoints, startup_timeout=30.0
        )

    # Sync again so all ranks wait for coordinator
    if ddp:
        dist.barrier()

    prover = DistributedProver(
        inference_model=tactic_model,
        poll_interval=args.poll_interval,
    )
else:
    prover = LocalProver(
        config=search_config,
        tactic_model=tactic_model,
        lean_address=local_lean_server.address,
        lean_port=local_lean_server.port,
        num_simulations_eval=args.num_simulations_eval,
    )

# Go!
step = 0
minif2f_results = None
leanworkbook_results = None

# Register cleanup to run on exit (handles normal exit, sys.exit, and unhandled exceptions)
def cleanup():
    """Cleanup function to ensure resources are released on shutdown."""
    log0("Shutting down...", component="Main")
    prover.shutdown()
    log0("Shutdown complete", component="Main")

atexit.register(cleanup)

while True:
    timer = SimpleTimer()

    if step % args.eval_every == 0 and step >= args.eval_start:
        timer.start("eval")
        model.eval()
        rl_monitor.set_phase("evaluating")

        # Policy evaluation (tactic accuracy and critic errors on mathlib val)
        eval_steps = 200
        build_val_loader = lambda: sft_data_generator(mathlib_val, batch_size=args.device_batch_size)
        tactic_results = eval_tactic_accuracy(model, inner_tactic_model.tokenizer, build_val_loader(), eval_steps=eval_steps)
        critic_results = eval_critic_errors(model, inner_tactic_model.tokenizer, build_val_loader(), eval_steps=eval_steps)
        
        if master_process:
            log(f"Step {step:05d} | Tactic full acc: {tactic_results['full_acc']:.4%} | Tactic first acc: {tactic_results['first_token_acc']:.4%} | Critic argmax MSE: {critic_results['argmax_mse']:.4f} | Critic soft MSE: {critic_results['soft_mse']:.4f}", component="Eval")
            log(f"  Entropy - Tactic first: {tactic_results['first_token_entropy']:.4f} | Tactic all: {tactic_results['all_tokens_entropy']:.4f} | Critic: {critic_results['entropy']:.4f}", component="Eval")

        # Load eval theorems
        minif2f_theorems = minif2f.list_theorems(split="valid")
        leanworkbook_theorems = leanworkbook.list_theorems(split="valid")[:128]

        log(f"Evaluating on {len(minif2f_theorems)} theorems from MiniF2F", component="Eval")
        minif2f_results = prover.evaluate(minif2f_theorems, dataset_name="MiniF2F")
        log(f"Evaluating on {len(leanworkbook_theorems)} theorems from LeanWorkBook", component="Eval")
        leanworkbook_results = prover.evaluate(leanworkbook_theorems, dataset_name="LeanWorkBook")

        if master_process:
            rl_monitor.record_eval(step, "MiniF2F", minif2f_results['success_rate'],
                                   minif2f_results['solved'], minif2f_results['total'], minif2f_results['errors'])
            rl_monitor.record_eval(step, "LeanWorkBook", leanworkbook_results['success_rate'],
                                   leanworkbook_results['solved'], leanworkbook_results['total'],
                                   leanworkbook_results['errors'])

            # Log results (with timeout/invalid warning if applicable)
            minif2f_status = f"minif2f: {minif2f_results['success_rate']:.4%} ({minif2f_results['solved']}/{minif2f_results['total']}, errors={minif2f_results['errors']})"
            if minif2f_results.get('timed_out'):
                minif2f_status += " [TIMED OUT]"
            if minif2f_results.get('invalid'):
                minif2f_status += " [INVALID]"
            leanworkbook_status = f"leanworkbook: {leanworkbook_results['success_rate']:.4%} ({leanworkbook_results['solved']}/{leanworkbook_results['total']}, errors={leanworkbook_results['errors']})"
            if leanworkbook_results.get('timed_out'):
                leanworkbook_status += " [TIMED OUT]"
            if leanworkbook_results.get('invalid'):
                leanworkbook_status += " [INVALID]"
            log(f"Step {step:05d} | {minif2f_status} | {leanworkbook_status}", component="Eval")

            # Only log scores to wandb if eval completed (not timed out or invalid)
            wandb_data = {
                "step": step,
                # Policy evaluation metrics
                "val_full_acc": tactic_results["full_acc"],
                "val_first_token_acc": tactic_results["first_token_acc"],
                "val_first_token_entropy": tactic_results["first_token_entropy"],
                "val_all_tokens_entropy": tactic_results["all_tokens_entropy"],
                "val_critic_argmax_mse": critic_results["argmax_mse"],
                "val_critic_soft_mse": critic_results["soft_mse"],
                "val_critic_entropy": critic_results["entropy"],
            }
            if not minif2f_results.get('timed_out') and not minif2f_results.get('invalid'):
                wandb_data["minif2f_val"] = minif2f_results['success_rate']
            else:
                if minif2f_results.get('timed_out'):
                    wandb_data["minif2f_timed_out"] = True
                if minif2f_results.get('invalid'):
                    wandb_data["minif2f_invalid"] = True
            if not leanworkbook_results.get('timed_out') and not leanworkbook_results.get('invalid'):
                wandb_data["leanworkbook_val"] = leanworkbook_results['success_rate']
            else:
                if leanworkbook_results.get('timed_out'):
                    wandb_data["leanworkbook_timed_out"] = True
                if leanworkbook_results.get('invalid'):
                    wandb_data["leanworkbook_invalid"] = True
            wandb_run.log(wandb_data)

            # Save detailed evaluation results
            save_eval_results_to_run_dir(output_dir, step, "minif2f", minif2f_results)
            save_eval_results_to_run_dir(output_dir, step, "leanworkbook", leanworkbook_results)

        model.train()
        timer.end("eval")
        flush()  # Free memory from evaluation

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
            flush()  # Free any GPU memory from inference before training
            
            # Synchronize all ranks after loading
            if ddp:
                dist.barrier()
        else:
            # collect proofs
            timer.start("collect")
            model.eval()
            rl_monitor.start_collection(target_samples=args.collect_transitions, num_actors=args.num_actors)
            rl_monitor.set_step(step)

            prover.collect(theorems_sampler, args.collect_transitions, replay_buffer)

            model.train()
            timer.end("collect")
            flush()  # Free memory from collection before training
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
            optimizer.state_dict(),  # optimizer state
            checkpoint_meta,
            rank=ddp_rank,
        )

    timer.start("train")
    rl_monitor.set_phase("training")
    
    # Pause inference during training to prevent concurrent model access.
    # No-op for the LocalProver wrapper around BlockingTacticModel because
    # actors are already stopped at this point; meaningful in distributed mode
    # where the inference server is still serving remote provers.
    prover.pause()


    # evaluate the gradient
    num_tokens = torch.tensor(0, device=device)  # the number of "active" tokens of supervision seen
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_loader)  # prefetch the next batch while the GPU is busy with forward/backward
        # Compute per-token losses to apply different weights to value vs policy samples
        per_token_loss = model(train_inputs, train_targets, loss_reduction='none')  # (B*T,)
        per_token_loss = per_token_loss.view(train_inputs.shape)  # (B, T)

        # Identify value samples: those where input contains the value delimiter token
        is_value_sample = (train_inputs == value_delim_tok).any(dim=1)  # (B,)

        # Create per-sample weights: value_weight for value samples, 1.0 for policy samples
        sample_weights = torch.where(is_value_sample, args.value_weight, 1.0)  # (B,)

        # Compute weighted loss: weight each token by its sample's weight
        token_mask = (train_targets >= 0)  # (B, T)
        weighted_token_loss = per_token_loss * sample_weights.unsqueeze(1)  # (B, T)

        # Mean over all valid tokens (weighted)
        loss = (weighted_token_loss * token_mask).sum() / token_mask.sum()
        train_loss = loss.detach()  # for logging
        loss = loss / grad_accum_steps  # each .backward() is a grad sum => normalize loss here
        loss.backward()  # accumulate the gradient
        num_tokens += (train_targets >= 0).sum()
    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)  # sum over ranks

    # step the optimizer
    optimizer.step()
    model.zero_grad(set_to_none=True)
    
    # Resume inference after training. Clear CUDA cache first to free training memory.
    flush()
    prover.resume()


    timer.end("train")

    # logging
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
