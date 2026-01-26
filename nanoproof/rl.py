import atexit
import os
from datetime import datetime
import json
import sys
from contextlib import nullcontext
import random
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import wandb
import torch
import torch.distributed as dist
import leantree.augmentations
from leantree.core.lean import LeanGoal

from nanoproof.common import compute_init, compute_cleanup, get_base_dir, DummyWandb, autodetect_device_type, SimpleTimer, flush, active_barrier_master, active_barrier_wait
from nanoproof.checkpoints import load_model, save_checkpoint
from nanoproof.engine import Engine
from nanoproof.data.leantree import iter_data
from nanoproof.data.leantree_dataloader import rl_data_generator
from nanoproof.experience_collection import ReplayBuffer, TheoremsSampler, Config, run_actor
from nanoproof.inference import TacticModel, BlockingTacticModel
from nanoproof.data import minif2f
from nanoproof.data import leanworkbook
from nanoproof.cli import create_monitor, configure_logging, log, log0, set_ddp_info
from nanoproof.rl_server import distributed_collect, distributed_eval, start_coordinator, shutdown_coordinator
from nanoproof.inference import start_inference_server
from nanoproof.infra import load_infra_config, InfraConfig, parse_lean_server
from scripts.prover_eval import eval_success_rate
from scripts.policy_eval import eval_tactic_accuracy, eval_critic_errors
from nanoproof.data.leantree_dataloader import sft_data_generator

# TODO: in each episode, save a sample of training data (right before it goes into the model)

# TODO: matchmaker

# TODO: maybe log numbers of OR and AND nodes in the proof searches

# TODO: maybe migrate to newer Lean?


def save_eval_results(output_dir: str, step: int, dataset_name: str, results: dict):
    """
    Save evaluation results to a JSONL file.
    
    Args:
        output_dir: The output directory for the run
        step: Current training step
        dataset_name: Name of the dataset (e.g., "minif2f", "leanworkbook")
        results: Dict containing 'detailed_results' with evaluation details
    """
    # Create evals/{step}/ directory
    eval_dir = os.path.join(output_dir, "evals", str(step))
    os.makedirs(eval_dir, exist_ok=True)
    
    # Get detailed results
    detailed_results = results.get("detailed_results", [])
    
    # Save to JSONL
    jsonl_path = os.path.join(eval_dir, f"{dataset_name}.jsonl")
    with open(jsonl_path, "w") as f:
        for item in detailed_results:
            # Handle both dict (from distributed) and TheoremResult (from local) formats
            if hasattr(item, "theorem"):
                # It's a TheoremResult dataclass
                entry = {
                    "theorem": item.theorem,
                    "proof": item.proof_tree,
                    "unsimplified_proof": item.unsimplified_proof_tree,
                    "num_iterations": item.num_iterations,
                }
            else:
                # It's a dict (from distributed_eval)
                entry = {
                    "theorem": item["theorem"],
                    "proof": item["proof_tree"],
                    "unsimplified_proof": item.get("unsimplified_proof_tree"),
                    "num_iterations": item["num_iterations"],
                }
            f.write(json.dumps(entry) + "\n")
    
    log(f"Saved {len(detailed_results)} eval results to {jsonl_path}", component="Eval")

# -----------------------------------------------------------------------------
# RL Hyperparameters
run = "dummy"  # wandb run name default ("dummy" is special - we won't log to wandb)
seed = 0
model_tag = "d26"
model_step = 903
# model_tag = "d32"
# model_step = 4515
# compute/precision
device_type = ""  # cuda|cpu|mps (empty => autodetect)
dtype = "bfloat16"
device_batch_size = 8  # (maybe) max to avoid OOM (on A100 40GB)
# distributed mode - controlled by infra_file
infra_file = "infra-ms.toml"  # path to infra.toml for distributed mode (empty => local mode)
inference_server_port = 5000  # port for inference server (distributed mode only)
poll_interval = 3.0  # how often to poll provers for transitions (seconds)
# local mode settings
lean_server = "10.10.25.31:8000"  # lean server for local mode (host:port)
# data
fraction_sft = 0.2  # 20% of data will come from Mathlib (leantree), 80% from replay buffer
collect_every = 1  # how many steps to train between RL data collections  # TODO: when collect_every>1, we need some warmup (collect collect_every*collect_transitions)
collect_transitions = 100  # how many proof transitions to collect in one collection
# parallel experience collection (local mode only)
num_actors = 32  # number of parallel actor threads for experience collection and evaluation
num_sampled_tactics = 6  # number of tactics to sample per state in MCTS
batch_timeout = 0.2  # timeout in seconds for batching LLM calls
max_batch_tokens = 8000  # (maybe) max total tokens per inference batch (A100 40GB)
# optimization
target_examples_per_step = 512
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.02
augment_data = True
value_weight = 0.1  # weight for value (critic) samples relative to policy samples
# evaluation and logging there of
eval_every = 100 
eval_start = 0  # step to start evaluation at (skip evaluation before this step)
save_every = 500
# resuming from a previous run - Note: don't forget to also set eval_start and seed!
resume_from = ""  # path to a previous run's output directory to resume from (uses its replay buffers)
# now allow CLI to override the settings via the configurator lol
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join("nanoproof", "configurator.py")).read())  # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys}  # possibly useful for logging

# -----------------------------------------------------------------------------
# Distributed mode configuration
# If infra_file is provided, load the infrastructure config and enable distributed mode
# Otherwise, use local mode with lean_server parameter

distributed = bool(infra_file)  # distributed mode is enabled if infra_file is provided
infra_config: InfraConfig | None = None

if distributed:
    if not os.path.exists(infra_file):
        log(f"Error: Infrastructure file {infra_file} does not exist", component="Main")
        sys.exit(1)
    infra_config = load_infra_config(infra_file)
    # Override inference_server_port from infra config
    inference_server_port = infra_config.rl_server_port
    # Get lean servers for monitoring
    lean_servers = infra_config.get_lean_server_list()
    # Lean server for local actors not used in distributed mode
    local_lean_server = parse_lean_server("localhost:8000")
else:
    # Local mode: parse lean_server and use it for both actors and monitoring
    local_lean_server = parse_lean_server(lean_server)
    lean_servers = [str(local_lean_server)]

# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
set_ddp_info(is_master=master_process, rank=ddp_rank)
ptdtype = torch.float32 if dtype == "float32" else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# Output directory init

if master_process:
    base_dir = get_base_dir()
    timestamp = datetime.now().strftime("%y-%m-%d_%H-%M")
    output_dirname = f"{timestamp}-{run}"
    output_dir = os.path.join(base_dir, "rl", output_dirname)
    if os.path.exists(output_dir):
        log(f"Error: Output directory {output_dir} already exists", component="Main")
        if ddp:
            dist.destroy_process_group()
        sys.exit(1)
    os.makedirs(output_dir)
    log(f"Output directory: {output_dir}", component="Main")
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

# Create the tactic model
# In distributed mode, we use BlockingTacticModel for inference servers (started later)
# In local mode, we use BlockingTacticModel for parallel actors
log0(f"Distributed mode: {distributed}", component="Config")
inner_tactic_model = TacticModel.create(num_samples=num_sampled_tactics, model_tag=model_tag, step=model_step)
if distributed:
    # In distributed mode, we don't need BlockingTacticModel here - 
    # inference servers are started later with their own BlockingTacticModel
    tactic_model = None
    model = inner_tactic_model.network
else:
    tactic_model = BlockingTacticModel(
        inner_model=inner_tactic_model,
        timeout_seconds=batch_timeout,
        max_batch_tokens=max_batch_tokens
    )
    model = tactic_model.network

# -----------------------------------------------------------------------------
# DataLoader

examples_per_step = device_batch_size * ddp_world_size
log0(f"Target examples per step: {target_examples_per_step}", component="Config")
log0(f"Device batch size: {device_batch_size}", component="Config")
log0(f"Examples per step is device_batch_size * ddp_world_size: {examples_per_step}", component="Config")
assert target_examples_per_step % examples_per_step == 0, "Target examples per step must be divisible by examples per step"
grad_accum_steps = target_examples_per_step // examples_per_step
log0(f"=> Setting grad accum steps: {grad_accum_steps}", component="Config")

rank_seed = seed + ddp_rank

config = Config(
    num_actors=num_actors,
    num_sampled_tactics=num_sampled_tactics,
    server_address=local_lean_server.address,
    server_port=local_lean_server.port,
)
replay_buffer = ReplayBuffer(config, seed=rank_seed)
theorems_sampler = TheoremsSampler(seed=rank_seed)

# Create the RL monitor (only show on master process)
rl_monitor = create_monitor(num_actors=num_actors, enabled=master_process)
rl_monitor.set_output_dir(output_dir)
rl_monitor.set_lean_servers(lean_servers)

shuffle_goals_and_hypotheses = leantree.augmentations.ShuffleGoalsAndHypotheses(seed=seed)
random_rename = leantree.augmentations.RandomRename(seed=seed)

mathlib_train = list(iter_data(
    split="train",
    augmentations=[shuffle_goals_and_hypotheses, random_rename] if augment_data else None,
))
random.Random(rank_seed).shuffle(mathlib_train)
mathlib_val = list(iter_data(split="val"))

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

def train_generator():
    rng = random.Random(rank_seed)
    mathlib_iter = iter(mathlib_train)
    while True:
        assert len(replay_buffer.buffer) >= collect_transitions
        if rng.random() < fraction_sft:
            try:
                state, tactic, proof_depth = next(mathlib_iter)
            except StopIteration:
                mathlib_iter = iter(mathlib_train)
                state, tactic, proof_depth = next(mathlib_iter)
        else:
            state, tactic, value_target = replay_buffer.sample_transition()
            proof_depth = -value_target
            # Only run augmentations on replay buffer data - Mathlib data is already augmented.
            if augment_data:
                state, tactic = augment(state, tactic)

        yield state, tactic, proof_depth


train_loader = rl_data_generator(train_generator(), batch_size=device_batch_size)
value_delim_tok = inner_tactic_model.tokenizer.encode_special("<|value|>")  # for distinguishing policy vs value samples

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
# -----------------------------------------------------------------------------

# Start inference servers in distributed mode (each rank runs one)
if distributed:
    # Create BlockingTacticModel for the inference server
    inference_model = BlockingTacticModel(
        inner_model=inner_tactic_model,
        timeout_seconds=batch_timeout,
        max_batch_tokens=max_batch_tokens
    )
    # Each rank starts an inference server on its own port
    inference_port = inference_server_port + 1 + ddp_rank  # ports 5001, 5002, ...
    inference_server_thread = start_inference_server(inference_model, inference_port)
    
    # Sync to ensure all inference servers are up before coordinator starts
    if ddp:
        dist.barrier()
    
    # Master also starts the coordinator (proxies inference + handles registration)
    if master_process:
        inference_endpoints = [f"http://127.0.0.1:{inference_server_port + 1 + r}" for r in range(ddp_world_size)]
        coordinator_thread, inference_router = start_coordinator(
            inference_server_port, inference_endpoints, startup_timeout=30.0
        )
    
    # Sync again so all ranks wait for coordinator
    if ddp:
        dist.barrier()

# Go!
step = 0
minif2f_results = None
leanworkbook_results = None

# Register cleanup to run on exit (handles normal exit, sys.exit, and unhandled exceptions)
def cleanup():
    """Cleanup function to ensure resources are released on shutdown."""
    log0("Shutting down...", component="Main")
    # In distributed mode, shutdown coordinator first to stop provers from retrying
    if distributed and master_process:
        shutdown_coordinator()
    # Shutdown the batched tactic model to unblock any waiting threads
    if tactic_model is not None:
        tactic_model.shutdown()
    if distributed:
        inference_model.shutdown()
    log0("Shutdown complete", component="Main")

atexit.register(cleanup)

while True:
    timer = SimpleTimer()
    
    if step % eval_every == 0 and step >= eval_start:
        timer.start("eval")
        model.eval()
        rl_monitor.set_phase("evaluating")

        # Policy evaluation (tactic accuracy and critic errors on mathlib val)
        eval_steps = 200
        build_val_loader = lambda: sft_data_generator(mathlib_val, batch_size=device_batch_size)
        with autocast_ctx:
            tactic_results = eval_tactic_accuracy(model, inner_tactic_model.tokenizer, build_val_loader(), max_steps=eval_steps)
            critic_results = eval_critic_errors(model, inner_tactic_model.tokenizer, build_val_loader(), max_steps=eval_steps)
        
        if master_process:
            log(f"Step {step:05d} | Tactic full acc: {tactic_results['full_acc']:.4%} | Tactic first acc: {tactic_results['first_token_acc']:.4%} | Critic argmax MSE: {critic_results['argmax_mse']:.4f} | Critic soft MSE: {critic_results['soft_mse']:.4f}", component="Eval")
            log(f"  Entropy - Tactic first: {tactic_results['first_token_entropy']:.4f} | Tactic all: {tactic_results['all_tokens_entropy']:.4f} | Critic: {critic_results['entropy']:.4f}", component="Eval")

        # Load eval theorems
        minif2f_theorems = minif2f.list_theorems(split="Valid")
        leanworkbook_theorems = leanworkbook.list_theorems(split="val")[:128]

        if distributed:
            # Distributed mode: use prover servers for evaluation (master only)
            if master_process:
                log(f"Evaluating on {len(minif2f_theorems)} theorems from MiniF2F (distributed)", component="Eval")
                minif2f_results = distributed_eval(minif2f_theorems, dataset_name="MiniF2F")
                rl_monitor.record_eval(step, "MiniF2F", minif2f_results['success_rate'],
                                       minif2f_results['solved'], minif2f_results['total'], minif2f_results['errors'])

                log(f"Evaluating on {len(leanworkbook_theorems)} theorems from LeanWorkBook (distributed)", component="Eval")
                leanworkbook_results = distributed_eval(leanworkbook_theorems, dataset_name="LeanWorkBook")
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
                save_eval_results(output_dir, step, "minif2f", minif2f_results)
                save_eval_results(output_dir, step, "leanworkbook", leanworkbook_results)
                
                active_barrier_master(f"eval_done_{step}")
            else:
                active_barrier_wait(f"eval_done_{step}")
        else:
            # Local mode: use local model
            class MonitorProgress:
                def __init__(self, dataset_name: str):
                    self.dataset_name = dataset_name
                
                def on_start(self, dataset: str, total: int):
                    rl_monitor.start_eval(self.dataset_name, total)
                
                def on_update(self, current: int, solved: int, errors: int):
                    rl_monitor.update_eval_progress(current, solved, errors)

            log(f"Evaluating on {len(minif2f_theorems)} theorems from MiniF2F", component="Eval")
            minif2f_results = eval_success_rate(
                inner_tactic_model, minif2f_theorems, progress=MonitorProgress("MiniF2F"),
                num_actors=num_actors
            )
            rl_monitor.record_eval(step, "MiniF2F", minif2f_results['success_rate'],
                                   minif2f_results['solved'], minif2f_results['total'], minif2f_results['errors'])

            log(f"Evaluating on {len(leanworkbook_theorems)} theorems from LeanWorkBook", component="Eval")
            leanworkbook_results = eval_success_rate(
                inner_tactic_model, leanworkbook_theorems, progress=MonitorProgress("LeanWorkBook"),
                num_actors=num_actors
            )
            rl_monitor.record_eval(step, "LeanWorkBook", leanworkbook_results['success_rate'],
                                   leanworkbook_results['solved'], leanworkbook_results['total'],
                                   leanworkbook_results['errors'])

            log(f"Step {step:05d} | minif2f: {minif2f_results['success_rate']:.4%} ({minif2f_results['solved']}/{minif2f_results['total']}, errors={minif2f_results['errors']}) | leanworkbook: {leanworkbook_results['success_rate']:.4%} ({leanworkbook_results['solved']}/{leanworkbook_results['total']}, errors={leanworkbook_results['errors']})", component="Eval")
            wandb_run.log({
                "step": step,
                "minif2f_val": minif2f_results['success_rate'],
                "leanworkbook_val": leanworkbook_results['success_rate'],
                # Policy evaluation metrics
                "val_full_acc": tactic_results["full_acc"],
                "val_first_token_acc": tactic_results["first_token_acc"],
                "val_first_token_entropy": tactic_results["first_token_entropy"],
                "val_all_tokens_entropy": tactic_results["all_tokens_entropy"],
                "val_critic_argmax_mse": critic_results["argmax_mse"],
                "val_critic_soft_mse": critic_results["soft_mse"],
                "val_critic_entropy": critic_results["entropy"],
            })
            
            # Save detailed evaluation results
            save_eval_results(output_dir, step, "minif2f", minif2f_results)
            save_eval_results(output_dir, step, "leanworkbook", leanworkbook_results)

        model.train()
        timer.end("eval")
        flush()  # Free memory from evaluation

    if step % collect_every == 0:
        # Check if we can resume from a previous run's replay buffer
        resume_file = os.path.join(resume_from, f"replay_buffer_{step:05d}.jsonl") if resume_from else None
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
            rl_monitor.start_collection(target_samples=collect_transitions, num_actors=num_actors)
            rl_monitor.set_step(step)
            
            if distributed:
                if master_process:
                    # Distributed mode: coordinate remote provers (only master does this)
                    # Retry until we have enough transitions
                    while len(replay_buffer.local_buffer) < collect_transitions:
                        collected = distributed_collect(
                            sampler=theorems_sampler,
                            target_transitions=collect_transitions,
                            poll_interval=poll_interval,
                            replay_buffer=replay_buffer,
                        )
                        if collected < collect_transitions:
                            log(f"Collection incomplete ({collected}/{collect_transitions}), retrying...", 
                                component="Coordinator")
                    # Signal completion to other ranks via store.
                    active_barrier_master(f"collection_done_{step}")
                else:
                    # Active waiting to not block the Python thread (which is needed for inference).
                    active_barrier_wait(f"collection_done_{step}")
                
                # Wait for provers to abort their MCTS searches and release Lean processes
                time.sleep(3.0)
            else:
                # Local mode: run actors locally
                run_actor(collect_transitions, config, tactic_model, replay_buffer, theorems_sampler)
            
            model.train()
            timer.end("collect")
            flush()  # Free memory from collection before training
            replay_buffer.synchronize()
            if master_process:
                rl_monitor.set_replay_buffer_size(len(replay_buffer.buffer))
                with open(os.path.join(output_dir, f"replay_buffer_{step:05d}.jsonl"), "w") as f:
                    for context, tactic, value_target in replay_buffer.buffer:
                        f.write(json.dumps({"context": context, "tactic": tactic, "value_target": value_target}) + "\n")

    if step % save_every == 0 and step > 0 and master_process:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        model_config_kwargs = model.config.__dict__  # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
        checkpoint_meta = {
            "step": step,
            "model_config": model_config_kwargs,
        }
        if minif2f_results:
            checkpoint_meta["minif2f_val"] = minif2f_results['success_rate']
        if leanworkbook_results:
            checkpoint_meta["leanworkbook_val"] = leanworkbook_results['success_rate']
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            [opt.state_dict() for opt in optimizers],  # optimizer states
            checkpoint_meta,
            rank=ddp_rank,
        )

    timer.start("train")
    rl_monitor.set_phase("training")
    
    # Pause inference server during training to prevent concurrent model access
    if distributed:
        inference_model.pause()
    
    # evaluate the gradient
    num_tokens = torch.tensor(0, device=device)  # the number of "active" tokens of supervision seen
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_loader)  # prefetch the next batch while the GPU is busy with forward/backward
        with autocast_ctx:
            # Compute per-token losses to apply different weights to value vs policy samples
            per_token_loss = model(train_inputs, train_targets, loss_reduction='none')  # (B*T,)
            per_token_loss = per_token_loss.view(train_inputs.shape)  # (B, T)
            
            # Identify value samples: those where input contains the value delimiter token
            is_value_sample = (train_inputs == value_delim_tok).any(dim=1)  # (B,)
            
            # Create per-sample weights: value_weight for value samples, 1.0 for policy samples
            sample_weights = torch.where(is_value_sample, value_weight, 1.0)  # (B,)
            
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

    # step the optimizers
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    
    # Resume inference server after training
    if distributed:
        # Clear CUDA cache before resuming inference (frees training memory)
        flush()
        inference_model.resume()
    
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
