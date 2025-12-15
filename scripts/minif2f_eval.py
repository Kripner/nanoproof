import argparse
import os

import torch
import torch.distributed as dist
from tqdm import tqdm
from leantree.repl_adapter.server import LeanClient

from nanoproof.common import compute_init, compute_cleanup, print0, is_ddp, autodetect_device_type, get_dist_info
from nanoproof.data import minif2f
from nanoproof.search import run_mcts, Config, Game, Node, Player, TacticModel
from nanoproof.checkpoints import load_model
from nanoproof.engine import Engine

def eval_minif2f(tactic_model: TacticModel, max_theorems=None, split="Valid", use_tqdm=False):
    """
    Evaluates the success rate of the model on the MiniF2F benchmark.
    Returns a dictionary with 'success_rate', 'solved', and 'total'.
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    theorems = minif2f.list_theorems(split)
    if max_theorems is not None:
        theorems = theorems[:max_theorems]
    theorem_indices = list(range(ddp_rank, len(theorems), ddp_world_size))
    theorems = [theorems[i] for i in theorem_indices]
    print0(f"Evaluating on {len(theorems)} theorems from MiniF2F {split} (distributed across {ddp_world_size} ranks)")

    config = Config()
    client = LeanClient(config.server_address, config.server_port)
    
    solved_count = 0
    error_count = 0
    
    device = tactic_model.network.get_device()
    with client.get_process() as env:
        iterator = zip(theorem_indices, theorems)
        if use_tqdm:
            iterator = tqdm(iterator, total=len(theorems), desc=f"Rank {ddp_rank}", position=ddp_rank)
            
        for i, theorem in iterator:
            init_branch = env.proof_from_sorry(theorem)
            if not init_branch.is_success():
                error_count += 1
                continue
            init_branch = init_branch.value
            
            game = Game(theorem, num_simulations=config.num_simulations)
            game.root = Node(
                action=None,
                prior=None,
                state=[init_branch],
                to_play=Player.OR,
                reward=None,
            )
            
            run_mcts(config, game, tactic_model)
            
            if game.root.is_solved:
                solved_count += 1

    local_metrics = torch.tensor([solved_count, error_count, len(theorem_indices)], dtype=torch.long, device=device)
    if ddp:
        dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM)
    global_solved = local_metrics[0].item()
    global_error = local_metrics[1].item()
    global_total = local_metrics[2].item()
    
    success_rate = global_solved / global_total if global_total > 0 else 0.0
    error_rate = global_error / global_total if global_total > 0 else 0.0
    return {
        "success_rate": success_rate,
        "solved": global_solved,
        "total": global_total,
        "error": global_error,
        "error_rate": error_rate,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="Valid", choices=["Valid", "Test"], help="MiniF2F split to evaluate")
    parser.add_argument("--max-theorems", type=int, default=50, help="Max theorems to evaluate")
    args = parser.parse_args()


    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    tactic_model = TacticModel.create()
    results = eval_minif2f(tactic_model, max_theorems=args.max_theorems, split=args.split, use_tqdm=True)
    
    print0("-" * 80)
    print0(f"Evaluation results for MiniF2F {args.split}")
    print0(f"Total theorems evaluated: {results['total']}")
    print0(f"Total solved: {results['solved']}")
    print0(f"Success rate: {results['success_rate']:.2%}")
    print0(f"Error rate: {results['error_rate']:.2%}")
    print0("-" * 80)

    compute_cleanup()

if __name__ == "__main__":
    main()
