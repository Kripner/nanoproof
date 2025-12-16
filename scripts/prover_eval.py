import argparse
import os

import torch
import torch.distributed as dist
from tqdm import tqdm
from leantree.repl_adapter.server import LeanClient

from nanoproof.common import compute_init, compute_cleanup, print0, is_ddp, autodetect_device_type, get_dist_info
from nanoproof.data import minif2f
from nanoproof.data import leanworkbook
from nanoproof.search import run_mcts, Config, Game, Node, Player, TacticModel
from nanoproof.checkpoints import load_model
from nanoproof.engine import Engine

@torch.inference_mode()
def eval_success_rate(tactic_model: TacticModel, theorems=None, use_tqdm=False):
    """
    Evaluates the success rate of the model on the MiniF2F benchmark.
    Returns a dictionary with 'success_rate', 'solved', and 'total'.
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    theorem_indices = list(range(ddp_rank, len(theorems), ddp_world_size))
    theorems = [theorems[i] for i in theorem_indices]

    config = Config()
    client = LeanClient(config.server_address, config.server_port)
    
    solved_count = 0
    error_count = 0
    
    device = tactic_model.network.get_device()
    with client.get_process() as env:
        env.send_command("""
            open scoped Real
            open scoped Nat
            open scoped Topology
            open scoped Polynomial
        """)
        iterator = zip(theorem_indices, theorems)
        if use_tqdm:
            iterator = tqdm(iterator, total=len(theorems), desc=f"Rank {ddp_rank}", position=ddp_rank)
            
        for i, theorem in iterator:
            init_branch = env.proof_from_sorry(theorem)
            if not init_branch.is_success():
                error_count += 1
                print0(f"Error on theorem: {theorem}\n... error: {init_branch.error}")
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
        "errors": global_error,
        "error_rate": error_rate,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-theorems", type=int, default=50, help="Max theorems to evaluate")
    args = parser.parse_args()

    device_type = autodetect_device_type()
    compute_init(device_type)

    tactic_model = TacticModel.create()
    minif2f_theorems = minif2f.list_theorems(split="Valid")
    minif2f_theorems = minif2f_theorems[:args.max_theorems]
    leanworkbook_theorems = leanworkbook.list_theorems(split="val")
    leanworkbook_theorems = leanworkbook_theorems[:args.max_theorems]

    def print_results(results, name):
        print0("-" * 80)
        print0(f"Evaluation results for {name}")
        print0(f"Success rate: {results['success_rate']:.4%}")
        print0(f"Solved: {results['solved']}/{results['total']}")
        print0(f"Errors: {results['errors']}/{results['total']}")
        print0(f"Error rate: {results['error_rate']:.4%}")
        print0("-" * 80)

    leanworkbook_results = eval_success_rate(tactic_model, leanworkbook_theorems, use_tqdm=True)
    print_results(leanworkbook_results, "LeanWorkBook")

    minif2f_results = eval_success_rate(tactic_model, minif2f_theorems, use_tqdm=True)
    print_results(minif2f_results, "MiniF2F")

    compute_cleanup()

if __name__ == "__main__":
    main()
