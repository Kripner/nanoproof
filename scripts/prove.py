"""
Single-prover CLI for testing.

Usage:
    python -m scripts.prove --model-path PATH [--lean-server HOST:PORT] [--theorem ...]

Without --theorem, runs as a REPL: paste a Lean statement (ending with `sorry`)
followed by a blank line, and the prover will attempt to prove it. Type
q/quit/exit to leave.
"""

import argparse
import logging
import sys

from leantree.repl_adapter.server import LeanClient

from nanoproof.common import (
    add_dataclass_args,
    construct_proof_source,
    dataclass_from_args,
    linearize_proof,
)
from nanoproof.data.bench.common import BenchTheorem
from nanoproof.inference import TacticModel
from nanoproof.prover import Prover
from nanoproof.search import SearchConfig

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the prover on a single Lean statement."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to model_NNNNNN.pt (relative to models/ or absolute)",
    )
    parser.add_argument(
        "--lean-server",
        default="localhost:8000",
        help="Lean server address (HOST:PORT)",
    )
    parser.add_argument("--num-simulations", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--first-token-occurrences-cap", type=int, default=2)
    parser.add_argument(
        "--theorem",
        default=None,
        help="If given, run once on this source; otherwise REPL",
    )
    add_dataclass_args(parser, SearchConfig, prefix="search_")
    return parser.parse_args()


def parse_lean_server(addr: str) -> tuple[str, int]:
    if ":" in addr:
        host, port_str = addr.rsplit(":", 1)
        return host, int(port_str)
    return addr, 8000


def read_theorem_repl() -> str | None:
    """Read a theorem source from stdin, terminated by a blank line.

    Returns None if the user asked to quit or stdin closed.
    """
    print(
        "Enter a Lean statement ending with `sorry`, followed by a blank line "
        "(q/quit/exit to leave):"
    )
    try:
        line = input()
    except EOFError:
        return None
    if line.strip() in ("q", "quit", "exit"):
        return None
    lines = []
    while line.strip() or not lines:
        lines.append(line.rstrip())
        try:
            line = input()
        except EOFError:
            break
    return "\n".join(lines)


def prove_one(
    prover: Prover, client: LeanClient, source: str, num_simulations: int
) -> bool:
    theorem = BenchTheorem(source=source, dataset="cli", id="user")
    game = prover.prove(client, theorem, num_simulations=num_simulations)
    iters = game.num_iterations if game else 0
    is_solved = bool(game and game.root and game.root.is_solved)
    if is_solved:
        tactics = linearize_proof(game.root)
        proof_source = construct_proof_source(theorem.source, tactics)
        print()
        print(f"=== PROVEN (iters={iters}) ===")
        print(proof_source)
        print()
        print("=== Proof tree ===")
        print(game.root.pp_tree())
    else:
        print(f"UNPROVEN (iters={iters})")
    return is_solved


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    logger.info(f"Loading model: {args.model_path}")
    tactic_model = TacticModel.create(
        num_samples=args.num_samples,
        model_path=args.model_path,
        first_token_occurrences_cap=args.first_token_occurrences_cap,
    )
    host, port = parse_lean_server(args.lean_server)
    logger.info(f"Connecting to Lean server: {host}:{port}")
    client = LeanClient(host, port)
    search_config = dataclass_from_args(SearchConfig, args, prefix="search_")
    prover = Prover(search_config, tactic_model)

    if args.theorem is not None:
        is_solved = prove_one(prover, client, args.theorem, args.num_simulations)
        sys.exit(0 if is_solved else 1)

    while True:
        source = read_theorem_repl()
        if source is None:
            break
        if not source.strip():
            continue
        try:
            print("Starting proof search.")
            prove_one(prover, client, source, args.num_simulations)
        except KeyboardInterrupt:
            print("\n[interrupted; staying in REPL]")
        except AssertionError as e:
            print(f"[error] AssertionError: {e}")
        except Exception as e:
            print(f"[error] {type(e).__name__}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
