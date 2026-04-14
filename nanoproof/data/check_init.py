"""Shared helpers for theorem ``check-init`` actions and Lean-version whitelists.

Theorem statements drift out of sync with the current Lean/Mathlib toolchain
(e.g. deprecated ``in`` binder in ``∑ k in s``). A whitelist of theorems that
successfully initialize in the REPL is built once per (dataset, Lean version)
pair and consulted by ``list_theorems`` at load time so the RL loop does not
waste collect budget attempting them.

Whitelist files live next to the source data file as
``<data_file>.whitelist.<lean_version>.json`` and contain BOTH passing and
failing theorem hashes, so a theorem whose hash is in neither set can be
flagged as ``whitelist stale`` rather than silently dropped.
"""

import concurrent.futures
import hashlib
import json
import os
import re
import threading
from urllib.request import urlopen

from leantree.repl_adapter.server import LeanClient

from nanoproof.cli import log
from nanoproof.common import theorem_to_example
from nanoproof.data.bench.common import BenchTheorem


def theorem_hash(theorem: BenchTheorem) -> str:
    """16-hex-char content hash over ``header + "\\n" + source``.

    Hash-keyed whitelists survive upstream dataset growth and reshuffles.
    """
    payload = (theorem.header + "\n" + theorem.source).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


# -----------------------------------------------------------------------------
# Lean version discovery
# -----------------------------------------------------------------------------

_TOOLCHAIN_RE = re.compile(r"leanprover/lean4:(v[0-9][^\s]*)")


def read_lean_version(lean_project: str) -> str:
    """Parse ``<lean_project>/lean-toolchain`` and return e.g. ``"v4.27.0"``."""
    path = os.path.join(lean_project, "lean-toolchain")
    with open(path, "r") as f:
        content = f.read().strip()
    m = _TOOLCHAIN_RE.search(content)
    if not m:
        raise ValueError(f"Could not parse Lean version from {path}: {content!r}")
    return m.group(1)


# -----------------------------------------------------------------------------
# Whitelist I/O
# -----------------------------------------------------------------------------

def whitelist_path(dataset_file: str, lean_version: str) -> str:
    return f"{dataset_file}.whitelist.{lean_version}.json"


def save_whitelist(path: str, lean_version: str, results: dict[str, bool]) -> None:
    passing = sorted(h for h, ok in results.items() if ok)
    failing = sorted(h for h, ok in results.items() if not ok)
    payload = {
        "lean_version": lean_version,
        "num_checked": len(results),
        "num_passing": len(passing),
        "num_failing": len(failing),
        "passing": passing,
        "failing": failing,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_whitelist(path: str) -> dict | None:
    """Return ``{"lean_version", "passing": set, "failing": set}`` or ``None`` if missing."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return {
        "lean_version": data.get("lean_version"),
        "passing": set(data.get("passing", [])),
        "failing": set(data.get("failing", [])),
    }


def filter_by_whitelist(
    theorems: list[BenchTheorem],
    whitelist_file: str,
    dataset_name: str,
) -> list[BenchTheorem]:
    """Filter ``theorems`` via the on-disk whitelist.

    Drops known-failing theorems, keeps known-passing theorems. Theorems whose
    hash is in neither set are kept (fail-open) but trigger a single aggregated
    warning so stale whitelists are not silently hiding theorems.
    """
    wl = load_whitelist(whitelist_file)
    if wl is None:
        raise FileNotFoundError(
            f"Whitelist not found: {whitelist_file}. "
            f"Generate it via the {dataset_name!r} module's `check-init` CLI action."
        )

    passing = wl["passing"]
    failing = wl["failing"]

    kept: list[BenchTheorem] = []
    num_passing = 0
    num_failing = 0
    unknown: list[BenchTheorem] = []

    for t in theorems:
        h = theorem_hash(t)
        if h in passing:
            kept.append(t)
            num_passing += 1
        elif h in failing:
            num_failing += 1
        else:
            kept.append(t)
            unknown.append(t)

    if unknown:
        example = unknown[0].source[:200].replace("\n", " ")
        log(
            f"{dataset_name}: {len(unknown)} theorems not in whitelist {whitelist_file} "
            f"(stale whitelist?). Example: {example!r}",
            component="Whitelist",
        )

    log(
        f"{dataset_name}: {num_passing} passing + {len(unknown)} unknown kept, "
        f"{num_failing} dropped (whitelist {wl['lean_version']})",
        component="Whitelist",
    )
    return kept


# -----------------------------------------------------------------------------
# check-init runner
# -----------------------------------------------------------------------------

def _split_addr(lean_server: str) -> tuple[str, int]:
    if ":" in lean_server:
        host, port_str = lean_server.rsplit(":", 1)
        return host, int(port_str)
    return lean_server, 8000


def _query_max_processes(host: str, port: int) -> int:
    with urlopen(f"http://{host}:{port}/status", timeout=10) as resp:
        status = json.loads(resp.read())
    n = int(status.get("max_processes", 0))
    if n == 0:
        raise ConnectionError(f"Lean server {host}:{port} reports 0 available processes")
    return n


def _check_one(client: LeanClient, theorem: BenchTheorem) -> tuple[bool, str | None]:
    """Attempt ``proof_from_sorry`` on a single theorem. Returns ``(ok, error_msg)``."""
    process = client.get_process()
    if process is None:
        return False, "could not acquire Lean process"
    try:
        with process as env:
            env.send_command(theorem.header)
            example = theorem_to_example(theorem.source)
            init_branch = env.proof_from_sorry(example)
            if not init_branch.is_success():
                err = init_branch.error if hasattr(init_branch, "error") else "unknown error"
                return False, str(err)
            return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def run_check_init(
    theorems: list[BenchTheorem],
    lean_server: str,
    num_workers: int = 1,
    verbose: bool = False,
) -> dict[str, bool]:
    """Run ``proof_from_sorry`` on each theorem and return ``{hash: passed}``.

    ``num_workers=0`` auto-detects from the server's ``max_processes``.
    ``num_workers=1`` runs serially (used by minif2f / proofnet defaults).
    Duplicate hashes in the input are checked once.
    """
    host, port = _split_addr(lean_server)
    if num_workers == 0:
        num_workers = _query_max_processes(host, port)
        print(f"Auto-detected {num_workers} workers from Lean server {host}:{port}")
    client = LeanClient(host, port)

    # Deduplicate by hash (same theorem may appear multiple times across datasets)
    unique: dict[str, BenchTheorem] = {}
    for t in theorems:
        unique.setdefault(theorem_hash(t), t)

    results: dict[str, bool] = {}
    results_lock = threading.Lock()
    num_done = [0]
    total = len(unique)
    progress_every = max(1, total // 100)

    def worker(item: tuple[str, BenchTheorem]) -> None:
        h, theorem = item
        ok, err = _check_one(client, theorem)
        with results_lock:
            results[h] = ok
            num_done[0] += 1
            n = num_done[0]
            if not ok and verbose:
                print(
                    f"[{n}/{total}] FAILED ({h}): {err}\n"
                    f"--- theorem ---\n{theorem.source}\n",
                    flush=True,
                )
            elif verbose:
                print(f"[{n}/{total}] ok ({h})", flush=True)
            if n % progress_every == 0 or n == total:
                num_passed = sum(1 for v in results.values() if v)
                print(
                    f"[check-init] {n}/{total}  passing={num_passed}  "
                    f"failing={n - num_passed}",
                    flush=True,
                )

    items = list(unique.items())
    if num_workers <= 1:
        for item in items:
            worker(item)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
            list(ex.map(worker, items))

    return results


# -----------------------------------------------------------------------------
# Shared CLI entrypoint
# -----------------------------------------------------------------------------

def run_check_init_cli(
    theorems: list[BenchTheorem],
    dataset_file: str,
    lean_server: str,
    lean_project: str,
    num_workers: int,
    limit: int | None,
    verbose: bool,
    save: bool,
) -> None:
    """Top-level helper wired into each dataset module's ``check-init`` subcommand.

    Runs the checker and prints a summary. When ``save=True`` (training corpora
    only), writes the whitelist JSON next to ``dataset_file`` with the Lean
    version from ``<lean_project>/lean-toolchain`` baked into the filename.
    When ``save=False`` (benchmarks), failures are surfaced as a prominent
    warning since we never want to silently drop benchmark theorems.
    """
    lean_version = read_lean_version(lean_project)
    out_path = whitelist_path(dataset_file, lean_version) if save else None

    if limit is not None:
        theorems = theorems[:limit]

    print(
        f"Running check-init on {len(theorems)} theorems "
        f"(Lean {lean_version}, server={lean_server}, workers={num_workers or 'auto'})"
    )
    if save:
        print(f"Output: {out_path}")
    else:
        print("Benchmark check-init: not writing a whitelist (benchmarks run every theorem).")

    results = run_check_init(theorems, lean_server, num_workers=num_workers, verbose=verbose)

    num_passing = sum(1 for v in results.values() if v)
    num_failing = len(results) - num_passing

    if save:
        save_whitelist(out_path, lean_version, results)
        print(
            f"Done: {num_passing} passing, {num_failing} failing, "
            f"{len(results)} unique theorems checked. Wrote {out_path}"
        )
    else:
        print(
            f"Done: {num_passing} passing, {num_failing} failing, "
            f"{len(results)} unique theorems checked (no whitelist written)."
        )
        if num_failing > 0:
            bar = "!" * 72
            print(f"\n{bar}\nWARNING: {num_failing} benchmark theorems failed to initialize "
                  f"under Lean {lean_version}.\nThese will be counted as unsolved during eval; "
                  f"fix the sources or the toolchain.\n{bar}\n")


# -----------------------------------------------------------------------------
# Shared argparse wiring
# -----------------------------------------------------------------------------

def add_check_init_args(parser, default_jobs: int) -> None:
    """Add the standard ``check-init`` flags to an argparse subparser."""
    parser.add_argument(
        "--lean-server",
        type=str,
        required=True,
        help="Lean server address (e.g. 10.10.25.33:8000); port defaults to 8000",
    )
    parser.add_argument(
        "--lean-project",
        type=str,
        required=True,
        help="Path to the Lean project directory (contains lean-toolchain). "
        "The Lean version from that file is baked into the whitelist filename.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=default_jobs,
        help="Number of parallel workers. 1 = serial; 0 = query /status and use max_processes.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only check the first N theorems")
    parser.add_argument("--verbose", action="store_true", help="Log each theorem's result")
