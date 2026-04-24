"""Download any or all of the nanoproof datasets with one command.

Usage:
    python -m nanoproof.data.download                          # download everything
    python -m nanoproof.data.download minif2f                  # just one
    python -m nanoproof.data.download sft rl                   # stage aliases
    python -m nanoproof.data.download minif2f proofnet leantree
"""

import argparse
import traceback

from nanoproof.data.bench import minif2f, proofnet
from nanoproof.data.midtrain import leangithubraw
from nanoproof.data.pretrain import nemotron
from nanoproof.data.rl import deepseek_prover, leanworkbook, numinamath
from nanoproof.data.sft import leantree

DATASETS = {
    "nemotron": nemotron.download_dataset,
    "leangithubraw": leangithubraw.download_dataset,
    "leantree": leantree.download_dataset,
    "leanworkbook": leanworkbook.download_dataset,
    "numinamath": numinamath.download_dataset,
    "deepseek_prover": deepseek_prover.download_dataset,
    "minif2f": minif2f.download_dataset,
    "proofnet": proofnet.download_dataset,
}

STAGE_ALIASES = {
    "pretrain": ["nemotron"],
    "midtrain": ["leangithubraw"],
    "sft": ["leantree"],
    "rl": ["leanworkbook", "numinamath", "deepseek_prover"],
    "bench": ["minif2f", "proofnet"],
}


def _expand(selected):
    """Expand stage aliases to dataset names, de-duplicating in input order.

    Empty input expands to every dataset.
    """
    if not selected:
        return list(DATASETS)
    valid = set(DATASETS) | set(STAGE_ALIASES)
    out = []
    seen = set()
    for name in selected:
        if name not in valid:
            raise SystemExit(
                f"error: invalid dataset {name!r}. Choose from: "
                + ", ".join(list(DATASETS) + list(STAGE_ALIASES))
            )
        for resolved in STAGE_ALIASES.get(name, [name]):
            if resolved not in seen:
                seen.add(resolved)
                out.append(resolved)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Download nanoproof datasets.", allow_abbrev=False
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        metavar="NAME",
        help=(
            "Datasets to download (default: all). "
            "Datasets: " + ", ".join(DATASETS) + ". "
            "Stage aliases: " + ", ".join(STAGE_ALIASES) + "."
        ),
    )
    args = parser.parse_args()

    datasets = _expand(args.datasets)

    failures = []
    for name in datasets:
        print(f"\n=== Downloading {name} ===")
        try:
            DATASETS[name]()
        except Exception as e:
            traceback.print_exc()
            failures.append((name, e))

    print("\n=== Summary ===")
    for name in datasets:
        status = "FAILED" if any(n == name for n, _ in failures) else "ok"
        print(f"  {name}: {status}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
