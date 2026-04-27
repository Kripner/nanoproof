"""Shared types for benchmark / RL theorem datasets.

``BenchTheorem`` wraps a Lean source string that is ready to feed to
``proof_from_sorry`` via the leantree REPL. The ``source`` contains
everything needed to initialize the proof -- ``open`` / ``open scoped``
directives, auxiliary ``def``s, and the theorem/example declaration ending
in ``sorry``.

Imports (``import Mathlib``, etc.) are applied once per Lean process at
server startup and must NOT appear in ``source``.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchTheorem:
    """One benchmark/RL theorem.

    ``(dataset, id)`` is the global key used by the matchmaker, the on-disk
    attempt log, the inspector script, and the web UI. ``id`` must be unique
    within ``dataset`` and stable across runs (sourced from the upstream
    dataset, not synthesized at load time).
    """

    source: str
    dataset: str
    id: str


# Per-dataset preambles (open statements). These are prepended to each
# theorem's source by the respective dataset loader.

# miniF2F preamble. Mirrors upstream Valid.lean. Uses ``open scoped`` so
# e.g. ``Nat.gcd`` is not pulled in as top-level ``gcd``.
MINIF2F_PREAMBLE = (
    "open scoped Real\n"
    "open scoped Nat\n"
    "open scoped Topology\n"
    "open scoped Polynomial\n\n"
)

# LeanWorkBook preamble. Matches InternLM's upstream header at
# https://github.com/InternLM/InternLM-Math/blob/main/leanworkbook/header.lean
LEANWORKBOOK_PREAMBLE = "open BigOperators\nopen Nat\nopen Real\nopen Rat\n\n"

# DeepSeek-Prover-V1 preamble. All 27k rows share the same header upstream.
DEEPSEEK_PROVER_PREAMBLE = "open BigOperators Real Nat Topology Rat\n\n"
