"""Shared types for benchmark / RL theorem datasets.

``BenchTheorem`` bundles a theorem source with the Lean REPL preamble it
needs (``open``s, ``open scoped``s, and any auxiliary ``def``s). The
preamble is sent via ``env.send_command(theorem.header)`` before
``env.proof_from_sorry(theorem.source)`` inside :class:`nanoproof.prover.Prover`.

The ``header`` must NOT contain ``import`` directives. Imports are applied
once per Lean process at server startup via ``leanserver --imports Mathlib``.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchTheorem:
    source: str
    header: str
    name: str | None = None


# Shared preamble used by miniF2F and the RL datasets (leanworkbook,
# deepseek_prover, numinamath). Uses ``open scoped`` instead of bare ``open``
# so e.g. ``Nat.gcd`` is not pulled in as top-level ``gcd`` (which would
# conflict with ``GCDMonoid.gcd`` from Mathlib and produce
# "Ambiguous term: gcd" init failures).
MINIF2F_HEADER = (
    "open scoped Real\n"
    "open scoped Nat\n"
    "open scoped Topology\n"
    "open scoped Polynomial"
)
