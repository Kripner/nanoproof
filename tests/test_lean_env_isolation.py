"""Verify LeanProcessPool isolates per-session commands across acquisitions.

Each theorem's source may contain ``open`` / ``def`` preambles that are
processed by ``proof_from_sorry``. The pool must restore each process to
its post-``import Mathlib`` checkpoint on release so these don't leak.

Requires a live Lean REPL server. Skipped unless ``NANOPROOF_LEAN_SERVER``
is set (e.g. ``NANOPROOF_LEAN_SERVER=10.10.25.33:8000 pytest -k isolation``).
"""

import os

import pytest

from leantree.repl_adapter.server import LeanClient

LEAN_SERVER = os.environ.get("NANOPROOF_LEAN_SERVER")

pytestmark = pytest.mark.skipif(
    not LEAN_SERVER,
    reason="Set NANOPROOF_LEAN_SERVER=host:port to run live Lean REPL tests",
)


def _parse_addr(addr: str) -> tuple[str, int]:
    if ":" in addr:
        host, port_str = addr.rsplit(":", 1)
        return host, int(port_str)
    return addr, 8000


def test_per_session_commands_do_not_leak():
    """A ``def`` and ``open`` sent in session 1 must not be visible in session 2."""
    host, port = _parse_addr(LEAN_SERVER)
    client = LeanClient(host, port)

    proc1 = client.get_process()
    assert proc1 is not None, "Failed to acquire process for session 1"
    with proc1 as env:
        env.send_command("def __isolation_test_foo := 5")
        env.send_command("open Nat")
        branch = env.proof_from_sorry("example : __isolation_test_foo = 5 := by sorry")
        assert branch.is_success(), (
            f"session 1 sanity check: def should be visible within the session, got {branch.error}"
        )

    proc2 = client.get_process()
    assert proc2 is not None, "Failed to acquire process for session 2"
    with proc2 as env:
        branch = env.proof_from_sorry("example : __isolation_test_foo = 5 := by sorry")
        assert not branch.is_success(), (
            "env isolation broken: def from session 1 is still visible in session 2"
        )
