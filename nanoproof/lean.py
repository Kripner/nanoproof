"""
Lean server discovery and connection assignment.

Parses server addresses, queries each server for capacity, and builds
the per-actor (host, port) assignment list used by ``ProverWorker``.
"""

import json as json_mod
from urllib.request import urlopen

from nanoproof.cli import log


def parse_lean_server_addrs(raw: list[str], default_port: int = 8000) -> list[str]:
    """Normalize raw server strings to ``"host:port"`` form.

    Entries without a port get *default_port* appended.
    """
    return [s if ":" in s else f"{s}:{default_port}" for s in raw]


def query_lean_servers(server_addresses: list[str]) -> list[tuple[str, int, int]]:
    """Query Lean servers for their ``max_processes`` capacity.

    Args:
        server_addresses: list of ``"host:port"`` strings (use
            :func:`parse_lean_server_addrs` first).

    Returns:
        list of ``(host, port, max_processes)`` tuples.

    Raises:
        ConnectionError: if any server is unreachable or reports 0 processes.
    """
    servers = []
    for addr in server_addresses:
        host, port_str = addr.split(":")
        port = int(port_str)
        try:
            with urlopen(f"http://{host}:{port}/status", timeout=10) as resp:
                status = json_mod.loads(resp.read())
            max_procs = status.get("max_processes", 0)
        except Exception as e:
            raise ConnectionError(
                f"Could not reach Lean server {addr}: {e}"
            ) from e
        if max_procs == 0:
            raise ConnectionError(
                f"Lean server {addr} reports 0 available processes"
            )
        log(f"Lean server {addr}: {max_procs} processes", component="LeanPool")
        servers.append((host, port, max_procs))
    return servers


def assign_lean_servers(
    servers: list[tuple[str, int, int]],
) -> list[tuple[str, int]]:
    """Build a per-actor list of ``(host, port)`` assignments.

    Each Lean process maps to one actor thread (1:1).  Actors ``0..N-1``
    are assigned to servers in order of the server list.

    Returns:
        list of ``(host, port)`` -- one per actor.
        Length equals the total ``max_processes`` across all servers.
    """
    assignments = []
    for host, port, max_procs in servers:
        for _ in range(max_procs):
            assignments.append((host, port))
    return assignments
