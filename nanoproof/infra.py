"""
Infrastructure configuration for distributed RL training.

The infra.toml file defines the distributed setup:
- RL server address and port
- List of lean servers
- Mapping from prover servers to lean servers

Example infra.toml:

```toml
[rl_server]
address = "10.10.25.30"
port = 5000

[[lean_servers]]
address = "10.10.25.31"
port = 8000

[[lean_servers]]
address = "10.10.25.32"
port = 8000

[prover_to_lean]
# Maps prover server addresses (IP:port) to lean server addresses
# Each prover should know its own IP:port and look up which lean server to use
"10.10.25.40:6001" = "10.10.25.31:8000"
"10.10.25.40:6002" = "10.10.25.31:8000"
"10.10.25.41:6001" = "10.10.25.32:8000"
"10.10.25.41:6002" = "10.10.25.32:8000"
```
"""

import threading
import tomllib
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from nanoproof.inference import TacticModel, BlockingTacticModel
    from nanoproof.rl_server import InferenceRouter


@dataclass
class LeanServer:
    """A lean server endpoint."""
    address: str
    port: int
    
    def __str__(self) -> str:
        return f"{self.address}:{self.port}"
    
    @classmethod
    def from_string(cls, s: str) -> "LeanServer":
        """Parse from 'address:port' format."""
        parts = s.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid lean server format: '{s}', expected 'address:port'")
        return cls(address=parts[0], port=int(parts[1]))


def parse_lean_server(s: str) -> LeanServer:
    """Parse a lean server string in 'address:port' format."""
    return LeanServer.from_string(s)


@dataclass
class InfraConfig:
    """Infrastructure configuration for distributed RL training."""
    
    # RL server (coordinator) address and port
    rl_server_address: str
    rl_server_port: int
    
    # List of lean servers available
    lean_servers: list[LeanServer] = field(default_factory=list)
    
    # Mapping from prover addresses (IP:port) to lean server addresses (host:port)
    prover_to_lean: dict[str, str] = field(default_factory=dict)
    
    @property
    def rl_server(self) -> str:
        """Get RL server as 'address:port'."""
        return f"{self.rl_server_address}:{self.rl_server_port}"
    
    def get_lean_server_for_prover(self, prover_address: str) -> Optional[LeanServer]:
        """
        Get the lean server that a prover should use.
        
        Args:
            prover_address: The prover's address as IP:port (e.g. "10.10.25.40:6001")
        
        Returns:
            LeanServer if found in mapping, None otherwise
        """
        lean_addr = self.prover_to_lean.get(prover_address)
        if lean_addr is None:
            return None
        return LeanServer.from_string(lean_addr)
    
    def get_lean_server_list(self) -> list[str]:
        """Get list of lean servers as 'address:port' strings (for monitoring)."""
        return [str(s) for s in self.lean_servers]


def load_infra_config(path: str) -> InfraConfig:
    """
    Load infrastructure configuration from a TOML file.
    
    Args:
        path: Path to the infra.toml file
    
    Returns:
        InfraConfig object
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is malformed
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)
    
    # Parse RL server
    rl_server = data.get("rl_server", {})
    if "address" not in rl_server or "port" not in rl_server:
        raise ValueError("infra.toml must have [rl_server] with 'address' and 'port'")
    
    rl_server_address = rl_server["address"]
    rl_server_port = int(rl_server["port"])
    
    # Parse lean servers (optional, for monitoring)
    lean_servers = []
    for ls in data.get("lean_servers", []):
        if "address" not in ls or "port" not in ls:
            raise ValueError("Each [[lean_servers]] entry must have 'address' and 'port'")
        lean_servers.append(LeanServer(address=ls["address"], port=int(ls["port"])))
    
    # Parse prover to lean mapping
    prover_to_lean = {}
    for prover_addr, lean_addr in data.get("prover_to_lean", {}).items():
        prover_to_lean[prover_addr] = lean_addr
    
    return InfraConfig(
        rl_server_address=rl_server_address,
        rl_server_port=rl_server_port,
        lean_servers=lean_servers,
        prover_to_lean=prover_to_lean,
    )


# -----------------------------------------------------------------------------
# Distributed Infrastructure Setup
# -----------------------------------------------------------------------------

@dataclass
class DistributedInfraHandles:
    """Handles for distributed infrastructure components (for cleanup)."""
    inference_model: "BlockingTacticModel"
    inference_server_thread: threading.Thread
    coordinator_thread: threading.Thread
    inference_router: "InferenceRouter"
    coordinator_port: int
    
    def shutdown(self):
        """Shutdown all distributed infrastructure components."""
        from nanoproof.rl_server import shutdown_coordinator
        shutdown_coordinator()
        self.inference_model.shutdown()


def start_distributed_eval_servers(
    tactic_model: "TacticModel",
    coordinator_port: int,
    batch_timeout: float = 0.2,
    max_batch_tokens: int = 8000,
    startup_timeout: float = 30.0,
) -> DistributedInfraHandles:
    """
    Start distributed evaluation infrastructure (inference server + coordinator).
    
    This is used for standalone evaluation scripts that need to set up the same
    infrastructure as the RL training loop.
    
    Args:
        tactic_model: The TacticModel to use for inference
        coordinator_port: Port for the coordinator to listen on
        batch_timeout: Timeout in seconds for batching LLM calls
        max_batch_tokens: Maximum tokens per inference batch
        startup_timeout: Maximum time to wait for servers to be healthy
    
    Returns:
        DistributedInfraHandles with references to all started components
    """
    from nanoproof.inference import BlockingTacticModel, start_inference_server
    from nanoproof.rl_server import start_coordinator
    from nanoproof.cli import log
    
    # Create BlockingTacticModel for the inference server
    inference_model = BlockingTacticModel(
        inner_model=tactic_model,
        timeout_seconds=batch_timeout,
        max_batch_tokens=max_batch_tokens
    )
    
    # Start inference server on port coordinator_port + 1
    inference_port = coordinator_port + 1
    inference_server_thread = start_inference_server(inference_model, inference_port)
    
    log(f"Started inference server on port {inference_port}", component="Infra")
    
    # Start the coordinator (proxies inference + handles prover registration)
    inference_endpoints = [f"http://127.0.0.1:{inference_port}"]
    coordinator_thread, inference_router = start_coordinator(
        coordinator_port, inference_endpoints, startup_timeout=startup_timeout
    )
    
    log(f"Started coordinator on port {coordinator_port}", component="Infra")
    
    return DistributedInfraHandles(
        inference_model=inference_model,
        inference_server_thread=inference_server_thread,
        coordinator_thread=coordinator_thread,
        inference_router=inference_router,
        coordinator_port=coordinator_port,
    )
