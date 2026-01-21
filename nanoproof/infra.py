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
# Maps prover server addresses to lean server addresses
# Each prover should know its own IP and look up which lean server to use
"10.10.25.40" = "10.10.25.31:8000"
"10.10.25.41" = "10.10.25.31:8000"
"10.10.25.42" = "10.10.25.32:8000"
"10.10.25.43" = "10.10.25.32:8000"
```
"""

import tomllib
from dataclasses import dataclass, field
from typing import Optional


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
    
    # Mapping from prover IP addresses to lean server addresses (host:port)
    prover_to_lean: dict[str, str] = field(default_factory=dict)
    
    @property
    def rl_server(self) -> str:
        """Get RL server as 'address:port'."""
        return f"{self.rl_server_address}:{self.rl_server_port}"
    
    def get_lean_server_for_prover(self, prover_address: str) -> Optional[LeanServer]:
        """
        Get the lean server that a prover should use.
        
        Args:
            prover_address: The prover's IP address (without port)
        
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
