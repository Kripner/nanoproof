"""
Inference - provides tactic models and batched inference for prover servers.

This module contains:
- TacticModel: Core tactic generation model wrapping the Transformer
- BlockingTacticModel: Thread-safe wrapper that batches LLM calls from multiple threads
- RemoteTacticModel: Client that calls a remote inference server (for CPU-only nodes)
- start_inference_server: Starts a Flask inference server for a BlockingTacticModel

The inference server (Flask) runs on GPU nodes and handles tactic generation requests
from multiple prover servers. It batches requests for efficient GPU utilization.

For multi-GPU setups, use DDP mode which runs separate inference servers per GPU
with the coordinator (rl_server.py) handling load balancing.

Usage:
    python -m nanoproof.inference --port 5000
"""

import argparse
import threading
import time
from dataclasses import dataclass
from typing import Self, TYPE_CHECKING

import requests as http_requests
import torch
from flask import Flask, request, jsonify

from nanoproof.checkpoints import load_model
from nanoproof.cli import get_monitor, log
from nanoproof.common import ValueOrError
from nanoproof.engine import Engine

# Heavy imports are done lazily inside TacticModel to speed up prover_server startup
# These are only imported when TacticModel.create() is called:
# - torch
# - nanoproof.checkpoints (load_model)
# - nanoproof.engine (Engine)
# - nanoproof.model (Transformer)
# - nanoproof.tokenizer (HuggingFaceTokenizer)

if TYPE_CHECKING:
    # For type hints only - not actually imported at runtime
    from nanoproof.model import Transformer
    from nanoproof.tokenizer import HuggingFaceTokenizer


# Type alias for State - avoid circular import with search.py
State = list  # list[LeanProofBranch]


# -----------------------------------------------------------------------------
# Tactic Models
# -----------------------------------------------------------------------------

@dataclass
class TacticModel:
    # Type hints use strings for lazy imports
    network: "Transformer"
    tokenizer: "HuggingFaceTokenizer"
    engine: "Engine"
    num_samples: int = 6

    def __post_init__(self):
        import torch
        self.rng = torch.Generator(device=self.network.get_device())
        self.rng.manual_seed(0)

    def sample_tactic(self, state: State) -> list[str]:
        import torch
        assert len(state) == 1, \
            f"expected single branch in state when generating tactic, got {len(state)} - choose one goal first"
        device = self.network.get_device()
        assert device.type == "cuda"

        state_str = str(state[0].state).strip()
        tokens = self.tokenizer(state_str + "\n<|tactic|>", prepend=self.tokenizer.get_bos_token_id())
        
        # Check if state is too long for the model's rotary cache
        max_prompt_len = self.network.config.sequence_len * 9
        if len(tokens) > max_prompt_len:
            return []  # State too long, return no tactics
        
        seed = torch.randint(torch.iinfo(torch.int32).max, (1,), device=device, generator=self.rng).item()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            sample_toks, masks = self.engine.generate_batch(
                tokens, num_samples=self.num_samples, min_tokens=1, max_tokens=64, seed=seed
            )
        tactics = []
        for i in range(self.num_samples):
            tactic_toks = [token for token, mask in zip(sample_toks[i], masks[i]) if mask == 1]
            tactic = self.tokenizer.decode(tactic_toks)
            if "sorry" in tactic or "admit" in tactic:
                continue
            tactics.append(tactic)
        return tactics

    def sample_tactic_batch(self, states: list[State]) -> list[list[str]]:
        """Batched version of sample_tactic for multiple states at once."""
        state_strs = []
        for state in states:
            assert len(state) == 1, \
                f"expected single branch in state when generating tactic, got {len(state)} - choose one goal first"
            state_strs.append(str(state[0].state).strip())
        return self.sample_tactic_from_str_batch(state_strs)

    def sample_tactic_from_str(self, state_str: str) -> list[str]:
        """Generate tactics from a state string directly (no State object needed)."""
        return self.sample_tactic_from_str_batch([state_str])[0]

    def sample_tactic_from_str_batch(self, state_strs: list[str]) -> list[list[str]]:
        """Batched tactic generation from state strings directly."""
        import torch
        device = self.network.get_device()
        assert device.type == "cuda"

        # Maximum prompt length: leave room for generated tokens (64) within rotary cache
        # Rotary cache is sequence_len * 10, so we use sequence_len * 9 as safe max
        max_prompt_len = self.network.config.sequence_len * 9

        # Prepare tokenized prompts, tracking which ones are too long
        prompts = []
        too_long_indices = set()
        for idx, state_str in enumerate(state_strs):
            tokens = self.tokenizer(state_str + "\n<|tactic|>", prepend=self.tokenizer.get_bos_token_id())
            if len(tokens) > max_prompt_len:
                # State is too long - use a placeholder (will return empty tactics)
                too_long_indices.add(idx)
                prompts.append([self.tokenizer.get_bos_token_id()])  # Minimal valid prompt
            else:
                prompts.append(tokens)

        seed = torch.randint(torch.iinfo(torch.int32).max, (1,), device=device, generator=self.rng).item()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            # prompts is list[list[int]], which triggers batched generation
            sample_toks_batch, masks_batch = self.engine.generate_batch(
                prompts, num_samples=self.num_samples, min_tokens=1, max_tokens=64, seed=seed
            )

        # sample_toks_batch shape: (num_prompts, num_samples, seq_len)
        results = []
        for prompt_idx in range(len(state_strs)):
            # Return empty tactics for states that were too long
            if prompt_idx in too_long_indices:
                results.append([])
                continue
            
            tactics = []
            for sample_idx in range(self.num_samples):
                tactic_toks = [
                    token for token, mask in zip(
                        sample_toks_batch[prompt_idx][sample_idx],
                        masks_batch[prompt_idx][sample_idx]
                    ) if mask == 1
                ]
                tactic = self.tokenizer.decode(tactic_toks)
                if "sorry" in tactic or "admit" in tactic:
                    continue
                tactics.append(tactic)
            results.append(tactics)
        return results

    def shutdown(self):
        """No-op for non-batched model. Exists for API compatibility with BlockingTacticModel."""
        pass

    @classmethod
    def create(cls, num_samples: int = 6) -> Self:
        # Lazy imports - only load heavy dependencies when actually creating the model
        import torch
        from nanoproof.checkpoints import load_model
        from nanoproof.engine import Engine
        
        source = "sft"  # which checkpoint to load the model from
        model_tag = "d26"  # model tag to load the model from
        device = torch.device("cuda")

        model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=model_tag)
        engine = Engine(model, tokenizer)
        return cls(model, tokenizer, engine, num_samples=num_samples)


class BlockingTacticModel:
    """
    Thread-safe wrapper around TacticModel that batches LLM calls from multiple threads.
    
    When sample_tactic is called, it blocks until either:
    1. batch_size inputs have been accumulated, or
    2. timeout_seconds has elapsed since the first pending request
    
    Then a single batched LLM call is made and results are distributed to waiting threads.
    """

    def __init__(self, inner_model: TacticModel, batch_size: int = 32, timeout_seconds: float = 0.1):
        self.inner_model = inner_model
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds

        # Synchronization primitives
        self._lock = threading.Lock()
        self._batch_ready = threading.Condition(self._lock)

        # Pending requests: list of (state_str, result_event, result_slot)
        self._pending: list[tuple[str, threading.Event, list]] = []
        self._first_request_time: float | None = None
        self._batch_in_progress = False
        
        # Shutdown flag to unblock waiting threads
        self._shutdown = False

        # Stats for logging
        self._total_requests = 0
        self._total_batches = 0
        log(f"Initialized with batch_size={batch_size}, timeout={timeout_seconds}s", component="BlockingTacticModel")

    @property
    def network(self):
        """Expose network for compatibility with code that accesses model.network."""
        return self.inner_model.network
    
    def is_batch_in_progress(self) -> bool:
        """Check if a batch is currently being processed (for load balancing)."""
        with self._lock:
            return self._batch_in_progress

    def shutdown(self):
        """
        Signal shutdown to unblock all waiting threads.
        Waiting threads will receive ValueOrError with error.
        """
        with self._lock:
            self._shutdown = True
            # Set all pending result events to unblock waiting threads
            for _, result_event, result_slot in self._pending:
                result_slot.append(ValueOrError.from_error("Model shutdown"))
                result_event.set()
            self._pending = []
            self._batch_ready.notify_all()
        log("Shutdown initiated, all waiting threads unblocked", component="BlockingTacticModel")

    def sample_tactic(self, state: State) -> ValueOrError[list[str]]:
        """
        Thread-safe sample_tactic that batches calls from multiple threads.
        Blocks until result is available.
        
        Returns ValueOrError with error if shutdown is in progress or an error occurs.
        """
        assert len(state) == 1, \
            f"expected single branch in state when generating tactic, got {len(state)} - choose one goal first"
        state_str = str(state[0].state).strip()
        return self.sample_tactic_from_str(state_str)

    def sample_tactic_from_str(self, state_str: str) -> ValueOrError[list[str]]:
        """
        Thread-safe tactic generation from a state string.
        Blocks until result is available.
        
        Returns ValueOrError with error if shutdown is in progress or an error occurs.
        """
        if self._shutdown:
            return ValueOrError.from_error("Model shutdown")
            
        wait_start = time.time()
        result_event = threading.Event()
        result_slot: list[ValueOrError[list[str]]] = []  # Will hold the result

        with self._lock:
            if self._shutdown:
                return ValueOrError.from_error("Model shutdown")
                
            self._total_requests += 1
            request_num = self._total_requests
            pending_count = len(self._pending) + 1

            # Add this request to the pending batch
            self._pending.append((state_str, result_event, result_slot))

            if self._first_request_time is None:
                self._first_request_time = time.time()
                log(f"Request #{request_num}: first in batch", component="BlockingTacticModel")

            # Log every 10th request or when batch is full
            if request_num % 10 == 0 or pending_count >= self.batch_size:
                log(f"Request #{request_num}: pending={pending_count}/{self.batch_size}",
                    component="BlockingTacticModel")

            # Check if we should trigger the batch
            should_process = len(self._pending) >= self.batch_size

            if should_process and not self._batch_in_progress:
                self._process_batch_locked()

        # If not processing immediately, wait for batch to be ready or timeout
        if not result_event.is_set():
            self._wait_for_batch_or_timeout(result_event)

        # Wait for the result with timeout to avoid hanging forever
        # Use a loop with short timeouts to stay responsive to shutdown
        max_wait = 60.0  # Maximum total wait time
        wait_interval = 1.0  # Check for shutdown every second
        total_waited = 0.0
        while not result_event.is_set() and total_waited < max_wait:
            if self._shutdown:
                return ValueOrError.from_error("Model shutdown")
            result_event.wait(timeout=wait_interval)
            total_waited += wait_interval
        
        if not result_event.is_set():
            log(f"Request timed out after {max_wait}s", component="BlockingTacticModel")
            return ValueOrError.from_error("Request timed out")

        # Record wait time for monitoring
        wait_time = time.time() - wait_start
        monitor = get_monitor()
        if monitor is not None:
            monitor.record_batch_wait(wait_time)

        return result_slot[0] if result_slot else ValueOrError.from_error("No result received")

    def _wait_for_batch_or_timeout(self, my_event: threading.Event):
        """Wait until batch is ready or timeout, then potentially trigger processing."""
        max_iterations = 100  # Prevent infinite loop
        iterations = 0
        while not my_event.is_set() and iterations < max_iterations:
            iterations += 1
            with self._lock:
                if self._shutdown:
                    return
                    
                if my_event.is_set():
                    return

                if self._first_request_time is None:
                    # Batch was already processed, but our event should be set
                    # Wait briefly and check again (handles race condition)
                    self._batch_ready.wait(timeout=0.1)
                    continue

                elapsed = time.time() - self._first_request_time
                remaining = self.timeout_seconds - elapsed

                if remaining <= 0 or len(self._pending) >= self.batch_size:
                    # Time to process
                    if not self._batch_in_progress and len(self._pending) > 0:
                        self._process_batch_locked()
                    return

                # Wait with timeout
                self._batch_ready.wait(timeout=min(remaining, 0.5))  # Cap wait to check shutdown

    def _process_batch_locked(self):
        """Process the current batch. Must be called with lock held."""
        if len(self._pending) == 0:
            return

        self._batch_in_progress = True
        self._total_batches += 1
        batch_num = self._total_batches

        # Grab the current batch
        batch = self._pending[:]
        self._pending = []
        self._first_request_time = None

        log(f"Batch #{batch_num}: processing {len(batch)} requests", component="BlockingTacticModel")

        # Release lock during LLM inference
        self._lock.release()
        results = None
        inference_error = None
        try:
            state_strs = [item[0] for item in batch]
            results = self.inner_model.sample_tactic_from_str_batch(state_strs)
        except Exception as e:
            inference_error = e
            log(f"Batch #{batch_num}: LLM inference FAILED: {type(e).__name__}: {e}", component="BlockingTacticModel")
        finally:
            self._lock.acquire()
            # CRITICAL: Always reset _batch_in_progress, even on error
            self._batch_in_progress = False

        # Distribute results to waiting threads
        for i, (_, result_event, result_slot) in enumerate(batch):
            if results is not None:
                result_slot.append(ValueOrError.from_success(results[i]))
            else:
                # On error, return error so caller knows inference failed
                error_msg = f"LLM inference failed: {type(inference_error).__name__}: {inference_error}"
                result_slot.append(ValueOrError.from_error(error_msg))
            result_event.set()

        if inference_error:
            log(f"Batch #{batch_num}: returned errors to {len(batch)} threads due to error", component="BlockingTacticModel")
        else:
            log(f"Batch #{batch_num}: results distributed to {len(batch)} threads", component="BlockingTacticModel")

        # Notify any threads waiting for this batch
        self._batch_ready.notify_all()


# -----------------------------------------------------------------------------
# Remote Tactic Model (for CPU-only prover nodes)
# -----------------------------------------------------------------------------

class RLServerDisconnectedError(Exception):
    """Raised when the RL server appears to be permanently disconnected."""
    pass


class RemoteTacticModel:
    """
    Tactic model that calls a remote inference server for tactic generation.
    
    This is used by prover servers running on CPU-only nodes.
    The inference server handles batching and GPU inference.
    """
    
    def __init__(self, server_address: str, timeout: float = 60.0, max_pool_size: int = 50):
        self.server_address = server_address
        self.timeout = timeout
        # Configure session with larger connection pool for many concurrent actors
        self._session = http_requests.Session()
        adapter = http_requests.adapters.HTTPAdapter(
            pool_connections=max_pool_size,
            pool_maxsize=max_pool_size,
            max_retries=0,  # We handle retries ourselves
        )
        self._session.mount('http://', adapter)
        self._session.mount('https://', adapter)
        self._consecutive_failures = 0
        self._max_failures = 10  # After this many failures, raise exception
        self._last_error_log_time = 0
        self._error_log_interval = 5.0  # Only log errors every N seconds
        self._lock = threading.Lock()
    
    def sample_tactic(self, state) -> list[str]:
        """Sample tactics for a single state by calling the remote inference server."""
        assert len(state) == 1, \
            f"expected single branch in state, got {len(state)}"
        
        state_str = str(state[0].state).strip()
        
        try:
            response = self._session.post(
                f"http://{self.server_address}/generate",
                json={"states": [state_str]},
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            # Reset failure counter on success
            with self._lock:
                self._consecutive_failures = 0
            
            return result["tactics"][0]
        except http_requests.exceptions.Timeout:
            self._record_failure("Timeout calling inference server")
            return []
        except http_requests.exceptions.RequestException as e:
            self._record_failure(f"Error calling inference server: {e}")
            return []
    
    def _record_failure(self, message: str):
        """Record a failure and potentially raise if too many consecutive failures."""
        with self._lock:
            self._consecutive_failures += 1
            failures = self._consecutive_failures
            
            # Rate-limit error logging
            now = time.time()
            if now - self._last_error_log_time >= self._error_log_interval:
                print(f"[RemoteTacticModel] {message} (failures: {failures}/{self._max_failures})")
                self._last_error_log_time = now
            
            # If too many failures, signal that server is down
            if failures >= self._max_failures:
                raise RLServerDisconnectedError(
                    f"Inference server appears disconnected after {failures} consecutive failures"
                )
    
    def shutdown(self):
        """No-op, exists for API compatibility."""
        pass


# -----------------------------------------------------------------------------
# Flask Server
# -----------------------------------------------------------------------------

def create_blocking_model_app(model: BlockingTacticModel):
    """Create Flask app for a single BlockingTacticModel (used in DDP mode, one per rank)."""
    app = Flask(__name__)
    
    # Disable Flask request logging to reduce spam
    import logging
    log_flask = logging.getLogger('werkzeug')
    log_flask.setLevel(logging.ERROR)
    
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})
    
    @app.route("/generate", methods=["POST"])
    def generate():
        data = request.get_json()
        states = data.get("states", [])
        if len(states) == 0:
            return jsonify({"tactics": []})
        # Generate tactics for each state (BlockingTacticModel handles batching)
        tactics = []
        for state_str in states:
            result = model.sample_tactic_from_str(state_str)
            tactics.append(result.value if result.is_success() else [])
        return jsonify({"tactics": tactics})
    
    return app


def start_inference_server(model: BlockingTacticModel, port: int, host: str = "0.0.0.0"):
    """
    Start inference server for a single BlockingTacticModel in a background thread.
    
    Used in DDP mode where each rank runs one inference server on its GPU.
    Returns the background thread.
    """
    app = create_blocking_model_app(model)
    
    def run_server():
        import logging
        log_flask = logging.getLogger('werkzeug')
        log_flask.setLevel(logging.ERROR)
        app.run(host=host, port=port, threaded=True)
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    log(f"Inference server started on port {port}", component="InferenceServer")
    return thread


# -----------------------------------------------------------------------------
# Main (standalone inference server)
# -----------------------------------------------------------------------------

def main():
    """
    Standalone inference server using a single BlockingTacticModel.
    
    For multi-GPU setups, use DDP mode in rl.py which runs separate
    inference servers per GPU with a coordinator for load balancing.
    """
    parser = argparse.ArgumentParser(description="Inference Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--num-samples", type=int, default=6, help="Tactics to sample per state")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for batching requests")
    parser.add_argument("--batch-timeout", type=float, default=0.1, help="Timeout for batching (seconds)")
    parser.add_argument("--model-source", default="sft", help="Model source (sft, pretrain, etc)")
    parser.add_argument("--model-tag", default="d26", help="Model tag")
    args = parser.parse_args()
    
    print(f"[Inference] Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = load_model(args.model_source, device, phase="eval", model_tag=args.model_tag)
    engine = Engine(model, tokenizer)
    tactic_model = TacticModel(model, tokenizer, engine, num_samples=args.num_samples)
    blocking_model = BlockingTacticModel(tactic_model, batch_size=args.batch_size, timeout_seconds=args.batch_timeout)
    
    app = create_blocking_model_app(blocking_model)
    
    print(f"[Inference] Server starting on {args.host}:{args.port}")
    print(f"[Inference] Device: {device}")
    print(f"[Inference] Samples per state: {args.num_samples}")
    print(f"[Inference] Batch size: {args.batch_size}, timeout: {args.batch_timeout}s")
    
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()

