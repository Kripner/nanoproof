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
        max_prompt_len = self.network.config.sequence_len * 9

        # Prepare tokenized prompts, tracking which ones are too long
        prompts = []
        too_long_indices = set()
        for idx, state_str in enumerate(state_strs):
            tokens = self.tokenizer(state_str + "\n<|tactic|>", prepend=self.tokenizer.get_bos_token_id())
            if len(tokens) > max_prompt_len:
                too_long_indices.add(idx)
                prompts.append([self.tokenizer.get_bos_token_id()])
            else:
                prompts.append(tokens)

        seed = torch.randint(torch.iinfo(torch.int32).max, (1,), device=device, generator=self.rng).item()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
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
    
    Batches are processed when either:
    1. pending items exceed max_batch_tokens, or
    2. timeout_seconds has elapsed since the first pending request
    """

    def __init__(self, inner_model: TacticModel, timeout_seconds: float, max_batch_tokens: int):
        self.inner_model = inner_model
        self.timeout_seconds = timeout_seconds
        self.max_batch_tokens = max_batch_tokens

        self._lock = threading.Lock()
        self._batch_ready = threading.Condition(self._lock)
        self._pending: list[tuple[str, threading.Event, list]] = []
        self._first_request_time: float | None = None
        self._batch_in_progress = False
        self._shutdown = False
        self._paused = False
        self._total_batches = 0

    @property
    def network(self):
        return self.inner_model.network

    def shutdown(self):
        """Signal shutdown to unblock all waiting threads."""
        with self._lock:
            self._shutdown = True
            for _, event, slot in self._pending:
                slot.append(ValueOrError.from_error("Model shutdown"))
                event.set()
            self._pending = []
            self._batch_ready.notify_all()
    
    def pause(self):
        """Pause inference and wait for any in-progress batch to complete.
        
        After this returns, no new batches will be processed until resume() is called.
        Safe to call multiple times.
        """
        with self._lock:
            self._paused = True
            # Wait for any in-progress batch to complete
            while self._batch_in_progress:
                self._batch_ready.wait(timeout=0.1)
    
    def resume(self):
        """Resume inference after pause()."""
        with self._lock:
            self._paused = False
            self._batch_ready.notify_all()

    def sample_tactic(self, state: State) -> ValueOrError[list[str]]:
        """Thread-safe sample_tactic that batches calls from multiple threads."""
        assert len(state) == 1
        return self.sample_tactic_from_str(str(state[0].state).strip())

    def sample_tactic_from_str(self, state_str: str) -> ValueOrError[list[str]]:
        """Thread-safe tactic generation from a state string."""
        results = self.sample_tactic_from_str_batch([state_str])
        return results[0]

    def sample_tactic_from_str_batch(self, state_strs: list[str]) -> list[ValueOrError[list[str]]]:
        """
        Thread-safe batch tactic generation. All states are queued together
        and will be processed in the same or consecutive GPU batches.
        """
        if not state_strs:
            return []
        if self._shutdown:
            return [ValueOrError.from_error("Model shutdown") for _ in state_strs]
        if self._paused:
            return [ValueOrError.from_error("Model paused for training") for _ in state_strs]

        # Create events and slots for all states
        entries = [(s, threading.Event(), []) for s in state_strs]

        with self._lock:
            if self._shutdown:
                return [ValueOrError.from_error("Model shutdown") for _ in state_strs]
            if self._paused:
                return [ValueOrError.from_error("Model paused for training") for _ in state_strs]

            # Add all to pending queue
            for entry in entries:
                self._pending.append(entry)
            
            if self._first_request_time is None:
                self._first_request_time = time.time()

            # Trigger batch if we have enough tokens queued
            if not self._batch_in_progress and self._pending_tokens() >= self.max_batch_tokens:
                self._process_batch_locked()

        # Wait for all results
        for _, event, _ in entries:
            self._wait_for_result(event)

        return [slot[0] if slot else ValueOrError.from_error("No result") for _, _, slot in entries]

    def _pending_tokens(self) -> int:
        """Estimate total tokens in pending queue (assumes padding to max length)."""
        if not self._pending:
            return 0
        # Estimate tokens as chars/4 (rough average for code/math text)
        max_len = max(len(item[0]) // 4 + 1 for item in self._pending)
        # Memory scales with max_len * batch_size * num_samples (due to padding + sample expansion)
        return max_len * len(self._pending) * self.inner_model.num_samples

    def _wait_for_result(self, event: threading.Event):
        """Wait for a result, triggering batch processing if needed."""
        while not event.is_set():
            with self._lock:
                if self._shutdown or self._paused or event.is_set():
                    return
                
                # Check if we should trigger processing
                if self._first_request_time is not None:
                    elapsed = time.time() - self._first_request_time
                    should_trigger = elapsed >= self.timeout_seconds or self._pending_tokens() >= self.max_batch_tokens
                    if should_trigger and not self._batch_in_progress and self._pending:
                        self._process_batch_locked()
                        continue

                self._batch_ready.wait(timeout=0.1)

    def _process_batch_locked(self):
        """Process pending batch. Must be called with lock held."""
        if not self._pending or self._batch_in_progress:
            return

        self._batch_in_progress = True
        self._total_batches += 1
        batch_num = self._total_batches

        # Take all pending items
        batch = self._pending[:]
        approx_tokens = self._pending_tokens()
        self._pending = []
        self._first_request_time = None

        # log(f"Batch #{batch_num}: processing {len(batch)} requests", component="BlockingTacticModel")

        # Release lock during inference
        self._lock.release()
        try:
            state_strs = [item[0] for item in batch]
            start = time.time()
            results = self.inner_model.sample_tactic_from_str_batch(state_strs)
            elapsed = time.time() - start
            
            # log(f"Batch #{batch_num}: processed {len(batch)} samples (~{approx_tokens} tokens), took {elapsed:.1f}s", component="BlockingTacticModel")
            # if elapsed > 2.0:
            #     log(f"Batch #{batch_num}: took {elapsed:.1f}s", component="BlockingTacticModel")

            # Distribute results
            for i, (_, event, slot) in enumerate(batch):
                slot.append(ValueOrError.from_success(results[i]))
                event.set()
                
        except Exception as e:
            log(f"Batch #{batch_num}: FAILED - {e}", component="BlockingTacticModel")
            for _, event, slot in batch:
                slot.append(ValueOrError.from_error(str(e)))
                event.set()
        finally:
            self._lock.acquire()
            self._batch_in_progress = False
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
            self._record_failure(f"Timeout calling inference server at {self.server_address} (timeout={self.timeout})")
            return []
        except http_requests.exceptions.RequestException as e:
            self._record_failure(f"Error calling inference server at {self.server_address}: {e}")
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
        if not states:
            return jsonify({"tactics": []})
        # Submit all states at once - they'll be batched together
        results = model.sample_tactic_from_str_batch(states)
        tactics = [r.value if r.is_success() else [] for r in results]
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
    parser.add_argument("--batch-timeout", type=float, default=0.1, help="Timeout for batching (seconds)")
    parser.add_argument("--max-batch-tokens", type=int, default=32000, help="Max total tokens per batch (controls memory)")
    parser.add_argument("--model-source", default="sft", help="Model source (sft, pretrain, etc)")
    parser.add_argument("--model-tag", default="d26", help="Model tag")
    args = parser.parse_args()
    
    print(f"[Inference] Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = load_model(args.model_source, device, phase="eval", model_tag=args.model_tag)
    engine = Engine(model, tokenizer)
    tactic_model = TacticModel(model, tokenizer, engine, num_samples=args.num_samples)
    blocking_model = BlockingTacticModel(
        tactic_model, 
        timeout_seconds=args.batch_timeout,
        max_batch_tokens=args.max_batch_tokens
    )
    
    app = create_blocking_model_app(blocking_model)
    
    print(f"[Inference] Server starting on {args.host}:{args.port}")
    print(f"[Inference] Device: {device}")
    print(f"[Inference] Samples per state: {args.num_samples}")
    print(f"[Inference] Timeout: {args.batch_timeout}s, max tokens: {args.max_batch_tokens}")
    
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()

