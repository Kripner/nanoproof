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
from nanoproof.tokenizer import _MIN_VALUE, _MAX_VALUE

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

    def sample_tactic(self, state: State) -> ValueOrError[list[str]]:
        assert len(state) == 1, \
            f"expected single branch in state when generating tactic, got {len(state)} - choose one goal first"
        device = self.network.get_device()
        assert device.type == "cuda"

        state_str = str(state[0].state).strip()
        tokens = self.tokenizer(state_str + "\n<|tactic|>", prepend=self.tokenizer.get_bos_token_id())
        
        # Check if state is too long for the model's rotary cache
        max_prompt_len = self.network.config.sequence_len * 9
        if len(tokens) > max_prompt_len:
            return ValueOrError.from_error("State too long for model's rotary cache")
        
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
        return ValueOrError.from_success(tactics)

    def sample_tactic_from_str(self, state_str: str) -> ValueOrError[list[str]]:
        return self.sample_tactic_from_str_batch([state_str])[0]

    def sample_tactic_from_str_batch(self, state_strs: list[str]) -> list[ValueOrError[list[str]]]:
        """Batched tactic generation from state strings directly."""
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
                prompts.append([self.tokenizer.get_bos_token_id()])  # dummy prompt
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
                results.append(ValueOrError.from_error("State too long for model's rotary cache"))
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
            results.append(ValueOrError.from_success(tactics))
        return results

    def predict_value(self, state: State) -> ValueOrError[float]:
        assert len(state) == 1, \
            f"expected single branch in state when predicting value, got {len(state)} - choose one goal first"
        return self.predict_value_from_str_batch([str(state[0].state).strip()])[0]

    def predict_value_from_str(self, state_str: str) -> ValueOrError[float]:
        return self.predict_value_from_str_batch([state_str])[0]

    def predict_value_from_str_batch(self, state_strs: list[str]) -> list[ValueOrError[float]]:
        device = self.network.get_device()
        assert device.type == "cuda"

        # Get special tokens
        value_delim_tok = self.tokenizer.encode_special("<|value|>")
        bin_token_ids = self.tokenizer.get_value_token_ids()

        # Maximum prompt length (same as tactic generation)
        max_prompt_len = self.network.config.sequence_len * 9

        # Tokenize all states
        prompts = []
        too_long_indices = set()
        for idx, state_str in enumerate(state_strs):
            tokens = self.tokenizer(state_str + "\n<|value|>", prepend=self.tokenizer.get_bos_token_id())
            if len(tokens) > max_prompt_len:
                too_long_indices.add(idx)
                # Use a minimal prompt - will return default value
                prompts.append([self.tokenizer.get_bos_token_id(), value_delim_tok])
            else:
                prompts.append(tokens)

        # Forward pass
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, _, value_logits = self.engine.generate_batch(
                prompts, num_samples=1, min_tokens=1, max_tokens=1, return_logits=True
            )

        # Get logits at the generated position (first token after <|value|>). Since prompts are left-padded,
        # the last position in each sequence contains the logits for the bin prediction.
        # So for each prompt, we select the first sample (there is just one sample), and the last position.
        value_logits = torch.stack([value_logits[i][0][-1] for i in range(len(prompts))])  # (B, V)

        # Extract bin logits and compute soft predictions
        bin_probs = torch.softmax(value_logits.float(), dim=-1)  # (B, V)
        bin_probs = bin_probs[:, bin_token_ids]  # (B, 64)
        bin_values = torch.arange(_MIN_VALUE, _MAX_VALUE + 1, dtype=bin_probs.dtype, device=device)
        values = (bin_probs * bin_values).sum(dim=-1)  # (B,)

        results = []
        for idx, value in enumerate(values.tolist()):
            if idx in too_long_indices:
                # State too long to process - return pessimistic value
                results.append(ValueOrError.from_error("State too long to process"))
            else:
                results.append(ValueOrError.from_success(float(value)))
        return results

    def shutdown(self):
        """No-op for non-batched model. Exists for API compatibility with BlockingTacticModel."""
        pass

    @classmethod
    def create(cls, num_samples: int = 6, model_tag: str = "d26", step: int | None = None) -> Self:
        # Lazy imports - only load heavy dependencies when actually creating the model
        
        source = "sft"  # which checkpoint to load the model from
        device = torch.device("cuda")

        model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
        engine = Engine(model, tokenizer)
        return cls(model, tokenizer, engine, num_samples=num_samples)


class RequestQueue:
    """
    Manages a queue of pending requests for batched processing.
    
    This class handles the queue state (pending items, timing) but does not
    own synchronization primitives - those are managed by the parent BlockingTacticModel.
    """
    
    def __init__(self, name: str, num_samples_for_estimation: int):
        """
        Args:
            name: Queue name for logging
            num_samples_for_estimation: Number of samples used for token estimation
                (e.g., 6 for tactics, 1 for values)
        """
        self.name = name
        self.num_samples = num_samples_for_estimation
        self._pending: list[tuple[str, threading.Event, list]] = []
        self._first_request_time: float | None = None
        self._total_batches = 0
    
    def add(self, entries: list[tuple[str, threading.Event, list]]):
        """Add entries to the pending queue. Sets first_request_time if queue was empty."""
        if self._first_request_time is None and entries:
            self._first_request_time = time.time()
        self._pending.extend(entries)
    
    def take_batch(self) -> list[tuple[str, threading.Event, list]]:
        """Remove and return all pending items. Increments batch counter."""
        batch = self._pending[:]
        self._pending = []
        self._first_request_time = None
        self._total_batches += 1
        return batch
    
    def pending_tokens(self) -> int:
        """Estimate total tokens in pending queue (assumes padding to max length)."""
        if not self._pending:
            return 0
        # Estimate tokens as chars/4 (rough average for code/math text)
        max_len = max(len(item[0]) // 4 + 1 for item in self._pending)
        # Memory scales with max_len * batch_size * num_samples (due to padding + sample expansion)
        return max_len * len(self._pending) * self.num_samples
    
    def should_process(self, timeout_seconds: float, max_batch_tokens: int) -> bool:
        """Check if batch should be processed based on timeout or token count."""
        if not self._pending or self._first_request_time is None:
            return False
        elapsed = time.time() - self._first_request_time
        return elapsed >= timeout_seconds or self.pending_tokens() >= max_batch_tokens
    
    def has_pending(self) -> bool:
        return len(self._pending) > 0
    
    def shutdown_all(self):
        """Signal shutdown to all pending requests."""
        for _, event, slot in self._pending:
            slot.append(ValueOrError.from_error("Model shutdown"))
            event.set()
        self._pending = []
        self._first_request_time = None


class BlockingTacticModel:
    """
    Thread-safe wrapper around TacticModel that batches LLM calls from multiple threads.
    
    Batches are processed when either:
    1. pending items exceed max_batch_tokens, or
    2. timeout_seconds has elapsed since the first pending request
    
    Supports both tactic generation and value prediction through separate request queues.
    """

    def __init__(self, inner_model: TacticModel, timeout_seconds: float, max_batch_tokens: int):
        self.inner_model = inner_model
        self.timeout_seconds = timeout_seconds
        self.max_batch_tokens = max_batch_tokens

        # Shared synchronization primitives
        self._lock = threading.Lock()
        self._batch_ready = threading.Condition(self._lock)
        self._batch_in_progress = False
        self._shutdown = False
        self._paused = False

        # Separate queues for tactics and values
        self._tactic_queue = RequestQueue("tactics", self.inner_model.num_samples)
        self._value_queue = RequestQueue("values", 1)  # Value prediction uses num_samples=1

    @property
    def network(self):
        return self.inner_model.network

    def shutdown(self):
        """Signal shutdown to unblock all waiting threads."""
        with self._lock:
            self._shutdown = True
            self._tactic_queue.shutdown_all()
            self._value_queue.shutdown_all()
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
        """Sample tactics for a single state string."""
        results = self.sample_tactic_from_str_batch([state_str])
        return results[0]

    def sample_tactic_from_str_batch(self, state_strs: list[str]) -> list[ValueOrError[list[str]]]:
        """
        Thread-safe batch tactic generation. All states are queued together
        and will be processed in the same or consecutive GPU batches.
        """
        return self._submit_batch(
            state_strs,
            self._tactic_queue,
            self.inner_model.sample_tactic_from_str_batch
        )

    def predict_value(self, state: State) -> ValueOrError[float]:
        """Thread-safe value prediction that batches calls from multiple threads."""
        assert len(state) == 1, \
            f"expected single branch in state when predicting value, got {len(state)} - choose one goal first"
        return self.predict_value_from_str(str(state[0].state).strip())

    def predict_value_from_str(self, state_str: str) -> ValueOrError[float]:
        """Predict value for a single state string."""
        results = self.predict_value_from_str_batch([state_str])
        return results[0]

    def predict_value_from_str_batch(self, state_strs: list[str]) -> list[ValueOrError[float]]:
        """
        Thread-safe batch value prediction. All states are queued together
        and will be processed in the same or consecutive GPU batches.
        """
        return self._submit_batch(
            state_strs,
            self._value_queue,
            self.inner_model.predict_value_from_str_batch
        )

    def _submit_batch(
        self,
        state_strs: list[str],
        queue: RequestQueue,
        process_fn
    ) -> list:
        """
        Submit a batch of requests to the specified queue and wait for results.
        
        Args:
            state_strs: List of state strings to process
            queue: The RequestQueue to submit to
            process_fn: Function to call for batch processing (from inner_model)
        
        Returns:
            List of results (ValueOrError objects)
        """
        if not state_strs:
            return []

        with self._lock:
            if self._shutdown:
                return [ValueOrError.from_error("Model shutdown") for _ in state_strs]
            if self._paused:
                return [ValueOrError.from_error("Model paused for training") for _ in state_strs]

            # Create events and slots for all states
            entries = [(s, threading.Event(), []) for s in state_strs]

            # Add all to pending queue
            queue.add(entries)

            # Trigger batch if we have enough tokens queued
            if not self._batch_in_progress and queue.should_process(self.timeout_seconds, self.max_batch_tokens):
                self._process_queue_locked(queue, process_fn)

        # Wait for all results
        for _, event, _ in entries:
            self._wait_for_result(event, queue, process_fn)

        # If we shutdown/paused, _wait_for_result returns early. Here we return nicer messages instead of "No result".
        if self._shutdown:
            return [ValueOrError.from_error("Model shutdown") for _ in state_strs]
        if self._paused:
            return [ValueOrError.from_error("Model paused for training") for _ in state_strs]

        return [slot[0] if slot else ValueOrError.from_error("No result") for _, _, slot in entries]

    def _wait_for_result(self, event: threading.Event, queue: RequestQueue, process_fn):
        """Wait for a result, triggering batch processing if needed."""
        while not event.is_set():
            with self._lock:
                if self._shutdown or self._paused or event.is_set():
                    return
                
                # Check if we should trigger processing for any queue
                if not self._batch_in_progress:
                    # Check tactic queue
                    if self._tactic_queue.should_process(self.timeout_seconds, self.max_batch_tokens):
                        self._process_queue_locked(
                            self._tactic_queue,
                            self.inner_model.sample_tactic_from_str_batch
                        )
                        continue
                    # Check value queue
                    if self._value_queue.should_process(self.timeout_seconds, self.max_batch_tokens):
                        self._process_queue_locked(
                            self._value_queue,
                            self.inner_model.predict_value_from_str_batch
                        )
                        continue

                self._batch_ready.wait(timeout=0.1)

    def _process_queue_locked(self, queue: RequestQueue, process_fn):
        """Process pending batch from the specified queue. Must be called with lock held."""
        if not queue.has_pending() or self._batch_in_progress:
            return

        self._batch_in_progress = True
        batch_num = queue._total_batches + 1  # Will be incremented by take_batch

        # Take all pending items
        batch = queue.take_batch()

        # Release lock during inference
        self._lock.release()
        try:
            state_strs = [item[0] for item in batch]
            results = process_fn(state_strs)

            # Distribute results
            for i, (_, event, slot) in enumerate(batch):
                slot.append(results[i])
                event.set()
                
        except Exception as e:
            log(f"Batch #{batch_num} ({queue.name}): FAILED - {e}", component="BlockingTacticModel")
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
    
    def sample_tactic(self, state) -> ValueOrError[list[str]]:
        """Sample tactics for a single state by calling the remote inference server."""
        assert len(state) == 1, \
            f"expected single branch in state, got {len(state)}"
        state_str = str(state[0].state).strip()
        return self.sample_tactic_from_str(state_str)

    def sample_tactic_from_str(self, state_str: str) -> ValueOrError[list[str]]:
        """Sample tactics for a single state string."""
        results = self.sample_tactic_from_str_batch([state_str])
        return results[0]

    def sample_tactic_from_str_batch(self, state_strs: list[str]) -> list[ValueOrError[list[str]]]:
        """Sample tactics for multiple state strings by calling the remote inference server."""
        if not state_strs:
            return []
        
        try:
            response = self._session.post(
                f"http://{self.server_address}/generate",
                json={"states": state_strs},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            # Reset failure counter on success
            with self._lock:
                self._consecutive_failures = 0
            
            results = []
            for item in data["results"]:
                if "error" in item:
                    results.append(ValueOrError.from_error(item["error"]))
                else:
                    results.append(ValueOrError.from_success(item["tactics"]))
            return results
        except http_requests.exceptions.Timeout:
            self._record_failure(f"Timeout calling inference server at {self.server_address} (timeout={self.timeout})")
            return [ValueOrError.from_error("Timeout") for _ in state_strs]
        except http_requests.exceptions.RequestException as e:
            self._record_failure(f"Error calling inference server at {self.server_address}: {e}")
            return [ValueOrError.from_error(str(e)) for _ in state_strs]
    
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

    def predict_value(self, state: State) -> ValueOrError[float]:
        """Predict value for a single state by calling the remote inference server."""
        assert len(state) == 1, \
            f"expected single branch in state, got {len(state)}"
        state_str = str(state[0].state).strip()
        return self.predict_value_from_str(state_str)

    def predict_value_from_str(self, state_str: str) -> ValueOrError[float]:
        """Predict value for a single state string."""
        results = self.predict_value_from_str_batch([state_str])
        return results[0]

    def predict_value_from_str_batch(self, state_strs: list[str]) -> list[ValueOrError[float]]:
        """Predict values for multiple state strings by calling the remote inference server."""
        if not state_strs:
            return []
        
        try:
            response = self._session.post(
                f"http://{self.server_address}/predict_value",
                json={"states": state_strs},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            with self._lock:
                self._consecutive_failures = 0
            
            results = []
            for item in data["results"]:
                if "error" in item:
                    results.append(ValueOrError.from_error(item["error"]))
                else:
                    results.append(ValueOrError.from_success(float(item["value"])))
            return results
        except http_requests.exceptions.Timeout:
            self._record_failure(f"Timeout calling inference server for value prediction")
            return [ValueOrError.from_error("Timeout") for _ in state_strs]
        except http_requests.exceptions.RequestException as e:
            self._record_failure(f"Error calling inference server for value prediction: {e}")
            return [ValueOrError.from_error(str(e)) for _ in state_strs]
    
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
            return jsonify({"results": []})
        # Submit all states at once - they'll be batched together
        results = model.sample_tactic_from_str_batch(states)
        return jsonify({
            "results": [
                {"tactics": r.value} if r.is_success() else {"error": r.error}
                for r in results
            ]
        })

    @app.route("/predict_value", methods=["POST"])
    def predict_value():
        data = request.get_json()
        states = data.get("states", [])
        if not states:
            return jsonify({"results": []})
        results = model.predict_value_from_str_batch(states)
        return jsonify({
            "results": [
                {"value": r.value} if r.is_success() else {"error": r.error}
                for r in results
            ]
        })
    
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

