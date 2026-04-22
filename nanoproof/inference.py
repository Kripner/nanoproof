"""
Inference - tactic models and batched inference for proof search.

This module contains:
- TacticModel: Core tactic generation model wrapping the Transformer
- BlockingTacticModel: Thread-safe wrapper that batches LLM calls from multiple threads
- RemoteTacticModel: Client that calls a remote inference server (for multi-GPU load balancing)
- InferenceBalancer: Routes inference across N remote backends via round-robin
- start_inference_server: Starts a Flask inference server for a BlockingTacticModel
- setup_distributed_inference: Starts servers on all DDP ranks, builds InferenceBalancer on master

Usage:
    python -m nanoproof.inference --model-path sft/.../model_005000.pt
"""

import argparse
import gc
import logging
import threading
import uuid
import time
from collections import deque
from dataclasses import dataclass
from typing import Self

import requests as http_requests
import torch
from flask import Flask, request, jsonify

from nanoproof.checkpoints import load_model
from nanoproof.cli import log, log0, log_actionable_error

logger = logging.getLogger(__name__)
from nanoproof.common import ValueOrError, GLOBAL_CONFIG, get_dist_info
from nanoproof.engine import Engine
from nanoproof.tokenizer import HuggingFaceTokenizer
from nanoproof.model import Transformer


# Type alias for State - avoid circular import with search.py
State = list  # list[LeanProofBranch]


# -----------------------------------------------------------------------------
# Tactic Models
# -----------------------------------------------------------------------------

# Result type for combined tactic + value prediction
TacticAndValue = tuple[list[str], float]  # (tactics, value)


@dataclass
class TacticModel:
    """
    Core tactic model that generates tactics and predicts values.

    The main API is `sample_tactic` which returns both tactics AND value for a state,
    since these are always needed together during MCTS.
    """
    network: Transformer
    tokenizer: HuggingFaceTokenizer
    engine: Engine
    num_samples: int
    seed: int

    def __post_init__(self):
        self.rng = torch.Generator(device=self.network.get_device())
        self.rng.manual_seed(self.seed)

    def sample_tactic(self, state: State) -> ValueOrError[TacticAndValue]:
        """Sample tactics and predict value for a state. Returns (tactics, value)."""
        assert len(state) == 1, \
            f"expected single branch in state when generating tactic, got {len(state)} - choose one goal first"
        return self.sample_tactic_from_str(str(state[0].state).strip())

    def sample_tactic_from_str(self, state_str: str) -> ValueOrError[TacticAndValue]:
        """Sample tactics and predict value for a state string. Returns (tactics, value)."""
        return self.sample_tactic_from_str_batch([state_str])[0]

    def _max_prompt_len(self) -> int:
        """Maximum tokenized prompt length the model accepts.

        RoPE tables are allocated for sequence_len * 10; we cap prompts at * 9
        to leave headroom for up to ~64 generated tokens.
        """
        return self.network.config.sequence_len * 9

    def prepare_tactic_prompt(self, state_str: str) -> tuple[list[int], bool]:
        """Tokenize a state string for tactic generation.

        Returns (prompt_tokens, is_too_long). When too_long, prompt_tokens is a
        single-token dummy and the caller is expected to surface an error for
        that index. Single source of truth for the tactic-prompt format and the
        oversized-prompt substitution; reused by BlockingTacticModel for its
        KV-cache budget.
        """
        bos = self.tokenizer.get_bos_token_id()
        tokens = self.tokenizer(state_str + "\n<|tactic|>", prepend=bos)
        if len(tokens) > self._max_prompt_len():
            return [bos], True
        return tokens, False

    def sample_tactic_from_str_batch(self, state_strs: list[str]) -> list[ValueOrError[TacticAndValue]]:
        """
        Batched tactic generation and value prediction from state strings.

        Returns list of (tactics, value) tuples for each state.
        """
        device = self.network.get_device()
        assert device.type == "cuda"

        # Prepare tokenized prompts for tactic generation
        tactic_prompts = []
        too_long_indices = set()
        for idx, state_str in enumerate(state_strs):
            tokens, too_long = self.prepare_tactic_prompt(state_str)
            tactic_prompts.append(tokens)
            if too_long:
                too_long_indices.add(idx)

        # Generate tactics
        seed = torch.randint(torch.iinfo(torch.int32).max, (1,), device=device, generator=self.rng).item()
        sample_toks_batch, masks_batch = self.engine.generate_batch(
            tactic_prompts, num_samples=self.num_samples, min_tokens=1, max_tokens=64, seed=seed
        )

        # Decode tactics
        tactics_results = []
        for prompt_idx in range(len(state_strs)):
            if prompt_idx in too_long_indices:
                tactics_results.append([])
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
                if tactic.strip() == "bound":
                    # `bound` tactic messes with the kernel check
                    continue
                tactics.append(tactic)
            tactics_results.append(tactics)

        # Free tactic generation intermediates before value prediction
        del sample_toks_batch, masks_batch
        gc.collect()
        torch.cuda.empty_cache()

        # Prepare prompts for value prediction
        value_delim_tok = self.tokenizer.encode_special("<|value|>")
        bin_token_ids = self.tokenizer.get_value_token_ids()

        value_prompts = []
        for idx, state_str in enumerate(state_strs):
            if idx in too_long_indices:
                value_prompts.append([self.tokenizer.get_bos_token_id(), value_delim_tok])
            else:
                tokens = self.tokenizer(state_str + "\n<|value|>", prepend=self.tokenizer.get_bos_token_id())
                value_prompts.append(tokens)

        # Predict values
        _, _, value_logits = self.engine.generate_batch(
            value_prompts, num_samples=1, min_tokens=1, max_tokens=1, return_logits=True
        )

        # Get logits at the generated position
        value_logits = torch.stack([value_logits[i][0][-1] for i in range(len(value_prompts))])  # (B, V)

        # Extract bin logits and compute soft predictions
        bin_logits = value_logits[:, bin_token_ids].float()  # (B, 64)
        bin_probs = torch.softmax(bin_logits, dim=-1)  # (B, 64)
        bin_values = torch.arange(1, GLOBAL_CONFIG.num_value_bins + 1, dtype=bin_probs.dtype, device=device)
        values = (bin_probs * bin_values).sum(dim=-1)  # (B,)
        values_list = values.tolist()

        # Combine results
        results = []
        for idx in range(len(state_strs)):
            if idx in too_long_indices:
                results.append(ValueOrError.from_error("State too long for model's rotary cache"))
            else:
                results.append(ValueOrError.from_success((tactics_results[idx], values_list[idx])))
        return results

    def shutdown(self):
        """No-op for non-batched model. Exists for API compatibility with BlockingTacticModel."""
        pass

    def pause(self):
        """No-op, exists for API compatibility with BlockingTacticModel."""
        pass

    def resume(self):
        """No-op, exists for API compatibility with BlockingTacticModel."""
        pass

    @classmethod
    def create(cls, num_samples: int, model_path: str, seed: int = 0) -> Self:
        model, tokenizer, _ = load_model(model_path, torch.device("cuda"), phase="eval")
        engine = Engine(model, tokenizer)
        return cls(model, tokenizer, engine, num_samples=num_samples, seed=seed)


def compute_max_batch_prompt_tokens(model_config, num_samples: int, device: torch.device) -> int:
    """Compute max_batch_prompt_tokens from available VRAM.

    The main memory consumer during batched inference is the KV cache, allocated as:
        2 (k+v) * n_layers * total_rows * kv_seq_len * n_kv_head * head_dim * dtype_bytes

    where total_rows = N_states * num_samples and kv_seq_len ~ max_prompt_len + max_gen_tokens.
    We approximate total memory as proportional to sum_of_prompt_tokens * num_samples,
    and set the limit so the KV cache fits in available VRAM with headroom for activations.
    """
    # mem_get_info returns (free, total) at the CUDA driver level, which
    # accounts for memory consumed by ALL processes on this GPU (e.g. NCCL
    # buffers from DDP peers).  get_device_properties().total_memory is the
    # raw physical VRAM and would over-estimate available space.
    free_driver, _ = torch.cuda.mem_get_info(device)
    # free_driver excludes PyTorch's caching-allocator reserved-but-unused
    # blocks, which are available for new allocations.  Add them back.
    reserved_unused = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    # Use 70% of remaining VRAM for the KV cache.  The remaining 30% covers
    # per-prompt prefill KV caches, forward-pass activations, allocator
    # fragmentation, and NCCL communication buffers that may appear after
    # this measurement (lazily allocated on the first large all_reduce).
    available = (free_driver + reserved_unused) * 0.70
    dtype_bytes = 2  # bf16/fp16
    head_dim = model_config.n_embd // model_config.n_head
    kv_bytes_per_token = 2 * model_config.n_layer * model_config.n_kv_head * head_dim * dtype_bytes
    max_tokens = int(available / (num_samples * kv_bytes_per_token))
    return max(1, max_tokens)


class BlockingTacticModel:
    """
    Thread-safe wrapper around TacticModel that batches LLM calls from multiple threads.

    Batches are processed when either:
    1. pending items exceed max_gen_samples, or
    2. timeout_seconds has elapsed since the first pending request

    Each request generates both tactics and value prediction together.
    """

    def __init__(self, inner_model: TacticModel, timeout_seconds: float, max_gen_samples: int | None,
                 max_batch_prompt_tokens: int | None = None):
        self.inner_model = inner_model
        self.timeout_seconds = timeout_seconds
        self.max_gen_samples = max_gen_samples
        self.max_batch_prompt_tokens = max_batch_prompt_tokens

        # Synchronization primitives
        self._lock = threading.Lock()
        self._batch_ready = threading.Condition(self._lock)
        self._batch_in_progress = False
        self._shutdown = False
        self._paused = False

        # Single queue for combined tactic+value requests.
        # Each entry is (state_str, token_count, event, result_slot), where
        # token_count is the tokenized length of the tactic-gen prompt, used
        # for the KV-cache budget (see _process_batch_locked).
        self._pending: list[tuple[str, int, threading.Event, list]] = []
        self._first_request_time: float | None = None
        self._total_batches = 0

        # LLM profiler instrumentation. Each BlockingTacticModel tracks its
        # own timeline of inference intervals plus a periodic queue-depth
        # sample. Rank 0's WebMonitor polls every rank's /llm_timeline
        # endpoint to aggregate these for the frontend LLM profiler tab.
        self._llm_events: deque[dict] = deque(maxlen=20000)
        self._llm_samples: deque[dict] = deque(maxlen=100000)
        self._llm_seq = 0
        self._llm_stop = threading.Event()
        self._llm_sampler_thread = threading.Thread(target=self._llm_sampler_loop, daemon=True)
        self._llm_sampler_thread.start()

    @property
    def network(self):
        return self.inner_model.network

    def _count_prompt_tokens(self, state_str: str) -> int:
        """Token count for the tactic-gen prompt (the dominant KV-cache driver).

        Delegates to TacticModel.prepare_tactic_prompt so the count matches the
        actual prompt that will be fed to Engine.generate, including the
        single-token dummy substitution for oversized prompts.
        """
        tokens, _ = self.inner_model.prepare_tactic_prompt(state_str)
        return len(tokens)

    def _pending_gen_samples(self) -> int:
        """Number of generation samples if pending items were batched now."""
        return len(self._pending) * self.inner_model.num_samples

    def _should_process(self) -> bool:
        """Check if batch should be processed based on timeout or sample count."""
        if not self._pending or self._first_request_time is None or self._paused:
            return False
        elapsed = time.time() - self._first_request_time
        if self.max_gen_samples is None:
            return elapsed >= self.timeout_seconds
        return elapsed >= self.timeout_seconds or self._pending_gen_samples() >= self.max_gen_samples

    def shutdown(self):
        """Signal shutdown to unblock all waiting threads."""
        self._llm_stop.set()
        with self._lock:
            self._shutdown = True
            for _, _, event, slot in self._pending:
                slot.append(ValueOrError.from_error("Model shutdown"))
                event.set()
            self._pending = []
            self._first_request_time = None
            self._batch_ready.notify_all()

    def pause(self):
        """Pause inference and free the CUDA allocator.

        Blocks new batches from starting (``_should_process`` returns False
        while paused) and waits for any in-progress batch to finish, then
        runs GC + empty_cache so the first subsequent training forward
        starts with a clean allocator state.

        Pending requests are kept on the queue: ``sample_tactic`` callers
        that straddle the pause just sit in ``_wait_for_result`` until
        resume(). This is what lets prover actors preserve their mid-MCTS
        state across a training step.
        """
        with self._lock:
            self._paused = True
            while self._batch_in_progress:
                self._batch_ready.wait(timeout=0.1)
        gc.collect()
        torch.cuda.empty_cache()

    def resume(self):
        """Resume inference after pause(). Wakes callers waiting in
        ``_wait_for_result``; their ``_should_process`` re-check will
        trigger the first post-resume batch if pending is non-empty."""
        with self._lock:
            self._paused = False
            self._batch_ready.notify_all()

    def sample_tactic(self, state: State) -> ValueOrError[TacticAndValue]:
        """Thread-safe sample_tactic that batches calls from multiple threads.

        Returns (tactics, value) tuple on success.
        """
        assert len(state) == 1
        return self.sample_tactic_from_str(str(state[0].state).strip())

    def sample_tactic_from_str(self, state_str: str) -> ValueOrError[TacticAndValue]:
        """Sample tactics and predict value for a single state string."""
        results = self.sample_tactic_from_str_batch([state_str])
        return results[0]

    def sample_tactic_from_str_batch(self, state_strs: list[str]) -> list[ValueOrError[TacticAndValue]]:
        """
        Thread-safe batch tactic+value generation. All states are queued together
        and will be processed in the same or consecutive GPU batches.

        Returns list of (tactics, value) tuples.
        """
        if not state_strs:
            return []

        # Tokenize upfront to get accurate prompt lengths for the KV-cache
        # budget. Done outside the lock since tokenization is pure-CPU work.
        # We use the same suffix as the tactic-generation path in TacticModel,
        # which is the dominant allocation (num_samples times bigger than value).
        token_counts = [self._count_prompt_tokens(s) for s in state_strs]

        with self._lock:
            if self._shutdown:
                return [ValueOrError.from_error("Model shutdown") for _ in state_strs]

            # Create events and slots for all states
            entries = [(s, tc, threading.Event(), []) for s, tc in zip(state_strs, token_counts)]

            # Add all to pending queue
            if self._first_request_time is None and entries:
                self._first_request_time = time.time()
            self._pending.extend(entries)

            # Trigger batch if we have enough tokens queued
            if not self._batch_in_progress and self._should_process():
                self._process_batch_locked()

        # Wait for all results (returns early on shutdown; pause blocks
        # until resume, so a paused rank does not short-circuit the caller).
        for _, _, event, _ in entries:
            self._wait_for_result(event)

        return [slot[0] if slot else ValueOrError.from_error("No result") for _, _, _, slot in entries]

    def _wait_for_result(self, event: threading.Event):
        """Wait for a result, triggering batch processing if needed.

        Pause does NOT break the wait: ``_should_process`` returns False
        while paused, so the caller simply sits on ``_batch_ready`` until
        resume() notifies and the first post-resume batch processes them.
        """
        while not event.is_set():
            with self._lock:
                if self._shutdown or event.is_set():
                    return

                # Check if we should trigger processing
                if not self._batch_in_progress and self._should_process():
                    self._process_batch_locked()
                    continue

                self._batch_ready.wait(timeout=0.1)

    def _llm_sampler_loop(self):
        """Background thread: sample pending-queue depth every 200ms.

        Sampling cadence is a compromise: fast enough to resolve sub-second
        queue buildup around pause/resume, slow enough that a multi-hour
        run doesn't overflow the bounded sample deque.
        """
        while not self._llm_stop.wait(timeout=0.2):
            with self._lock:
                self._llm_seq += 1
                self._llm_samples.append({
                    "t": time.time(),
                    "n": len(self._pending),
                    "seq": self._llm_seq,
                })

    def get_llm_timeline(self, since: float = float("-inf")) -> dict:
        """Return inference intervals + queue-depth samples with seq > since.

        Thread-safe. Polled by the master rank's WebMonitor to aggregate
        per-rank data for the LLM profiler tab.
        """
        events: list[dict] = []
        samples: list[dict] = []
        max_cursor = since if since != float("-inf") else 0.0
        with self._lock:
            for ev in self._llm_events:
                if ev["seq"] <= since:
                    continue
                events.append({"start": ev["start"], "end": ev["end"]})
                if ev["seq"] > max_cursor:
                    max_cursor = ev["seq"]
            for s in self._llm_samples:
                if s["seq"] <= since:
                    continue
                samples.append({"t": s["t"], "n": s["n"]})
                if s["seq"] > max_cursor:
                    max_cursor = s["seq"]
        return {"events": events, "samples": samples, "cursor": max_cursor}

    def _process_batch_locked(self):
        """Process pending batch. Must be called with lock held."""
        if not self._pending or self._batch_in_progress:
            return

        self._batch_in_progress = True
        self._total_batches += 1
        batch_num = self._total_batches
        inference_start_time = time.time()

        # Take items up to max_gen_samples limit to avoid OOM
        if self.max_gen_samples is not None:
            max_items = self.max_gen_samples // self.inner_model.num_samples
        else:
            max_items = len(self._pending)
        batch = self._pending[:max(1, max_items)]

        # Further limit by prompt tokens: the KV cache is allocated as
        # N_rows * max_prompt_len, so one long prompt inflates memory for ALL
        # rows. We track count * running_max to match the actual allocation.
        # Token counts are pre-computed by the tokenizer on queue entry.
        if self.max_batch_prompt_tokens is not None and len(batch) > 1:
            GEN_OVERHEAD = 64
            max_tokens = 0
            cut = len(batch)
            # Recompute the token limit from actual free VRAM rather than
            # relying on the init-time budget.  After training, persistent
            # allocations (NCCL buffers, weight casts, etc.) can reduce
            # available VRAM well below what was measured at init.
            device = self.inner_model.network.get_device()
            live_limit = compute_max_batch_prompt_tokens(
                self.inner_model.network.config, self.inner_model.num_samples, device)
            token_limit = min(self.max_batch_prompt_tokens, live_limit)
            for i, (_, tc, _, _) in enumerate(batch):
                max_tokens = max(max_tokens, tc + GEN_OVERHEAD)
                effective = (i + 1) * max_tokens
                if effective > token_limit:
                    cut = max(1, i)
                    break
            batch = batch[:cut]

        remaining = self._pending[len(batch):]

        self._pending = remaining
        self._first_request_time = time.time() if remaining else None

        # Release lock during inference
        self._lock.release()
        try:
            # Force cleanup of any lingering KV cache tensors from previous batches.
            # Python's GC may not have collected generator frames holding large GPU
            # allocations, so we explicitly collect before allocating new KV caches.
            gc.collect()
            torch.cuda.empty_cache()

            state_strs = [item[0] for item in batch]
            gen_samples = len(batch) * self.inner_model.num_samples
            max_tc = max((item[1] for item in batch), default=0)
            sum_tc = sum(item[1] for item in batch)
            allocated_gb = torch.cuda.memory_allocated() / 1024**3
            logger.debug(f"Batch #{batch_num}: {len(batch)} states, {gen_samples} gen samples, max_tokens={max_tc}, sum_tokens={sum_tc}, {allocated_gb:.1f} GiB allocated")
            t0 = time.time()
            results = self.inner_model.sample_tactic_from_str_batch(state_strs)
            logger.debug(f"Batch #{batch_num}: completed in {time.time() - t0:.3f}s")

            # Distribute results
            for i, (_, _, event, slot) in enumerate(batch):
                slot.append(results[i])
                event.set()

        except Exception as e:
            log(f"Batch #{batch_num}: FAILED - {e}", component="BlockingTacticModel")
            log_actionable_error("BlockingTacticModel", str(e),
                                batch=batch_num, states=len(batch), max_tokens=max_tc)
            # Note: OOM snapshots are dumped deeper in Engine.generate so the
            # snapshot captures the state *before* the finally-block cleanup.
            for _, _, event, slot in batch:
                slot.append(ValueOrError.from_error(str(e)))
                event.set()
        finally:
            self._lock.acquire()
            self._batch_in_progress = False
            self._llm_seq += 1
            self._llm_events.append({
                "start": inference_start_time,
                "end": time.time(),
                "seq": self._llm_seq,
            })
            self._batch_ready.notify_all()


# -----------------------------------------------------------------------------
# Remote Tactic Model (for multi-GPU inference via localhost HTTP)
# -----------------------------------------------------------------------------

class RemoteTacticModel:
    """
    Tactic model that calls a remote inference server.

    Used by InferenceBalancer to route requests to non-local GPU ranks.
    Each rank runs a BlockingTacticModel behind a Flask server; this client
    talks to it over localhost HTTP.
    """

    def __init__(self, server_address: str, timeout: float = 1800.0, max_pool_size: int = 512):
        # Timeout is deliberately large: during training, BlockingTacticModel
        # on each rank is paused and holds pending sample_tactic requests
        # without responding. The HTTP connection must outlast the whole
        # training step. 1800s (30min) is longer than any sane training
        # step; a true server hang will still surface, just slower.
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
        self._max_failures = 10
        self._last_error_log_time = 0
        self._error_log_interval = 5.0
        self._lock = threading.Lock()

    def sample_tactic(self, state) -> ValueOrError[TacticAndValue]:
        """Sample tactics and predict value for a single state."""
        assert len(state) == 1, \
            f"expected single branch in state, got {len(state)}"
        state_str = str(state[0].state).strip()
        return self.sample_tactic_from_str(state_str)

    def sample_tactic_from_str(self, state_str: str) -> ValueOrError[TacticAndValue]:
        """Sample tactics and predict value for a single state string."""
        results = self.sample_tactic_from_str_batch([state_str])
        return results[0]

    def sample_tactic_from_str_batch(self, state_strs: list[str]) -> list[ValueOrError[TacticAndValue]]:
        """Sample tactics and predict values for multiple state strings."""
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
                    tactics = item["tactics"]
                    value = float(item["value"])
                    results.append(ValueOrError.from_success((tactics, value)))
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

            now = time.time()
            if now - self._last_error_log_time >= self._error_log_interval:
                print(f"[RemoteTacticModel] {message} (failures: {failures}/{self._max_failures})")
                self._last_error_log_time = now

            if failures >= self._max_failures:
                raise ConnectionError(
                    f"Inference server appears disconnected after {failures} consecutive failures"
                )

    def shutdown(self):
        """No-op, exists for API compatibility."""
        pass

    def pause(self):
        """No-op, exists for API compatibility with BlockingTacticModel."""
        pass

    def resume(self):
        """No-op, exists for API compatibility with BlockingTacticModel."""
        pass


# -----------------------------------------------------------------------------
# Inference Balancer (load-balances across multiple GPUs)
# -----------------------------------------------------------------------------

class InferenceBalancer:
    """Load-balances inference across multiple GPU backends via HTTP.

    All backends are remote inference servers accessed via RemoteTacticModel.
    Requests are routed round-robin. Lifecycle management (pause/resume/shutdown)
    is handled by each rank's local BlockingTacticModel directly, not by the balancer.
    """

    def __init__(self, endpoints: list[str]):
        """
        Args:
            endpoints: List of "host:port" strings for inference servers (one per GPU rank).
        """
        self._backends = [RemoteTacticModel(ep) for ep in endpoints]
        self._next_idx = 0
        self._lock = threading.Lock()

    def _next_backend(self):
        with self._lock:
            idx = self._next_idx
            self._next_idx = (self._next_idx + 1) % len(self._backends)
        return self._backends[idx]

    def sample_tactic(self, state: State) -> ValueOrError[TacticAndValue]:
        return self._next_backend().sample_tactic(state)

    def sample_tactic_from_str(self, state_str: str) -> ValueOrError[TacticAndValue]:
        return self._next_backend().sample_tactic_from_str(state_str)

    def sample_tactic_from_str_batch(self, state_strs: list[str]) -> list[ValueOrError[TacticAndValue]]:
        return self._next_backend().sample_tactic_from_str_batch(state_strs)


# -----------------------------------------------------------------------------
# Flask Server (one per GPU rank for multi-GPU inference)
# -----------------------------------------------------------------------------

def create_blocking_model_app(model: BlockingTacticModel, server_id: str = ""):
    """Create Flask app for a single BlockingTacticModel (one per GPU rank)."""
    app = Flask(__name__)

    # Disable Flask request logging to reduce spam
    log_flask = logging.getLogger('werkzeug')
    log_flask.setLevel(logging.ERROR)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "server_id": server_id})

    @app.route("/generate", methods=["POST"])
    def generate():
        """Generate tactics and predict value for each state.

        Returns combined results with both 'tactics' and 'value' for each state.
        """
        data = request.get_json()
        states = data.get("states", [])
        if not states:
            return jsonify({"results": []})
        # Submit all states at once - they'll be batched together
        results = model.sample_tactic_from_str_batch(states)
        return jsonify({
            "results": [
                {"tactics": r.value[0], "value": r.value[1]} if r.is_success() else {"error": r.error}
                for r in results
            ]
        })

    @app.route("/llm_timeline", methods=["GET"])
    def llm_timeline():
        """Inference intervals + queue-depth samples for the LLM profiler.

        ``since`` is this server's monotonic seq; only items with seq > since
        are returned. Rank 0's WebMonitor polls this per rank and aggregates.
        """
        try:
            since = float(request.args.get("since", "-inf"))
        except ValueError:
            since = float("-inf")
        return jsonify(model.get_llm_timeline(since))

    return app


def start_inference_server(model: BlockingTacticModel, port: int, host: str = "0.0.0.0"):
    """
    Start inference server for a BlockingTacticModel in a background thread.

    Used by all DDP ranks to expose their GPU for inference via HTTP.
    Returns the background thread.
    """
    server_id = uuid.uuid4().hex
    app = create_blocking_model_app(model, server_id=server_id)

    def run_server():
        log_flask = logging.getLogger('werkzeug')
        log_flask.setLevel(logging.ERROR)
        app.run(host=host, port=port, threaded=True)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # Verify *our* server started, not a stale process on the same port.
    for _ in range(50):
        time.sleep(0.1)
        try:
            resp = http_requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
            if resp.ok and resp.json().get("server_id") == server_id:
                break
        except http_requests.ConnectionError:
            continue
    else:
        raise RuntimeError(f"Inference server failed to start on port {port}. "
                           f"Port may be in use - kill the old process or use a different --inference-server-port.")

    log0(f"Inference server started on port {port}", component="InferenceServer")
    return thread


# -----------------------------------------------------------------------------
# Distributed inference setup
# -----------------------------------------------------------------------------

def setup_distributed_inference(
    tactic_model: BlockingTacticModel,
    inference_server_port: int,
) -> InferenceBalancer | None:
    """Set up distributed inference across DDP ranks.

    Every rank starts a Flask inference server for its BlockingTacticModel.
    Master builds an InferenceBalancer that load-balances across all GPUs
    via HTTP. Lifecycle (pause/resume/shutdown) is managed by each rank's
    local BlockingTacticModel directly, not by the balancer.

    Returns the balancer on master, None on workers.
    Must be called by ALL DDP ranks.
    """
    ddp, rank, _, world_size = get_dist_info()

    # Every rank (including rank 0) starts a Flask inference server
    port = inference_server_port + rank
    start_inference_server(tactic_model, port)

    if rank != 0:
        return None

    all_endpoints = [
        f"127.0.0.1:{inference_server_port + r}"
        for r in range(world_size)
    ]
    return InferenceBalancer(all_endpoints)


# -----------------------------------------------------------------------------
# Interactive testing (python -m nanoproof.inference)
# -----------------------------------------------------------------------------

def _main():
    """
    Interactive tactic model: loads a model and lets you type tactic states
    to see generated tactics and value predictions.
    """
    parser = argparse.ArgumentParser(description="Interactive tactic model", allow_abbrev=False)
    parser.add_argument("--model-path", required=True, help="path to model_NNNNNN.pt (relative to models/ or absolute)")
    parser.add_argument("--num-samples", type=int, default=6, help="Tactics to sample per state")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    tactic_model = TacticModel.create(num_samples=args.num_samples, model_path=args.model_path)
    print(f"Model loaded. Device: {tactic_model.network.get_device()}")
    print()

    def get_input() -> str:
        lines = []
        print("Type a tactic state, followed by an empty line:")
        line = input()
        while line.strip() or not lines:
            lines.append(line.rstrip())
            line = input()
        return "\n".join(lines)

    inp = get_input()
    while inp.strip() not in ["q", "quit", "exit"]:
        print("Generating tactics...")
        result = tactic_model.sample_tactic_from_str(inp.strip())
        if result.is_success():
            tactics, value = result.value
            for i, tactic in enumerate(tactics):
                print(f"  [{i+1}] {tactic}")
            print(f"  Value: {value:.2f}")
        else:
            print(f"  Error: {result.error}")
        print()
        inp = get_input()

    print("Done.")


if __name__ == "__main__":
    _main()
