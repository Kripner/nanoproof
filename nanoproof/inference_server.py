"""
Inference Server - provides batched inference for prover servers.

This server runs on GPU nodes and handles tactic generation requests
from multiple prover servers. It batches requests for efficient GPU utilization.

Usage:
    python -m nanoproof.inference_server --port 5000
"""

import argparse
import threading
import time
from dataclasses import dataclass, field

import torch
from flask import Flask, request, jsonify

from nanoproof.checkpoints import load_model
from nanoproof.engine import Engine
from nanoproof.search import TacticModel


# -----------------------------------------------------------------------------
# Batched Inference Handler
# -----------------------------------------------------------------------------

@dataclass
class BatchedInferenceHandler:
    """
    Handles batched inference for tactic generation.
    
    Requests are queued and processed in batches when either:
    - The batch is full (batch_size reached)
    - The timeout has elapsed since the first pending request
    """
    
    tactic_model: TacticModel
    batch_size: int = 32
    timeout_seconds: float = 0.1
    
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _batch_ready: threading.Condition = field(init=False)
    _pending: list = field(default_factory=list)
    _first_request_time: float = None
    _batch_in_progress: bool = False
    _total_requests: int = 0
    _total_batches: int = 0
    
    def __post_init__(self):
        self._batch_ready = threading.Condition(self._lock)
    
    def generate_tactics(self, state_str: str) -> list[str]:
        """
        Generate tactics for a single state.
        Thread-safe, blocks until result is available.
        """
        result_event = threading.Event()
        result_slot: list = []
        
        with self._lock:
            self._total_requests += 1
            self._pending.append((state_str, result_event, result_slot))
            
            if self._first_request_time is None:
                self._first_request_time = time.time()
            
            # Check if we should process immediately
            if len(self._pending) >= self.batch_size and not self._batch_in_progress:
                self._process_batch_locked()
        
        # Wait for batch or timeout
        if not result_event.is_set():
            self._wait_for_batch_or_timeout(result_event)
        
        # Wait for result with timeout
        result_event.wait(timeout=60.0)
        
        return result_slot[0] if result_slot else []
    
    def generate_tactics_batch(self, state_strs: list[str]) -> list[list[str]]:
        """
        Generate tactics for multiple states at once.
        More efficient than calling generate_tactics repeatedly.
        """
        # Process directly without queuing for batch requests
        with self._lock:
            return self.tactic_model.sample_tactic_from_str_batch(state_strs)
    
    def _wait_for_batch_or_timeout(self, my_event: threading.Event):
        """Wait until batch is ready or timeout."""
        while not my_event.is_set():
            with self._lock:
                if my_event.is_set():
                    return
                
                if self._first_request_time is None:
                    self._batch_ready.wait(timeout=0.1)
                    continue
                
                elapsed = time.time() - self._first_request_time
                remaining = self.timeout_seconds - elapsed
                
                if remaining <= 0 or len(self._pending) >= self.batch_size:
                    if not self._batch_in_progress and len(self._pending) > 0:
                        self._process_batch_locked()
                    return
                
                self._batch_ready.wait(timeout=remaining)
    
    def _process_batch_locked(self):
        """Process the current batch. Must be called with lock held."""
        if len(self._pending) == 0:
            return
        
        self._batch_in_progress = True
        self._total_batches += 1
        
        batch = self._pending[:]
        self._pending = []
        self._first_request_time = None
        
        batch_size = len(batch)
        print(f"[Inference] Processing batch #{self._total_batches}: {batch_size} requests")
        
        # Release lock during inference
        self._lock.release()
        try:
            state_strs = [item[0] for item in batch]
            results = self.tactic_model.sample_tactic_from_str_batch(state_strs)
        finally:
            self._lock.acquire()
        
        self._batch_in_progress = False
        
        # Distribute results
        for i, (_, result_event, result_slot) in enumerate(batch):
            result_slot.append(results[i])
            result_event.set()
        
        self._batch_ready.notify_all()
    
    def get_stats(self) -> dict:
        """Return stats about the handler."""
        with self._lock:
            return {
                "total_requests": self._total_requests,
                "total_batches": self._total_batches,
                "pending": len(self._pending),
            }


# -----------------------------------------------------------------------------
# Flask Server
# -----------------------------------------------------------------------------

def create_app(handler: BatchedInferenceHandler):
    app = Flask(__name__)
    
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})
    
    @app.route("/generate", methods=["POST"])
    def generate():
        """
        Generate tactics for one or more states.
        
        Request body:
            {"states": ["state1", "state2", ...]}
        
        Response:
            {"tactics": [["tactic1a", "tactic1b"], ["tactic2a"], ...]}
        """
        data = request.get_json()
        states = data.get("states", [])
        
        if len(states) == 0:
            return jsonify({"tactics": []})
        
        # Use batch generation for efficiency
        tactics = handler.generate_tactics_batch(states)
        return jsonify({"tactics": tactics})
    
    @app.route("/stats", methods=["GET"])
    def stats():
        return jsonify(handler.get_stats())
    
    return app


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inference Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--batch-size", type=int, default=32, help="Max batch size")
    parser.add_argument("--timeout", type=float, default=0.1, help="Batch timeout in seconds")
    parser.add_argument("--num-samples", type=int, default=6, help="Tactics to sample per state")
    parser.add_argument("--model-source", default="sft", help="Model source (sft, pretrain, etc)")
    parser.add_argument("--model-tag", default="d26", help="Model tag")
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_source}/{args.model_tag}...")
    tactic_model = TacticModel.create(num_samples=args.num_samples)
    
    handler = BatchedInferenceHandler(
        tactic_model=tactic_model,
        batch_size=args.batch_size,
        timeout_seconds=args.timeout,
    )
    
    app = create_app(handler)
    
    print(f"Inference server starting on {args.host}:{args.port}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Timeout: {args.timeout}s")
    print(f"  Samples per state: {args.num_samples}")
    
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()

