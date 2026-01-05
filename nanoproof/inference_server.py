"""
Inference Server - provides batched inference for prover servers.

This server runs on GPU nodes and handles tactic generation requests
from multiple prover servers. It batches requests for efficient GPU utilization.

Supports multi-GPU: requests are distributed across GPUs in round-robin fashion.

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
# Multi-GPU Round-Robin Handler
# -----------------------------------------------------------------------------

class MultiGPUInferenceHandler:
    """
    Handles inference across multiple GPUs with round-robin distribution.
    
    Each GPU has its own TacticModel and lock to allow parallel inference.
    Requests are distributed across GPUs in round-robin fashion.
    """
    
    def __init__(self, num_samples: int = 6, model_source: str = "sft", model_tag: str = "d26"):
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available")
        
        print(f"[Inference] Initializing models on {self.num_gpus} GPU(s)...")
        
        # Create a model for each GPU
        self.models: list[TacticModel] = []
        self.locks: list[threading.Lock] = []
        
        for gpu_id in range(self.num_gpus):
            device = torch.device(f"cuda:{gpu_id}")
            print(f"[Inference] Loading model on GPU {gpu_id} ({torch.cuda.get_device_name(gpu_id)})...")
            
            model, tokenizer, _ = load_model(model_source, device, phase="eval", model_tag=model_tag)
            engine = Engine(model, tokenizer)
            tactic_model = TacticModel(model, tokenizer, engine, num_samples=num_samples)
            
            self.models.append(tactic_model)
            self.locks.append(threading.Lock())
        
        # Round-robin counter
        self._next_gpu = 0
        self._counter_lock = threading.Lock()
        
        # Stats
        self._total_requests = 0
        self._requests_per_gpu = [0] * self.num_gpus
        self._stats_lock = threading.Lock()
        
        print(f"[Inference] All {self.num_gpus} GPU(s) ready!")
    
    def _get_next_gpu(self) -> int:
        """Get the next GPU in round-robin order."""
        with self._counter_lock:
            gpu_id = self._next_gpu
            self._next_gpu = (self._next_gpu + 1) % self.num_gpus
            return gpu_id
    
    def generate_tactics_batch(self, state_strs: list[str]) -> list[list[str]]:
        """
        Generate tactics for multiple states.
        Uses round-robin GPU selection for load balancing.
        """
        if len(state_strs) == 0:
            return []
        
        # Select GPU (round-robin)
        gpu_id = self._get_next_gpu()
        
        # Update stats
        with self._stats_lock:
            self._total_requests += len(state_strs)
            self._requests_per_gpu[gpu_id] += len(state_strs)
        
        # Acquire lock for this GPU and run inference
        with self.locks[gpu_id]:
            return self.models[gpu_id].sample_tactic_from_str_batch(state_strs)
    
    def get_stats(self) -> dict:
        """Return stats about the handler."""
        with self._stats_lock:
            return {
                "num_gpus": self.num_gpus,
                "total_requests": self._total_requests,
                "requests_per_gpu": self._requests_per_gpu.copy(),
            }


# -----------------------------------------------------------------------------
# Flask Server
# -----------------------------------------------------------------------------

def create_app(handler: MultiGPUInferenceHandler):
    app = Flask(__name__)
    
    # Disable Flask request logging to reduce spam
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
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
        
        # Use batch generation for efficiency (round-robin across GPUs)
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
    parser.add_argument("--num-samples", type=int, default=6, help="Tactics to sample per state")
    parser.add_argument("--model-source", default="sft", help="Model source (sft, pretrain, etc)")
    parser.add_argument("--model-tag", default="d26", help="Model tag")
    args = parser.parse_args()
    
    print(f"[Inference] GPUs available: {torch.cuda.device_count()}")
    
    handler = MultiGPUInferenceHandler(
        num_samples=args.num_samples,
        model_source=args.model_source,
        model_tag=args.model_tag,
    )
    
    app = create_app(handler)
    
    print(f"[Inference] Server starting on {args.host}:{args.port}")
    print(f"[Inference] Using {handler.num_gpus} GPU(s) in round-robin mode")
    print(f"[Inference] Samples per state: {args.num_samples}")
    
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()

