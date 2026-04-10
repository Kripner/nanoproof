#!/usr/bin/env python3
"""
Test script for the web-based CLI monitor.

This simulates a training run to test the web monitor without running the actual
training loop. Run this script and open the printed URL in a browser.

Usage:
    python tests/test_cli.py
"""

import random
import time
import threading
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanoproof.cli import create_monitor, log


def simulate_local_actors(monitor, num_actors: int = 16):
    """Simulate local actor threads."""
    while True:
        for i in range(num_actors):
            r = random.random()
            if r < 0.6:
                state = "running"
            elif r < 0.8:
                state = "idle"
            else:
                state = "error"
            monitor.update_local_actor(i, state=state)
        time.sleep(0.5 + random.random())


def simulate_gpu(monitor, gpu_id: int, name: str):
    """Simulate GPU metrics."""
    while True:
        monitor.update_gpu(
            gpu_id=gpu_id,
            name=name,
            utilization=random.uniform(60, 100),
            memory_used=random.randint(20000, 38000),
            memory_total=40960,
            inference_queue_size=random.randint(0, 32),
            avg_wait_time_ms=random.uniform(5, 50),
        )
        time.sleep(1.0)


def simulate_collection_phase(monitor, duration: float = 20.0):
    """Simulate the collection phase."""
    target = 200
    monitor.start_collection(target_samples=target, num_actors=32)
    log("Starting collection phase", component="Coordinator")
    
    start_time = time.time()
    samples = 0
    proofs_attempted = 0
    proofs_successful = 0
    
    while time.time() - start_time < duration and samples < target:
        time.sleep(0.3)
        
        # Simulate proof attempts
        attempts = random.randint(1, 5)
        for _ in range(attempts):
            proofs_attempted += 1
            monitor.record_proof_attempt(successful=False)
            
            if random.random() < 0.25:
                transitions = random.randint(1, 8)
                samples += transitions
                proofs_successful += 1
                monitor.record_proof_attempt(successful=True, transitions=transitions)
                log(f"Proof found! +{transitions} transitions", component="Collection")
        
        # Simulate expansions
        for _ in range(random.randint(10, 50)):
            monitor.record_expansion()
        
        # Simulate batch wait times
        for _ in range(random.randint(1, 10)):
            monitor.record_batch_wait(random.uniform(0.01, 0.2))
        
        log(f"Collection progress: {samples}/{target}", component="Collection")
    
    log(f"Collection complete: {samples} samples from {proofs_successful} proofs", component="Collection")


def simulate_training_phase(monitor, num_steps: int = 10):
    """Simulate the training phase."""
    monitor.set_phase("training")
    log("Starting training phase", component="Train")
    
    loss = 2.5
    for step in range(num_steps):
        time.sleep(0.5)
        
        loss = loss * 0.95 + random.uniform(-0.05, 0.1)
        tokens = random.randint(50000, 100000)
        
        monitor.update_training(
            step=monitor.step + step,
            loss=loss,
            num_tokens=tokens,
            lr=0.0001,
        )
        
        monitor.set_replay_buffer_size(monitor.replay_buffer_size + random.randint(0, 20))
        
        log(f"Step {monitor.step + step}: loss={loss:.6f}, tokens={tokens}", component="Train")
    
    log("Training phase complete", component="Train")


def simulate_eval_phase(monitor, step: int):
    """Simulate the evaluation phase."""
    monitor.set_phase("evaluating")
    log("Starting evaluation phase", component="Eval")
    
    time.sleep(2)
    
    # Simulate MiniF2F eval
    minif2f_solved = random.randint(15, 25)
    minif2f_total = 64
    monitor.record_eval(
        step=step,
        dataset="MiniF2F",
        success_rate=minif2f_solved / minif2f_total,
        solved=minif2f_solved,
        total=minif2f_total,
        errors=random.randint(0, 3),
    )
    log(f"MiniF2F: {minif2f_solved}/{minif2f_total} ({minif2f_solved/minif2f_total*100:.1f}%)", component="Eval")
    
    time.sleep(1)
    
    # Simulate LeanWorkbook eval
    lwb_solved = random.randint(10, 20)
    lwb_total = 64
    monitor.record_eval(
        step=step,
        dataset="LeanWorkbook",
        success_rate=lwb_solved / lwb_total,
        solved=lwb_solved,
        total=lwb_total,
        errors=random.randint(0, 2),
    )
    log(f"LeanWorkbook: {lwb_solved}/{lwb_total} ({lwb_solved/lwb_total*100:.1f}%)", component="Eval")
    
    log("Evaluation complete", component="Eval")


def main():
    print("=" * 60)
    print("  NanoProof CLI Monitor Test")
    print("=" * 60)
    print()
    print("This script simulates a training run to test the web monitor.")
    print("Open the URL printed below in your browser.")
    print()
    
    # Create monitor
    monitor = create_monitor(num_actors=32, enabled=True, port=5050)
    
    print("Press Ctrl+C to stop.\n")
    
    # Start background simulator for local actors
    actor_thread = threading.Thread(
        target=simulate_local_actors,
        args=(monitor, 16),
        daemon=True
    )
    actor_thread.start()
    log("Started local actor simulator", component="Test")
    
    # Start GPU simulator
    gpu_thread = threading.Thread(
        target=simulate_gpu,
        args=(monitor, 0, "NVIDIA A100 40GB"),
        daemon=True
    )
    gpu_thread.start()
    log("Started GPU simulator", component="Test")
    
    # Simulate training loop
    try:
        step = 0
        while True:
            monitor.set_step(step)
            
            # Collection phase
            simulate_collection_phase(monitor, duration=15.0)
            
            # Evaluation every 5 steps
            if step % 5 == 0:
                simulate_eval_phase(monitor, step)
            
            # Training phase
            simulate_training_phase(monitor, num_steps=5)
            
            step += 5
            
            log(f"Completed training iteration, step={step}", component="Main")
            
    except KeyboardInterrupt:
        print("\n\nStopping test...")
        print("Monitor test complete.")


if __name__ == "__main__":
    main()

