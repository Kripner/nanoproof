# nanoproof

This is an attempt to replicate AlphaProof. It is based on [nanochat](https://github.com/karpathy/nanochat) and the
official AlphaProof pseudocode (together with many open-source datasets and tools). So far, we have:
- pretraining on Nemotron-CC-Math
- midtraining on Lean code from GitHub
- supervised fine-tuning on LeanTree (transitions extracted from Mathlib)
- interaction with Lean using a LeanTree server
- MCTS-based prover
- a simple RL training loop
- evaluation script for success rate on MiniF2F and Lean-Workbook

The best score achieved so far is **32.8% on MiniF2F** (more precisely on the subset of its first 64 theorems).

This project is in early stages and still a bit hard to work with. If you want to contribute, the best way to start is to write me an email!


# Questions

- how is the action prob obtained from tokens probs?
- is value predicted for state (as per paper) or for state-action (as per pseudocode)?
- more importantly: do the bins correspond to Q 0-1 or V 1-inf?
  - Milan: it's V
- how do the value bins correspond to values? Ie. what are the values of bins 0 and 63?
- Supplemental Data Table 4: is bs=4096 in tokens or in samples?
  Probably in samples - if in tokens, otherwise we would only use ~10% of the data: `4096/64 samples-per-batch * 500 steps = 32k samples`, but 300k are available.
  (Also 4096 samples-per-batch makes sense for the Pre-Training, where it yields something on the order of 50 epochs that Julian reported)
- what is the value of a child that was not visited yet?
  - zero as per pseudocode (line 570) - that would be weird/wrong
  - parent minus "UCB unvisited children value penalty" (=32) as per paper
- beta and gamma are the same? (as per code)
- In the pseudocode, is_optimal is not set on the new children/grandchildren created when expanding a node, even if they are terminal.
- Replay buffer size seems to be 250k in the pseudocode but 60M in the paper (Supplemental Data Table 6)

# Ideas

- try training on state_after as well, just to give the model more training signal (it was done in some paper, maybe GPT-f)
- let tokens attend bi-directionally inside the fixed-size state (a la PrefixLM)
- try proving the negation in each node (if critic deems it likely to succeed)

# Setup

```
cd nanoproof
uv sync --extra cpu --group dev
source .venv/bin/activate

hf auth login

python -m nanoproof.dataset
python -m scripts.tok_train
python -m nanoproof.pretrain
```

or

```
torchrun --standalone --nproc_per_node=2 -m nanoproof.pretrain
```

# Running the RL Loop

The RL loop alternates between collecting proof transitions using MCTS and training the model.

## Web Monitor

When the RL loop starts, it launches a web monitor on port 5050. Open `http://localhost:5050` in your browser to see:
- Training stats (loss, step, samples collected)
- Prover server status with thread-level indicators
- GPU utilization and memory
- Evaluation history
- Live log stream

To build the React frontend (optional, a fallback HTML is included):

```bash
cd nanoproof/web
npm install
npm run build
```

To test the monitor without running actual training:

```bash
python tests/test_cli.py
```

## Prerequisites

Before running RL, you need a Lean server running (provides proof verification):

```bash
leanserver --project-path /path/to/leantree_project/ \
    --repl-exe /path/to/leantree/lean-repl/.lake/build/bin/repl \
    --imports Mathlib \
    --max-processes 32 \
    --address=0.0.0.0 \
    --port=8000
```

## Local Mode (Single Node)

For development or single-GPU training, run everything on one machine:

```bash
# Single GPU
python -m nanoproof.rl

# Multi-GPU with DDP
torchrun --standalone --nproc_per_node=2 -m nanoproof.rl
```

Configuration can be passed via command line:

```bash
python -m nanoproof.rl --run=my_experiment --num_actors=16 --collect_transitions=200
```

## Distributed Mode (Multiple Nodes)

For scaling across multiple nodes, the system is split into:
- RL server (GPU nodes): handles inference and training
- Prover servers (CPU nodes): run MCTS proof search

Prover servers automatically register with the RL server on startup and unregister on shutdown.

### Step 1: Start RL Training (on GPU node)

```bash
# Single GPU
python -m nanoproof.rl --distributed=True

# Multi-GPU with DDP
torchrun --standalone --nproc_per_node=2 -m nanoproof.rl --distributed=True
```

The RL server will wait for prover agents to register before starting collection.

### Step 2: Start Prover Servers (on CPU nodes)

On each CPU node, start a prover server pointing to the RL server:

```bash
python -m nanoproof.prover_server \
    --rl-server <GPU_NODE_IP>:5000 \
    --lean-server <LEAN_SERVER_IP>:8000 \
    --port 5001 \
    --num-actors 8
```

The prover will automatically register itself with the RL server. You can start/stop prover servers at any time - collection will continue with available provers.

### Architecture Overview

```
+-------------------+       +-------------------+
|   RL Server       |       |  Prover Server 1  |
|   (GPU Node)      |<----->|  (CPU Node)       |
|                   |       +-------------------+
| - Training (DDP)  |       +-------------------+
| - Inference API   |<----->|  Prover Server 2  |
| - Coordination    |       |  (CPU Node)       |
| - Registry        |       +-------------------+
+-------------------+       +-------------------+
        ^                   |  Prover Server N  |
        +------------------>|  (CPU Node)       |
                            +-------------------+
                                    ^
                                    |
                            +-------------------+
                            |   Lean Server     |
                            +-------------------+
```

The RL server:
- Exposes an inference endpoint at `/generate` (port 5000)
- Maintains a registry of prover servers (`/register`, `/unregister`)
- Instructs provers to start/pause collection
- Polls provers for found transitions
- Aggregates transitions into a global replay buffer
- Trains the model using DDP

Each prover server:
- Registers itself on startup, unregisters on shutdown
- Runs multiple MCTS actors in parallel
- Calls the RL server for tactic generation
- Calls the Lean server for proof verification
- Buffers found transitions until polled