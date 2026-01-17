# nanoproof

This is an attempt to replicate AlphaProof. It is based on [nanochat](https://github.com/karpathy/nanochat) and the
official AlphaProof pseudocode (together with many open-source datasets and tools). So far, we have:
- pretraining on Nemotron-CC-Math (~20B tokens)
- midtraining on Lean code from GitHub (~65M tokens)
- supervised fine-tuning on LeanTree (~260k transitions extracted from Mathlib) - https://github.com/kripner/leantree
- tokenizer based on GPT-2, adapted for Lean
- interaction with Lean using LeanTree server
- MCTS-based prover
- evaluation script for MiniF2F and Lean-Workbook
- fully distributed RL training
  - a GPU node for inference + DDP training + coordination
  - CPU nodes for actors (provers)
  - CPU nodes for the LeanTree servers

The best score achieved so far is **37.3% on MiniF2F**.

This project is in early stages and still a bit hard to work with. If you want to contribute, the best way to start is to write me an email!


# Setup

```
cd nanoproof
uv sync --extra cpu --group dev
source .venv/bin/activate

hf auth login

python -m nanoproof.dataset
python -m scripts.tok_train
```

Pretrain:

```
python -m nanoproof.pretrain
```

or

```
torchrun --standalone --nproc_per_node=2 -m nanoproof.pretrain
```

Similarly with `nanoproof.midtrain` and `nanoproof.sft`.

# Running the RL Loop

The RL loop alternates between collecting proof transitions using MCTS and training the model.

## Web Monitor

When the RL loop starts, it launches a web monitor on port 5050. Open `http://localhost:5050` in your browser to see:
- Training stats (loss, step, samples collected)
- Prover server status with thread-level indicators
- GPU utilization and memory
- Evaluation history
- Live log stream

To build the React frontend:

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

Before running RL, you need LeanTree server(s) running (provides proof verification). Example run:

```bash
leanserver --project-path /path/to/leantree_project/ \
    --repl-exe /path/to/leantree/lean-repl/.lake/build/bin/repl \
    --imports Mathlib \
    --max-processes 32 \
    --address=0.0.0.0 \
    --port=8000
leanserver --project-path /path/to/leantree_project/ \
    --repl-exe /path/to/leantree/lean-repl/.lake/build/bin/repl \
    --imports Mathlib FormalConjectures.ForMathlib.Analysis.SpecialFunctions.NthRoot FormalConjectures.Util.Answer \
    --max-processes 32 \
    --address=0.0.0.0 \
    --port=8000
    --warmup
```

## Local Mode (Single Node)

For single-node training, simply run:

```bash
# Single GPU
python -m nanoproof.rl

# Multi-GPU
torchrun --standalone --nproc_per_node=2 -m nanoproof.rl
```

## Distributed Mode (Multiple Nodes)

For scaling across multiple nodes, the system is split into:
- RL server (GPU nodes): handles inference, training, and coordination
- Prover servers (CPU nodes): running MCTS proof search

Prover servers automatically register with the RL server on startup and unregister on shutdown.

### Step 1: Start RL Training (on GPU node)

```bash
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
    --num-actors 32
```

The prover will automatically register itself with the RL server. You can start/stop prover servers at any time - collection will continue with available provers.

### Architecture Overview

```
GPU Node (torchrun with DDP)
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────┐  ┌─────────────┐       ┌─────────────┐     │
│  │ Rank 0      │  │ Rank 1      │  ...  │ Rank N      │     │
│  │ - Training  │  │ - Training  │       │ - Training  │     │
│  │ - Inference │  │ - Inference │       │ - Inference │     │
│  │   :5001     │  │   :5002     │       │   :500N+1   │     │
│  └──────┬──────┘  └──────┬──────┘       └──────┬──────┘     │
│         │                │                     │            │
│         └────────────────┼─────────────────────┘            │
│                          ▼                                  │
│              ┌───────────────────────┐                      │
│              │  Coordinator (:5000)  │ (master only)        │
│              │  - Registry           │                      │
│              │  - Dispatcher         │                      │
│              │  - Load balancer      │                      │
│              │  - Web monitor (:5050)│                      │
│              └───────────┬───────────┘                      │
└──────────────────────────┼──────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Prover Server 1 │ │ Prover Server 2 │ │ Prover Server N │
│ (CPU Node)      │ │ (CPU Node)      │ │ (CPU Node)      │
│ - MCTS actors   │ │ - MCTS actors   │ │ - MCTS actors   │
│ - :5001         │ │ - :5001         │ │ - :5001         │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
              ┌─────────────────────────┐
              │  Lean Server(s) (:8000) │
              │  - Proof verification   │
              └─────────────────────────┘
```

**Coordinator** (master process, port 5000):
- Maintains registry of prover servers (`/register`, `/unregister`)
- Dispatches theorems to provers (`/get_theorem`)
- Receives back proof results (`/submit_result`)
- Load-balances inference requests across GPUs
- Hosts web monitor at port 5050

**Inference servers** (one per GPU rank, ports 5001+):
- Batches tactic generation requests (until enough are collected or timeout runs out)

**Prover servers** (CPU nodes):
- Registers on startup, unregisters on shutdown
- Runs multiple MCTS actors in parallel
- Requests theorems from coordinator
- Submits results (proofs and stats) back to coordinator

**Training loop** (all GPU ranks via DDP):
1. Collection: provers search for proofs, submit transitions
2. Training: pause inference, gradient step, resume inference
3. Evaluation (once in a while): provers evaluate on MiniF2F/LeanWorkbook
4. Repeat


# Ideas

- try training on state_after as well, just to give the model more training signal (it was done in some paper, maybe GPT-f)
- let tokens attend bi-directionally inside the fixed-size state (a la PrefixLM)
- try proving the negation in each node (if critic deems it likely to succeed)

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


# Cite

If you find nanoproof helpful in your research cite simply as:

```
@misc{nanoproof,
  author = {Matěj Kripner},
  title = {nanoproof},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/kripner/nanoproof}
}
```