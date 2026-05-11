# nanoproof

![minif2f plots](dev/combined_plots.png)

Open and efficient automated theorem prover. Built on top of
[nanochat](https://github.com/karpathy/nanochat) and the official AlphaProof
pseudocode, with several open-source datasets and tools wired in. The pipeline
covers:

- pretraining on Nemotron-CC-Math (~20B tokens)
- midtraining on Lean code from GitHub (~65M tokens)
- supervised fine-tuning on LeanTree (~260k transitions extracted from Mathlib, see https://github.com/kripner/leantree)
- a GPT-2 BPE tokenizer with extra Lean / math special tokens
- interaction with Lean via the LeanTree server
- an MCTS-based prover with a learned policy and value head
- multi-GPU RL training with DDP and a pool of actor threads driving a fleet of
  remote Lean servers

Best score so far: **52.7% on MiniF2F-Valid (512 simulations)**.


# Setup

```
cd nanoproof
uv sync --extra cpu --group dev
source .venv/bin/activate
```

For GPU hosts swap `--extra cpu` for `--extra gpu`.

Runs (logs, checkpoints, eval results) land under `$NANOPROOF_HOME` (default
`~/.nanoproof/`). Set the env var if you want them somewhere else.


## Datasets

nanoproof uses several datasets across pretraining, midtraining, SFT, RL, and
evaluation. They all live under [nanoproof/data/](nanoproof/data/):

| Name | Stage | Source |
| --- | --- | --- |
| `nemotron` | pretrain | Nemotron-CC-Math-v1 (~20B tokens) |
| `leangithubraw` | midtrain | Lean code from GitHub (~65M tokens) |
| `leantree` | SFT | LeanTree transitions from Mathlib (~260k) |
| `leanworkbook` | RL | Lean-Workbook formal statements |
| `numinamath` | RL | NuminaMath-LEAN formal statements |
| `deepseek_prover` | RL | DeepSeek-Prover-V1 formal statements |
| `minif2f` | benchmark | MiniF2F (valid + test) |
| `proofnet` | benchmark | ProofNet (valid + test) |

`nemotron` is a gated HuggingFace dataset: accept the terms at [huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) and run `hf auth login` before downloading. The other datasets need no authentication.

Download them all with a single command:

```
python -m nanoproof.data.download
```

Or pick a subset (individual datasets or stage aliases `pretrain`, `midtrain`,
`sft`, `rl`, `bench`):

```
python -m nanoproof.data.download minif2f proofnet leantree
python -m nanoproof.data.download sft rl
```

`leangithubraw` is not published to HuggingFace; build it locally from the
source repos listed in [nanoproof/data/midtrain/leangithub_urls.txt](nanoproof/data/midtrain/leangithub_urls.txt):

```
python -m nanoproof.data.midtrain.leangithubraw build
```

The RL datasets (`leanworkbook`, `numinamath`, `deepseek_prover`) also pull a
pre-computed whitelist from [data/whitelists/](data/whitelists/) alongside the
source file. `list_theorems(split, lean_version=...)` uses it to skip theorems
that don't initialize under the given Lean toolchain. Whitelists currently ship
for Lean `v4.27.0`; regenerate for other versions with each module's
`check-init` CLI action (which needs a running Lean server).


## Tokenizer

Build the tokenizer (GPT-2 BPE plus extra Lean / math special tokens):

```
python -m scripts.tok_build
```


# Training

The three pre-RL stages share the same launch shape. Run on a single GPU:

```
python -m nanoproof.pretrain
```

Or with DDP across N GPUs:

```
torchrun --standalone --nproc_per_node=N -m nanoproof.pretrain
```

The same applies to `nanoproof.midtrain` and `nanoproof.sft`. Each stage writes
its outputs under `$NANOPROOF_HOME/{stage}/` with a timestamped run directory
that holds logs, args, and checkpoints.


# RL Loop

The RL loop alternates between collecting MCTS rollouts and training the
policy / value model. It runs as a single multi-GPU process (typically launched
via `torchrun`) that talks to one or more remote LeanTree servers. There is no
separate prover worker process; actor threads live inside the RL process.

## Prerequisite: Lean project

The LeanTree server needs a Lean project with dependencies built. For MiniF2F
evaluation you need both `mathlib` and `formal_conjectures` (which contains the
MiniF2F formalizations).

Create the project with leantree:

```python
from leantree import LeanProject
LeanProject.create("my_project", lean_version="v4.27.0", libraries=["mathlib"])
```

Add `formal_conjectures` to `my_project/lakefile.toml`:

```toml
[[require]]
name = "formal_conjectures"
scope = "google-deepmind"
git = "https://github.com/google-deepmind/formal-conjectures"
rev = "89c6801f9f05cf63105d66843ed70b1e4ceb0c69"
```

Then add an import in the root module (e.g. `my_project/MyProject.lean`) so
that `lake build` actually builds the dependency:

```lean
import FormalConjecturesForMathlib.Analysis.SpecialFunctions.NthRoot
import FormalConjectures.Util.Answer
```

Finally, run `lake update && lake build` in the project directory. `lake
update` fetches dependencies and downloads the prebuilt `.olean` cache.

The `formal_conjectures` revision must be compatible with your Lean version.
The rev above is known to work with Lean v4.27.0.

## Prerequisite: LeanTree server(s)

For Mathlib-only setups (e.g. SFT data extraction):

```bash
leanserver --project-path /path/to/leantree_project/ \
    --repl-exe /path/to/leantree/lean-repl/.lake/build/bin/repl \
    --imports Mathlib \
    --max-processes 32 \
    --address=0.0.0.0 \
    --port=8000
```

For MiniF2F evaluation (also needs `formal_conjectures`):

```bash
leanserver --project-path /path/to/leantree_project/ \
    --repl-exe /path/to/leantree/lean-repl/.lake/build/bin/repl \
    --imports Mathlib FormalConjecturesForMathlib.Analysis.SpecialFunctions.NthRoot FormalConjectures.Util.Answer \
    --max-processes 32 \
    --address=0.0.0.0 \
    --port=8000 \
    --warmup
```

`--max-processes` controls how many concurrent Lean REPLs the server can serve.
The RL process queries each server's `/status` endpoint at startup and spawns
exactly one actor thread per process slot, fanning across all listed servers.
Wait for the server's imports to finish before launching RL; `/status` reports
ready before the imports actually settle.

## Launch

Single GPU:

```bash
python -m nanoproof.rl \
    --model-path sft/<run>/model_<step>.pt \
    --lean-servers 10.10.25.31:8000 \
    --lean-project /path/to/leantree_project
```

Multi-GPU with DDP:

```bash
torchrun --standalone --nproc_per_node=2 -m nanoproof.rl \
    --model-path sft/<run>/model_<step>.pt \
    --lean-servers 10.10.25.31:8000 10.10.25.32:8000 \
    --lean-project /path/to/leantree_project
```

`--model-path` is resolved relative to `$NANOPROOF_HOME/models/` if not
absolute. `--lean-project` falls back to `$LEAN_PROJECT_PATH` if unset; the
Lean version is read from its `lean-toolchain` file and used to pick the
matching dataset whitelists.

To resume a crashed run, point at the prior log directory instead of supplying
a model:

```bash
torchrun --standalone --nproc_per_node=2 -m nanoproof.rl \
    --resume-from rl/<prior_run> \
    --lean-servers ... --lean-project ...
```

This loads the latest checkpoint (model, optimizer, step, replay and negative
buffer shards, matchmaker stats). If the prior run only saved partial optimizer
state, add `--resume-fresh-optimizer` to start the optimizer from scratch.

Useful collection / training flags:

- `--datasets numinamath leanworkbook deepseek_prover` chooses the theorem mix.
- `--num-sampled-tactics`, `--num-simulations-eval`, `--first-token-occurrences-cap`,
  `--max-gen-tokens` tune the search.
- `--disable-solvers` filters `{grind, lia, grobner, aesop}` from model output.
  Collection still tries `grind` on unexpanded leaves as a free finisher (kept
  in the proof tree but excluded from the replay buffer); eval injects `grind`
  as a synthetic candidate at every node.
- `--no-proof-simplification` skips the redundant-node prune during collection.
- `--unlikelihood-weight`, `--negative-fraction`, `--negative-buffer-window-size`
  control unlikelihood training on failed tactics.
- `--value-weight`, `--fraction-sft`, `--device-batch-size`,
  `--target-examples-per-step`, `--num-updates-per-step` shape the training
  step.
- `--eval-every`, `--save-every`, `--eval-start` accept either `Nsteps` or a
  `H:M:S` interval.
- `--memory-profile DIR` dumps a CUDA memory snapshot on first OOM.

Run `python -m nanoproof.rl --help` for the full list.

## Web monitor

When the RL loop starts on the master rank, it launches a Flask monitor on
`http://localhost:5050`. The page shows training stats, prover thread states,
GPU and Lean server health, evaluation history, and a live log stream.

The React frontend lives in [nanoproof/web/](nanoproof/web/):

```bash
cd nanoproof/web
npm install
npm run build
```

To poke at the UI without running real training:

```bash
python tests/test_cli.py
```


# Evaluation

Use `scripts/prover_eval.py` to score a checkpoint on a benchmark:

```bash
python scripts/prover_eval.py \
    --model-path rl/<run>/model_<step>.pt \
    --lean-servers 10.10.25.31:8000 10.10.25.32:8000 \
    --datasets minif2f \
    --split valid \
    --num-simulations 512
```

Pass `--run-dir rl/<run>` instead of `--model-path` to sweep every checkpoint
in the run; the order is bisected (middle, quartiles, eighths) so an
interrupted sweep still has even step coverage. `--datasets` accepts a
comma-separated subset of `minif2f,leanworkbook,proofnet`. `--continue` retries
only theorems that previously errored, `--force` overwrites prior results, and
`--output-dir` overrides the default `<checkpoint_dir>/eval_<step>_<dataset>/`
location (single model + single dataset only). Most of the search and
inference flags from `nanoproof.rl` are also available here.

For a quick repeat-runs estimate of MiniF2F-test variance:

```bash
python scripts/test_eval.py \
    --model-path rl/<run>/model_<step>.pt \
    --lean-servers ... \
    --num-runs 8
```


# Other scripts

A handful of small utilities under `scripts/`:

- `scripts/prove.py` runs the prover against a single theorem (or REPL).
- `scripts/interact.py` is an interactive prover REPL (raw engine or tactic model).
- `scripts/bench_inference.py` benchmarks tactic-generation throughput.
- `scripts/inspect_buffer.py`, `inspect_parquet.py`, `inspect_problems.py`,
  `inspect_proofs.py` print the contents of replay buffers, parquet shards,
  and proof artifacts.
- `scripts/tok_eval.py`, `scripts/tok_show.py` inspect the tokenizer.


# Ideas

- try training on `state_after` as well, just to give the model more training
  signal (it was done in some paper, maybe GPT-f)
- let tokens attend bi-directionally inside the fixed-size state (a la PrefixLM)
- try proving the negation in each node (if the critic deems it likely to succeed)
- critic training: sampling ratio based on proof length (a la https://leandojo.org/leanprogress.html)
- tactic logprob filtering and retraining (akin to BFS-Prover-2)


# Cite

If you find nanoproof helpful in your research cite simply as:

```
@misc{nanoproof,
  author = {Matej Kripner},
  title = {nanoproof},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/kripner/nanoproof}
}
```
