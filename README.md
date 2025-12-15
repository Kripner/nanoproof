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

# Ideas

- try training on state_after as well, just to give the model more training signal (it was done in some paper, maybe GPT-f)
- datasets: LEAN-Github (https://huggingface.co/datasets/internlm/Lean-Github), LeanUniverse (https://github.com/facebookresearch/LeanUniverse), Lean-Workbook, DeepSeek-Prover-V2 dataset

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


OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=2 -m -- nanoproof.pretrain --depth=20 --run="baseline"