# Questions

- how is the action prob obtained from tokens probs?
- is value predicted for state (as per paper) or for state-action (as per pseudocode)?
- how do the value bins correspond to values? Ie. what are the values of bins 0 and 63?

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
torchrun --nproc_per_node=2 -m nanoproof.pretrain
```