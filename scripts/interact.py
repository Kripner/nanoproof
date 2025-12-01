import torch

from nanoproof.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb, autodetect_device_type
from nanoproof.checkpoints import load_model, save_checkpoint
from nanoproof.engine import Engine
from nanoproof.data.minif2f import list_theorems, get_imports

source = "sft" # which checkpoint to load the model from
model_tag = "d20" # model tag to load the model from
device_type = "" # cuda|cpu|mps (empty => autodetect)
dtype = "bfloat16"
base_dir = get_base_dir()

device_type = autodetect_device_type() if device_type == "" else device_type
device = torch.device(device_type)
ptdtype = torch.float32 if dtype == "float32" else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

model, tokenizer, meta = load_model(source, device, phase="eval", model_tag=model_tag)
engine = Engine(model, tokenizer)

def get_input() -> str:
    lines = []
    print("Type in a tactic state, followed by an empty line:")
    line = input()
    while line.strip() or not lines:
        lines.append(line)
        line = input()
    return "\n".join(lines)

inp = get_input()
while inp.strip() not in ["q", "quit", "exit"]:
    print("Sampling ...")
    # TODO: revert this after re-running the fixed SFT!!!
    # tokens = tokenizer(inp.strip() + "\n<|tactic|> ", prepend="<|endoftext|>")
    tokens = tokenizer(inp.strip() + "<|endoftext|>\n<|tactic|> ", prepend="<|endoftext|>")
    with autocast_ctx:
        sample_toks, _ = engine.generate_batch(tokens, num_samples=1, min_tokens=1)
    tactic = tokenizer.decode(sample_toks[0])
    print(f"Tactic:\n--\n'{tactic}'\n--")

    inp = get_input()
print("Done.")

"""
z : ℂ
h₀ : z = (1 + Complex.I) / ↑√2
⊢ (∑ k ∈ Finset.Icc 1 12, z ^ k ^ 2) * ∑ k ∈ Finset.Icc 1 12, 1 / z ^ k ^ 2 = 36
"""