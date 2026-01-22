from contextlib import nullcontext

import torch

from nanoproof.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb, autodetect_device_type
from nanoproof.checkpoints import load_model, load_rl_model
from nanoproof.engine import Engine
from nanoproof.inference import TacticModel

MODE = "raw_engine"  # raw_engine | tactic_model
generate = None
predict_value = None

model_tag = "d26" # model tag to load the model from

# step = 903
# source = "sft" # which checkpoint to load the model from

run_name = "26-01-21_18-37-value_i03guzht"
step = 2000
source = "rl" # which checkpoint to load the model from

if MODE == "raw_engine":
    device_type = "" # cuda|cpu|mps (empty => autodetect)
    dtype = "bfloat16"

    device_type = autodetect_device_type() if device_type == "" else device_type
    device = torch.device(device_type)
    ptdtype = torch.float32 if dtype == "float32" else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    if source == "sft":
        model, tokenizer, meta = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
    elif source == "rl":
        model, tokenizer, meta = load_rl_model(run_name, device, phase="eval", step=step)
    engine = Engine(model, tokenizer)
    value_token_ids = tokenizer.get_value_token_ids()
    value_bins = tokenizer.get_value_bins()

    def generate_(inp_) -> list[str]:
        tokens = tokenizer(inp_.strip() + "\n<|tactic|>", prepend=tokenizer.get_bos_token_id())
        with autocast_ctx:
            sample_toks, _ = engine.generate_batch(tokens, num_samples=6, min_tokens=1, max_tokens=64)
        return [tokenizer.decode(sample_toks[i]) for i in range(6)]

    def predict_value_(inp_) -> float:
        tokens = tokenizer(inp_.strip() + "\n<|value|>", prepend=tokenizer.get_bos_token_id())
        with autocast_ctx:
            _, _, value_logits = engine.generate_batch(tokens, num_samples=1, min_tokens=1, max_tokens=1, return_logits=True)
            value_logits = value_logits[0][-1]
            value_logits = torch.gather(value_logits, 0, torch.tensor(value_token_ids, device=device))
            value_probs = torch.softmax(value_logits, dim=-1)
            for i, prob in enumerate(value_probs):
                print(f"BIN {value_bins[i]}: {prob.item()}")
            value_probs = value_probs * torch.tensor(value_bins, device=device, dtype=value_probs.dtype)
            value_probs = value_probs.sum()
            return value_probs.item()

    generate = generate_
    predict_value = predict_value_

elif MODE == "tactic_model":
    assert source == "sft", "tactic_model mode only supports sft source for now"
    tactic_model = TacticModel.create(source=source, model_tag=model_tag, step=step)
    _cached_value = None  # Cache value from tactic generation

    def generate_(inp_) -> list[str]:
        global _cached_value
        result = tactic_model.sample_tactic_from_str(inp_.strip())
        if not result.is_success():
            raise RuntimeError(f"Tactic generation failed: {result.error}")
        tactics, value = result.value
        _cached_value = value
        return tactics

    def predict_value_(inp_) -> float:
        # Value was already computed during tactic generation
        global _cached_value
        if _cached_value is not None:
            return _cached_value
        # Fallback: call sample_tactic to get value
        result = tactic_model.sample_tactic_from_str(inp_.strip())
        if not result.is_success():
            raise RuntimeError(f"Value prediction failed: {result.error}")
        _, value = result.value
        return value

    generate = generate_
    predict_value = predict_value_

else:
    raise ValueError(f"Invalid mode: {MODE}")

def get_input() -> str:
    lines = []
    print("Type in a tactic state, followed by an empty line:")
    line = input()
    while line.strip() or not lines:
        lines.append(line.rstrip())
        line = input()
    return "\n".join(lines)

inp = get_input()
while inp.strip() not in ["q", "quit", "exit"]:
    print(f"Generating tactics ...")
    tactics = generate(inp)
    for tactic in tactics:
        print(f"Tactic:\n--\n'{tactic}'\n--")
        print()
    print(f"Predicting value...")
    value = predict_value(inp)
    print(f"Value: {value}")
    print()
    inp = get_input()
print("Done.")

INP1 = """
z : ℂ
h₀ : z = (1 + Complex.I) / ↑√2
⊢ (∑ k ∈ Finset.Icc 1 12, z ^ k ^ 2) * ∑ k ∈ Finset.Icc 1 12, 1 / z ^ k ^ 2 = 36
"""

INP2 = """
⊢ 2 + 3 = 5
"""