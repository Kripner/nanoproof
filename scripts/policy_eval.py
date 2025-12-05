import torch
from itertools import islice
import sys
import os

from nanoproof.common import compute_init, autodetect_device_type, print0
from nanoproof.checkpoints import load_model
from nanoproof.data.leantree import iter_data
from nanoproof.data.leantree_dataloader import sft_data_generator

@torch.inference_mode()
def eval_tactic_accuracy(model, leantree_batches, max_steps=None):
    total_samples = 0
    total_full_correct = 0
    total_first_token_correct = 0
    
    for x, y in leantree_batches if max_steps is None else islice(leantree_batches, max_steps):
        logits = model(x) # (B, T, V)
        predictions = torch.argmax(logits, dim=-1) # (B, T)

        mask = (y != -1)
        correct = predictions == y

        assert mask.any(dim=1).all(), "leantree sample contained no output tokens"
        total_samples += logits.shape[0]

        # Full Accuracy: correctness on all non-masked tokens
        total_full_correct += (correct | torch.logical_not(mask)).all(dim=1).sum().item()

        # First Token Accuracy: correctness on the first non-masked token
        first_token_indices = mask.argmax(dim=1)  # argmax returns the first True index
        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        total_first_token_correct += correct[batch_indices, first_token_indices].sum().item()

    return {
        "full_acc": total_full_correct / total_samples,
        "first_token_acc": total_first_token_correct / total_samples,
    }

def main():
    # Setup
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    
    # Load model
    # Try to load d26 base model
    print0("Loading model...")
    try:
        model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag="d26")
    except Exception as e:
        print0(f"Could not load d26 sft model: {e}")
        print0("Trying generic base model...")
        try:
             model, tokenizer, meta = load_model("base", device, phase="eval")
        except Exception as e2:
             print0(f"Failed to load base model: {e2}")
             return

    print0(f"Model loaded. Config: {meta.get('model_config', 'N/A')}")

    # Load Data
    print0("Loading dataset...")
    split = "val"
    try:
        dataset = list(iter_data(split=split))
    except Exception as e:
        print0(f"Val split failed ({e}), falling back to train split (first 1000 items)")
        dataset = list(iter_data(split="train"))
        dataset = dataset[:1000] 
    
    if len(dataset) == 0:
        print0("Dataset is empty!")
        return

    batch_size = 32
    
    # Calculate steps
    # We want to iterate through the dataset exactly once.
    # sft_data_generator yields batches of size `batch_size`.
    # It repeats the dataset indefinitely.
    # We calculate how many batches correspond to one epoch.
    # Each item in dataset produces 2 samples.
    # DDP handles sharding.
    
    my_dataset_len = len(range(ddp_rank, len(dataset), ddp_world_size))
    total_samples_local = my_dataset_len * 2
    steps = total_samples_local // batch_size
    
    if steps == 0:
        print0("Not enough data for one batch.")
        return

    print0(f"Evaluating on {steps} batches (approx {steps * batch_size} samples)...")
    
    data_gen = sft_data_generator(dataset, batch_size, device=device)
    
    results = eval_tactic_accuracy(model, data_gen, max_steps=steps)
    
    print0(f"Results for split '{split}':")
    print0(f"Full Accuracy: {results['full_acc']:.4%}")
    print0(f"First Token Accuracy: {results['first_token_acc']:.4%}")

if __name__ == "__main__":
    main()
