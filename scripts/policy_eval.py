import torch
from itertools import islice
import sys
import os
from contextlib import nullcontext

from nanoproof.common import compute_init, autodetect_device_type, print0
from nanoproof.checkpoints import load_model
from nanoproof.data.leantree import iter_data
from nanoproof.data.leantree_dataloader import sft_data_generator

_MIN_VALUE = 1
_MAX_VALUE = 64


@torch.inference_mode()
def eval_critic_errors(model, tokenizer, leantree_batches, max_steps=None):
    """Evaluate critic accuracy on value prediction.
    
    Returns confusion matrix and MSE for value bin predictions.
    Only processes samples where x contains the value_delim_tok.
    """
    value_delim_tok = tokenizer.encode_special("<|value|>")
    
    # Build mapping from bin token IDs to bin values (1-64)
    bin_tok_to_value = {
        tokenizer.encode_special(f"<|bin_{i:02d}|>"): i 
        for i in range(_MIN_VALUE, _MAX_VALUE + 1)
    }
    
    # Confusion matrix: rows are actual, columns are predicted (0-indexed for bins 1-64)
    confusion = torch.zeros(_MAX_VALUE, _MAX_VALUE, dtype=torch.int64)
    
    for x, y, _, _ in leantree_batches if max_steps is None else islice(leantree_batches, max_steps):
        # Only process samples where input contains value_delim_tok
        has_value = (x == value_delim_tok).any(dim=1)
        if not has_value.any():
            continue
        x, y = x[has_value], y[has_value]
        
        logits = model(x)  # (B, T, V)
        predictions = torch.argmax(logits, dim=-1)  # (B, T)
        
        # Find the index of value_delim_tok in each sample (first occurrence)
        value_positions = (x == value_delim_tok).int().argmax(dim=1)
        batch_indices = torch.arange(x.shape[0], device=x.device)
        
        # Get actual and predicted tokens at value position
        actual_tokens = y[batch_indices, value_positions]
        predicted_tokens = predictions[batch_indices, value_positions]
        
        # Convert to bin values and update confusion matrix
        for actual_tok, pred_tok in zip(actual_tokens.tolist(), predicted_tokens.tolist()):
            if actual_tok in bin_tok_to_value:
                actual_bin = bin_tok_to_value[actual_tok]
                pred_bin = bin_tok_to_value.get(pred_tok)
                if pred_bin is not None:
                    confusion[actual_bin - 1, pred_bin - 1] += 1
    
    # Calculate MSE from confusion matrix
    total_samples = confusion.sum().item()
    if total_samples == 0:
        return {"confusion": confusion, "mse": float('nan'), "total_samples": 0}
    
    # MSE = sum over all (actual - predicted)^2 * count / total_count
    actual_indices = torch.arange(1, _MAX_VALUE + 1).unsqueeze(1)  # (64, 1)
    pred_indices = torch.arange(1, _MAX_VALUE + 1).unsqueeze(0)    # (1, 64)
    squared_errors = (actual_indices - pred_indices) ** 2          # (64, 64)
    mse = (confusion * squared_errors).sum().item() / total_samples
    
    return {
        "confusion": confusion,
        "mse": mse,
        "total_samples": int(total_samples),
    }


@torch.inference_mode()
def eval_tactic_accuracy(model, tokenizer, leantree_batches, max_steps=None):
    total_samples = 0
    total_full_correct = 0
    total_first_token_correct = 0
    value_delim_tok = tokenizer.encode_special("<|value|>")
    
    for x, y, _, _ in leantree_batches if max_steps is None else islice(leantree_batches, max_steps):
        # Skip samples where input contains value_delim_tok
        valid = ~(x == value_delim_tok).any(dim=1)
        x, y = x[valid], y[valid]
        if x.shape[0] == 0:
            continue

        logits = model(x) # (B, T, V)
        predictions = torch.argmax(logits, dim=-1) # (B, T)

        mask = (y != -1)
        correct = predictions == y

        assert mask.any(dim=1).all(), "leantree sample contained no output tokens"
        total_samples += logits.shape[0]

        # Full Accuracy: correctness on all non-masked tokens
        total_full_correct += (correct | torch.logical_not(mask)).all(dim=1).sum().item()

        # First Token Accuracy: correctness on the first non-masked token
        first_token_indices = mask.int().argmax(dim=1)  # argmax returns the first True index
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
    
    print0("Loading model...")
    model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag="d26")
    model.eval()

    print0(f"Model loaded. Config: {meta.get('model_config', 'N/A')}")

    # Load Data
    print0("Loading dataset...")
    split = "val"
    dataset = list(iter_data(split=split))
    
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
    
    dtype = "bfloat16"
    ptdtype = torch.float32 if dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
    with autocast_ctx:
        results = eval_tactic_accuracy(model, tokenizer, data_gen, max_steps=steps)
    
    print0(f"Results for split '{split}':")
    print0(f"Full Accuracy: {results['full_acc']:.4%}")
    print0(f"First Token Accuracy: {results['first_token_acc']:.4%}")

if __name__ == "__main__":
    main()
