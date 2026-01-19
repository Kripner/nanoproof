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

# TODO: don't we need attention masks?

@torch.inference_mode()
def eval_critic_errors(model, tokenizer, leantree_batches, max_steps=None):
    """Evaluate critic accuracy on value prediction.
    
    Returns confusion matrix and MSE for value bin predictions.
    Only processes samples where x contains the value_delim_tok.
    
    Returns:
        y_true: list of actual bin values (1-64)
        y_pred: list of predicted bin values (1-64)
        argmax_mse: MSE using argmax prediction over bin tokens
        soft_mse: MSE using softmax-weighted expected value
        total_samples: number of samples evaluated
    """
    value_delim_tok = tokenizer.encode_special("<|value|>")
    
    # Bin token IDs in order: index i corresponds to bin value (i+1)
    bin_token_ids = torch.tensor([
        tokenizer.encode_special(f"<|bin_{i:02d}|>") 
        for i in range(_MIN_VALUE, _MAX_VALUE + 1)
    ])
    # Reverse mapping: token_id -> bin_index (0-based)
    token_to_bin_idx = {tok.item(): i for i, tok in enumerate(bin_token_ids)}
    
    y_true = []  # actual bin values (1-64)
    y_pred = []  # predicted bin values (1-64)
    soft_squared_error_sum = 0.0
    
    for x, y, _, _ in leantree_batches if max_steps is None else islice(leantree_batches, max_steps):
        has_value = (x == value_delim_tok).any(dim=1)
        if not has_value.any():
            continue
        x, y = x[has_value], y[has_value]
        
        logits = model(x)  # (B, T, V)
        
        # Find value position and extract logits there
        value_positions = (x == value_delim_tok).int().argmax(dim=1)
        batch_indices = torch.arange(x.shape[0], device=x.device)
        actual_tokens = y[batch_indices, value_positions]
        value_logits = logits[batch_indices, value_positions]  # (B, V)
        
        # Extract bin logits and compute predictions
        bin_logits = value_logits[:, bin_token_ids.to(x.device)]  # (B, 64)
        argmax_bin_idx = bin_logits.argmax(dim=-1)  # (B,) 0-indexed
        bin_probs = torch.softmax(bin_logits, dim=-1)  # (B, 64)
        bin_values = torch.arange(1, _MAX_VALUE + 1, dtype=bin_probs.dtype, device=x.device)
        soft_predictions = (bin_probs * bin_values).sum(dim=-1)  # (B,)
        
        # Collect predictions for samples with valid actual bin token
        for i, actual_tok in enumerate(actual_tokens.tolist()):
            if actual_tok in token_to_bin_idx:
                actual_idx = token_to_bin_idx[actual_tok]
                pred_idx = argmax_bin_idx[i].item()
                y_true.append(actual_idx + 1)  # 1-indexed bin value
                y_pred.append(pred_idx + 1)    # 1-indexed bin value
                soft_squared_error_sum += (actual_idx + 1 - soft_predictions[i].item()) ** 2
    
    total_samples = len(y_true)
    if total_samples == 0:
        return {"y_true": [], "y_pred": [], "argmax_mse": float('nan'), "soft_mse": float('nan'), "total_samples": 0}
    
    # Argmax MSE
    argmax_mse = sum((a - p) ** 2 for a, p in zip(y_true, y_pred)) / total_samples
    
    # Soft MSE
    soft_mse = soft_squared_error_sum / total_samples
    
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "argmax_mse": argmax_mse,
        "soft_mse": soft_mse,
        "total_samples": total_samples,
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

        # Print predicted vs correct for each sample
        # for i in range(x.shape[0]):
        #     sample_mask = mask[i]
        #     pred_tokens = predictions[i][sample_mask].tolist()
        #     correct_tokens = y[i][sample_mask].tolist()
        #     pred_str = tokenizer.decode(pred_tokens)
        #     correct_str = tokenizer.decode(correct_tokens)
        #     is_correct = pred_tokens == correct_tokens
        #     print(f"[{'OK' if is_correct else 'X'}] Pred: {pred_str!r}")
        #     print(f"     Correct: {correct_str!r}")
        #     print()

    return {
        "full_acc": total_full_correct / total_samples,
        "first_token_acc": total_first_token_correct / total_samples,
        "total_samples": total_samples,
    }



def main():
    # Setup
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    
    print0("Loading model...")
    model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag="d26", step=903)
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
