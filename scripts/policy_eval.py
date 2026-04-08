import argparse

import torch
import torch.distributed as dist


from nanoproof.common import compute_init, autodetect_device_type, print0, is_ddp, get_dist_info, GLOBAL_CONFIG
from nanoproof.checkpoints import load_model
from nanoproof.data.sft.leantree import iter_data
from nanoproof.data.sft.leantree_dataloader import sft_data_generator


def _reduce_if_ddp(tensor):
    """Reduce tensor across DDP ranks if in DDP mode."""
    if is_ddp():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _compute_entropy(logits):
    """Compute entropy of a distribution from logits. Returns entropy per sample.
    
    Args:
        logits: (..., V) logits over vocabulary
    Returns:
        entropy: (...) entropy values in nats
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    # H = -sum(p * log(p)), handle p=0 case (0 * -inf = 0)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


@torch.inference_mode()
def eval_critic_errors(model, tokenizer, leantree_batches, eval_steps=None):
    """Evaluate critic accuracy on value prediction.

    Returns confusion matrix and MSE for value bin predictions.
    Only processes samples where x contains the value_delim_tok. Batches that
    contain no value samples are skipped and do NOT count toward eval_steps;
    `eval_steps` is the number of batches that actually contributed.

    In DDP mode, results are automatically reduced across all ranks.

    Returns:
        y_true: list of actual bin values (1-64)
        y_pred: list of predicted bin values (1-64)
        argmax_mse: MSE using argmax prediction over bin tokens
        soft_mse: MSE using softmax-weighted expected value
        entropy: mean entropy of value prediction (over bin tokens)
        total_samples: number of samples evaluated
    """
    value_delim_tok = tokenizer.encode_special("<|value|>")
    device = next(model.parameters()).device
    
    # Bin token IDs in order: index i corresponds to bin value (i+1)
    bin_token_ids = torch.tensor([
        tokenizer.encode_special(f"<|bin_{i:02d}|>")
        for i in range(1, GLOBAL_CONFIG.num_value_bins + 1)
    ])
    # Reverse mapping: token_id -> bin_index (0-based)
    token_to_bin_idx = {tok.item(): i for i, tok in enumerate(bin_token_ids)}
    
    y_true = []  # actual bin values (1-64)
    y_pred = []  # predicted bin values (1-64)
    soft_squared_error_sum = 0.0
    argmax_squared_error_sum = 0.0
    entropy_sum = 0.0
    
    steps_done = 0
    for x, y, _, _ in leantree_batches:
        if eval_steps is not None and steps_done >= eval_steps:
            break
        has_value = (x == value_delim_tok).any(dim=1)
        if not has_value.any():
            continue
        x, y = x[has_value], y[has_value]
        steps_done += 1
        
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
        bin_values = torch.arange(1, GLOBAL_CONFIG.num_value_bins + 1, dtype=bin_probs.dtype, device=x.device)
        soft_predictions = (bin_probs * bin_values).sum(dim=-1)  # (B,)
        
        # Compute entropy over bin tokens
        bin_entropy = _compute_entropy(bin_logits)  # (B,)
        
        # Collect predictions for samples with valid actual bin token
        for i, actual_tok in enumerate(actual_tokens.tolist()):
            if actual_tok in token_to_bin_idx:
                actual_idx = token_to_bin_idx[actual_tok]
                pred_idx = argmax_bin_idx[i].item()
                y_true.append(actual_idx + 1)  # 1-indexed bin value
                y_pred.append(pred_idx + 1)    # 1-indexed bin value
                argmax_squared_error_sum += (actual_idx + 1 - (pred_idx + 1)) ** 2
                soft_squared_error_sum += (actual_idx + 1 - soft_predictions[i].item()) ** 2
                entropy_sum += bin_entropy[i].item()
    
    total_samples = len(y_true)
    
    # Reduce across DDP ranks
    stats = torch.tensor([argmax_squared_error_sum, soft_squared_error_sum, entropy_sum, total_samples], 
                         dtype=torch.float64, device=device)
    _reduce_if_ddp(stats)
    argmax_squared_error_sum, soft_squared_error_sum, entropy_sum, total_samples = stats.tolist()
    total_samples = int(total_samples)
    
    # Gather y_true/y_pred from all ranks for confusion matrix
    if is_ddp():
        _, _, _, ddp_world_size = get_dist_info()
        all_y_true = [None] * ddp_world_size
        all_y_pred = [None] * ddp_world_size
        dist.all_gather_object(all_y_true, y_true)
        dist.all_gather_object(all_y_pred, y_pred)
        y_true = [v for sublist in all_y_true for v in sublist]
        y_pred = [v for sublist in all_y_pred for v in sublist]
    
    if total_samples == 0:
        return {"y_true": [], "y_pred": [], "argmax_mse": float('nan'), "soft_mse": float('nan'), 
                "entropy": float('nan'), "total_samples": 0}
    
    # Compute final metrics from reduced sums
    argmax_mse = argmax_squared_error_sum / total_samples
    soft_mse = soft_squared_error_sum / total_samples
    entropy = entropy_sum / total_samples
    
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "argmax_mse": argmax_mse,
        "soft_mse": soft_mse,
        "entropy": entropy,
        "total_samples": total_samples,
    }


@torch.inference_mode()
def eval_tactic_accuracy(model, tokenizer, leantree_batches, eval_steps=None):
    """Evaluate tactic prediction accuracy.

    Batches with no policy samples (i.e. all rows are value samples) are
    skipped and do NOT count toward eval_steps; `eval_steps` is the number of
    batches that actually contributed.

    In DDP mode, results are automatically reduced across all ranks.

    Returns:
        full_acc: fraction of samples where all tokens are predicted correctly
        first_token_acc: fraction of samples where first token is predicted correctly
        first_token_entropy: mean entropy of first token prediction
        all_tokens_entropy: mean entropy across all predicted tokens
        total_samples: number of samples evaluated
    """
    total_samples = 0
    total_full_correct = 0
    total_first_token_correct = 0
    first_token_entropy_sum = 0.0
    all_tokens_entropy_sum = 0.0
    total_tokens = 0  # count of all masked tokens for entropy averaging
    value_delim_tok = tokenizer.encode_special("<|value|>")
    device = next(model.parameters()).device
    
    steps_done = 0
    for x, y, _, _ in leantree_batches:
        if eval_steps is not None and steps_done >= eval_steps:
            break
        # Skip samples where input contains value_delim_tok
        valid = ~(x == value_delim_tok).any(dim=1)
        x, y = x[valid], y[valid]
        if x.shape[0] == 0:
            continue
        steps_done += 1

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
        
        # Entropy calculations
        token_entropy = _compute_entropy(logits)  # (B, T)
        
        # First token entropy
        first_token_entropy_sum += token_entropy[batch_indices, first_token_indices].sum().item()
        
        # All tokens entropy (only for masked positions)
        all_tokens_entropy_sum += (token_entropy * mask).sum().item()
        total_tokens += mask.sum().item()

    # Reduce across DDP ranks
    stats = torch.tensor([
        total_full_correct, total_first_token_correct, 
        first_token_entropy_sum, all_tokens_entropy_sum, 
        total_samples, total_tokens
    ], dtype=torch.float64, device=device)
    _reduce_if_ddp(stats)
    (total_full_correct, total_first_token_correct, 
     first_token_entropy_sum, all_tokens_entropy_sum, 
     total_samples, total_tokens) = stats.tolist()
    total_samples = int(total_samples)
    total_tokens = int(total_tokens)
    
    if total_samples == 0:
        return {"full_acc": float('nan'), "first_token_acc": float('nan'), 
                "first_token_entropy": float('nan'), "all_tokens_entropy": float('nan'),
                "total_samples": 0}

    return {
        "full_acc": total_full_correct / total_samples,
        "first_token_acc": total_first_token_correct / total_samples,
        "first_token_entropy": first_token_entropy_sum / total_samples,
        "all_tokens_entropy": all_tokens_entropy_sum / total_tokens if total_tokens > 0 else float('nan'),
        "total_samples": total_samples,
    }



def _main():
    parser = argparse.ArgumentParser(description="Evaluate a model's tactic and critic accuracy on the leantree validation set")
    parser.add_argument("--model-path", type=str, required=True, help="path to model_NNNNNN.pt (relative to models/ or absolute)")
    parser.add_argument("--split", type=str, default="valid", help="dataset split to evaluate on")
    parser.add_argument("--batch-size", type=int, default=32, help="evaluation batch size")
    parser.add_argument("--eval-steps", type=int, default=None, help="cap on number of contributing batches per eval (default: full split)")
    args = parser.parse_args()

    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    print0(f"Loading model from {args.model_path}...")
    model, tokenizer, meta = load_model(args.model_path, device, phase="eval")
    model.eval()
    print0(f"Model loaded. Config: {meta.get('model_config', 'N/A')}")

    print0(f"Loading dataset (split={args.split})...")
    dataset = list(iter_data(split=args.split))
    if len(dataset) == 0:
        print0("Dataset is empty!")
        return
    print0(f"Dataset rows: {len(dataset)}")

    build_loader = lambda: sft_data_generator(dataset, batch_size=args.batch_size, device=device)

    tactic_results = eval_tactic_accuracy(model, tokenizer, build_loader(), eval_steps=args.eval_steps)
    critic_results = eval_critic_errors(model, tokenizer, build_loader(), eval_steps=args.eval_steps)

    print0(f"Results for split '{args.split}':")
    print0(f"  Tactic samples:           {tactic_results['total_samples']}")
    print0(f"  Tactic full accuracy:     {tactic_results['full_acc']:.4%}")
    print0(f"  Tactic first-token acc:   {tactic_results['first_token_acc']:.4%}")
    print0(f"  Tactic first-token entr:  {tactic_results['first_token_entropy']:.4f}")
    print0(f"  Tactic all-tokens entr:   {tactic_results['all_tokens_entropy']:.4f}")
    print0(f"  Critic samples:           {critic_results['total_samples']}")
    print0(f"  Critic argmax MSE:        {critic_results['argmax_mse']:.4f}")
    print0(f"  Critic soft MSE:          {critic_results['soft_mse']:.4f}")
    print0(f"  Critic entropy:           {critic_results['entropy']:.4f}")


if __name__ == "__main__":
    _main()
