"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import json
import logging
from dataclasses import fields

import torch

from nanoproof.model import Transformer, NetworkConfig
from nanoproof.tokenizer import get_tokenizer
from nanoproof.common import get_base_dir, setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)


def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")


# -----------------------------------------------------------------------------
# Checkpoint path parsing
#
# Throughout the codebase a "model path" is a path to a specific
# ``model_NNNNNN.pt`` file. The path may be absolute, or relative to
# ``$NANOPROOF_HOME/models/``. We never resolve "latest in directory" magic
# anywhere - every load explicitly names the checkpoint file it wants.

def parse_checkpoint_path(model_path: str) -> tuple[str, int]:
    """Resolve a checkpoint file path and parse the step from its filename.

    ``model_path`` is either absolute or relative to ``$NANOPROOF_HOME/models/``.
    It must point to a ``model_NNNNNN.pt`` file (the file does not need to
    exist at this point - only the filename is parsed).

    Returns ``(checkpoint_dir, step)``.
    """
    full = model_path if os.path.isabs(model_path) else os.path.join(get_base_dir(), "models", model_path)
    basename = os.path.basename(full)
    if not basename.startswith("model_") or not basename.endswith(".pt"):
        raise ValueError(
            f"Expected a path to a 'model_NNNNNN.pt' file, got: {model_path!r}"
        )
    step_str = basename.removeprefix("model_").removesuffix(".pt")
    try:
        step = int(step_str)
    except ValueError as e:
        raise ValueError(
            f"Could not parse step from checkpoint filename {basename!r}: {e}"
        )
    return os.path.dirname(full), step


def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    log0(f"Building model with config: {model_config_kwargs}")
    # Filter to known NetworkConfig fields so older checkpoints (which carried
    # extra fields like num_value_bins / max_tactic_len) still load.
    valid_fields = {f.name for f in fields(NetworkConfig)}
    model_config = NetworkConfig(**{k: v for k, v in model_config_kwargs.items() if k in valid_fields})
    with torch.device("meta"):
        model = Transformer(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init

    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]
    return model, tokenizer, meta_data


# -----------------------------------------------------------------------------
# Eval result persistence
# -----------------------------------------------------------------------------

from dataclasses import dataclass

@dataclass
class CheckpointInfo:
    """Information about a loaded checkpoint, used for saving eval results."""
    checkpoint_dir: str
    step: int
    seed: int = 0

    def get_eval_path(self, dataset_name: str) -> str:
        seed_suffix = f"-{self.seed}" if self.seed != 0 else ""
        return os.path.join(self.checkpoint_dir, f"eval_{self.step:06d}_{dataset_name}{seed_suffix}.jsonl")


def write_eval_results_jsonl(jsonl_path: str, results: dict, prepend_entries: list[dict] = None):
    """Write evaluation results to a JSONL file."""
    detailed_results = results.get("detailed_results", [])

    if not detailed_results and not prepend_entries:
        logger.info(f"Skipping write of empty eval results to {jsonl_path}")
        return

    with open(jsonl_path, "w") as f:
        if prepend_entries:
            for entry in prepend_entries:
                f.write(json.dumps(entry) + "\n")

        for item in detailed_results:
            entry = {
                "theorem": item["theorem"],
                "header": item.get("header"),
                "name": item.get("name"),
                "proof": item["proof_tree"],
                "unsimplified_proof": item.get("unsimplified_proof_tree"),
                "linearized_proof": item.get("linearized_proof"),
                "num_iterations": item["num_iterations"],
                "error": item.get("error"),
            }
            f.write(json.dumps(entry) + "\n")

    total_count = len(detailed_results) + (len(prepend_entries) if prepend_entries else 0)
    logger.info(f"Saved {total_count} eval results to {jsonl_path}")


def save_eval_results(checkpoint_info: CheckpointInfo, dataset_name: str, results: dict, prepend_entries: list[dict] = None):
    """Save evaluation results alongside the checkpoint."""
    jsonl_path = checkpoint_info.get_eval_path(dataset_name)
    write_eval_results_jsonl(jsonl_path, results, prepend_entries=prepend_entries)


def save_eval_results_to_run_dir(output_dir: str, step: int, dataset_name: str, results: dict):
    """Save evaluation results in the RL run's eval directory."""
    eval_dir = os.path.join(output_dir, "evals", str(step))
    os.makedirs(eval_dir, exist_ok=True)
    jsonl_path = os.path.join(eval_dir, f"{dataset_name}.jsonl")
    write_eval_results_jsonl(jsonl_path, results)


def load_existing_eval_results(jsonl_path: str) -> tuple[list[dict], list[dict]]:
    """Load existing results. Returns (successful_entries, error_entries)."""
    successful, errors = [], []
    with open(jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            (errors if entry.get("error") is not None else successful).append(entry)
    return successful, errors


# -----------------------------------------------------------------------------

def load_model(model_path: str, device, phase: str):
    """Load a model from a checkpoint .pt file path.

    ``model_path`` is either absolute or relative to ``$NANOPROOF_HOME/models/``,
    and must end in ``model_NNNNNN.pt``. Example:

        load_model("pretrain/10-49-50_07-04-26_baseline/model_005000.pt", device, "train")
    """
    checkpoint_dir, step = parse_checkpoint_path(model_path)
    log0(f"Loading model from {checkpoint_dir} (step {step})")
    return build_model(checkpoint_dir, step, device, phase)
