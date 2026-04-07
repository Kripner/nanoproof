"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import glob
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


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = None
    for f in checkpoint_files:
        step = os.path.basename(f).split("_")[-1].split(".")[0]
        try:
            step = int(step)
            last_step = step if last_step is None else max(last_step, step)
        except ValueError:
            pass
    if last_step is None:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return last_step

# -----------------------------------------------------------------------------
# Checkpoint path resolution


def resolve_model_dir(model_path: str) -> str:
    """Resolve a model path to an absolute checkpoint directory.

    If *model_path* is absolute, it is returned as-is.
    Otherwise it is resolved relative to ``{base_dir}/models/``,
    e.g. ``"pretrain/14-45-26_06-04-26_run"`` →
    ``~/.nanoproof/models/pretrain/14-45-26_06-04-26_run``.
    """
    if os.path.isabs(model_path):
        return model_path
    return os.path.join(get_base_dir(), "models", model_path)


def resolve_step(checkpoint_dir: str, step: int | None) -> int:
    """Resolve step to the latest if not specified."""
    if step is None:
        return find_last_step(checkpoint_dir)
    return step


# -----------------------------------------------------------------------------
# Model loading convenience functions


def load_model(model_path: str, device, phase: str, step: int | None = None):
    """Load a model from a checkpoint directory.

    *model_path* is resolved via :func:`resolve_model_dir`:
    absolute paths are used as-is; relative paths (e.g.
    ``"pretrain/14-45-26_06-04-26_run"``) are resolved under
    ``{NANOPROOF_HOME}/models/``.
    """
    checkpoint_dir = resolve_model_dir(model_path)
    step = resolve_step(checkpoint_dir, step)
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    return build_model(checkpoint_dir, step, device, phase)