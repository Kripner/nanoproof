# Source: https://github.com/karpathy/nanochat/blob/master/nanochat/tokenizer.py
"""
Tokenizer wrapper: GPT-2 BPE + extra special tokens for Lean / math.
Built by scripts/tok_build.py.
"""

import os

import torch
from tokenizers import Tokenizer as HFTokenizer

from nanoproof.common import GLOBAL_CONFIG, get_base_dir

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|pad|>",
    "<|tactic|>",
    "<|value|>",
    *[f"<|bin_{i:02d}|>" for i in range(1, GLOBAL_CONFIG.num_value_bins + 1)],
    # these occur at least 1000 times in Mathlib but do not have dedicated tokens in GPT-2
    "ˢ", "ˣ", "Γ", "Δ", "Λ", "Π", "Σ", "Φ", "Ω", "δ", "ζ", "η", "θ", "φ", "χ", "ψ", "ϕ", "ᵈ", "ᵐ", "ᵒ", "ᵖ", "ᵢ", "ᵣ", "ᵥ", "ᶜ", "ᶠ", "‖", "‹", "›", "⁅", "⁆", "⁰", "⁻", "₀", "₁", "₂", "₃", "₄", "₊", "ₐ", "ₑ", "ₗ", "ₘ", "ₙ", "ₚ", "ₛ", "ₜ", "ℂ", "ℕ", "ℚ", "ℝ", "ℤ", "ℱ", "←", "↔", "↦", "↪", "⇑", "∀", "∂", "∃", "∅", "∈", "∉", "∏", "∑", "∘", "∞", "∣", "∧", "∨", "∩", "∪", "∫", "≃", "≅", "≠", "≡", "≤", "≥", "≪", "≫", "⊆", "⊓", "⊔", "⊕", "⊗", "⊢", "⊤", "⊥", "⋂", "⋃", "⋆", "⋙", "▷", "▸", "◁", "⟦", "⟧", "⟨", "⟩", "⟪", "⟫", "⟶", "⥤", "⦃", "⦄", "⧸", "⨅", "⨆", "𝒜", "𝒰", "𝓘", "𝓝", "𝔖", "𝕜", "𝟙",
    # these are left out because they are already in GPT2 tokenizer (although weirdly not reported in tok_show): "¬", "¹"
]

class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        # init from a HuggingFace pretrained tokenizer (e.g. "gpt2")
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # init from a local directory on disk (e.g. "out/tokenizer")
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None):
        # encode a single string
        # prepend/append can be either a string of a special token or a token id directly.
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        # encode a single special token via exact match
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        bos = self.encode_special("<|endoftext|>")
        return bos

    def get_eos_token_id(self):
        eos = self.encode_special("<|endoftext|>")
        return eos

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        # save the tokenizer to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

    def get_value_token_ids(self) -> list[int]:
        return [self.encode_special(f"<|bin_{i:02d}|>") for i in range(1, GLOBAL_CONFIG.num_value_bins + 1)]

    def get_value_bins(self) -> list[int]:
        return list(range(1, GLOBAL_CONFIG.num_value_bins + 1))

def value_to_token_ids(tokenizer, value: int) -> list[int]:
    """Convert a value (1..num_value_bins) to a single bin token ID."""
    assert 1 <= value <= GLOBAL_CONFIG.num_value_bins
    bin_token = f"<|bin_{value:02d}|>"
    return [tokenizer.encode_special(bin_token)]

def token_ids_to_value(tokenizer, token_ids: list[int]) -> int | None:
    """Convert a bin token ID back to a value (1..num_value_bins). Returns None if not a valid bin token."""
    if len(token_ids) != 1:
        return None
    token_id = token_ids[0]
    # Check each bin token
    for i in range(1, GLOBAL_CONFIG.num_value_bins + 1):
        if tokenizer.encode_special(f"<|bin_{i:02d}|>") == token_id:
            return i
    return None

def get_tokenizer():
    # return HuggingFaceTokenizer.from_pretrained("gpt2")
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    return HuggingFaceTokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_build.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes