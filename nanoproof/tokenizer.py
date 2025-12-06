# Source: https://github.com/karpathy/nanochat/blob/master/nanochat/tokenizer.py
"""
BPE Tokenizer in the style of GPT-4.

Two implementations are available:
1) HuggingFace Tokenizer that can do both training and inference but is really confusing
2) Our own RustBPE Tokenizer for training and tiktoken for efficient inference
"""

import os

from nanoproof.model import NetworkConfig

_MIN_VALUE = 1
_MAX_VALUE = 64  # max value corresponds to "infinity"

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|pad|>",
    "<|tactic|>",
    "<|value|>",
    *[f"<|bin_{i:02d}|>" for i in range(_MIN_VALUE, _MAX_VALUE + 1)],
    # these occur at least 1000 times in Mathlib but do not have dedicated tokens in GPT-2
    "ˢ", "ˣ", "Γ", "Δ", "Λ", "Π", "Σ", "Φ", "Ω", "δ", "ζ", "η", "θ", "φ", "χ", "ψ", "ϕ", "ᵈ", "ᵐ", "ᵒ", "ᵖ", "ᵢ", "ᵣ", "ᵥ", "ᶜ", "ᶠ", "‖", "‹", "›", "⁅", "⁆", "⁰", "⁻", "₀", "₁", "₂", "₃", "₄", "₊", "ₐ", "ₑ", "ₗ", "ₘ", "ₙ", "ₚ", "ₛ", "ₜ", "ℂ", "ℕ", "ℚ", "ℝ", "ℤ", "ℱ", "←", "↔", "↦", "↪", "⇑", "∀", "∂", "∃", "∅", "∈", "∉", "∏", "∑", "∘", "∞", "∣", "∧", "∨", "∩", "∪", "∫", "≃", "≅", "≠", "≡", "≤", "≥", "≪", "≫", "⊆", "⊓", "⊔", "⊕", "⊗", "⊢", "⊤", "⊥", "⋂", "⋃", "⋆", "⋙", "▷", "▸", "◁", "⟦", "⟧", "⟨", "⟩", "⟪", "⟫", "⟶", "⥤", "⦃", "⦄", "⧸", "⨅", "⨆", "𝒜", "𝒰", "𝓘", "𝓝", "𝔖", "𝕜", "𝟙",
    # these are left out because they are already in GPT2 tokenizer (although weirdly not reported in tok_show): "¬", "¹"
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I haven't validated that this is actually a good idea, TODO.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Generic GPT-4-style tokenizer based on HuggingFace Tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

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

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # train from an iterator of text
        # Configure the HuggingFace Tokenizer
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True, # needed!
            unk_token=None,
            fuse_unk=False,
        ))
        # Normalizer: None
        tokenizer.normalizer = None
        # Pre-tokenizer: GPT-4 style
        # the regex pattern used by GPT-4 to split text into groups before BPE
        # NOTE: The pattern was changed from \p{N}{1,3} to \p{N}{1,2} because I suspect it is harmful to
        # very small models and smaller vocab sizes, because it is a little bit wasteful in the token space.
        # (but I haven't validated this! TODO)
        gpt4_split_regex = Regex(SPLIT_PATTERN) # huggingface demands that you wrap it in Regex!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None
        tokenizer.post_processor = None
        # Trainer: BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # no minimum frequency
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        # Kick off the training
        tokenizer.train_from_iterator(text_iterator, trainer)
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

# TODO: use special tokens!
def value_to_token_ids(tokenizer, value: int) -> list[int]:
    assert value >= _MIN_VALUE
    value = min(value, _MAX_VALUE)
    return tokenizer.encode(str(value))
    
def token_ids_to_value(tokenizer, token_ids: list[int]) -> float | None:
    try:
        return int(tokenizer.decode(token_ids))
    except ValueError:
        return None

def get_tokenizer():
    # return HuggingFaceTokenizer.from_pretrained("gpt2")
    from nanoproof.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    return HuggingFaceTokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    import torch
    from nanoproof.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes