"""
Build the nanoproof tokenizer: GPT-2 BPE + extra special tokens for Lean / math.
"""

import os

import torch

from nanoproof.common import get_base_dir
from nanoproof.tokenizer import HuggingFaceTokenizer, SPECIAL_TOKENS

tokenizer = HuggingFaceTokenizer.from_pretrained("gpt2")
tokenizer.tokenizer.add_special_tokens(SPECIAL_TOKENS)

base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# Sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# Cache token_id -> utf-8 byte-length so we can report bits-per-byte
# (vocab-size-invariant) on the validation set.
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_bytes = []
for token_id in range(vocab_size):
    token_str = tokenizer.decode([token_id])
    if token_str in special_set:
        token_bytes.append(0)
    else:
        token_bytes.append(len(token_str.encode("utf-8")))
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")
