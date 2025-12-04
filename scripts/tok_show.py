import sys

from tqdm import tqdm

from nanoproof.tokenizer import get_tokenizer, HuggingFaceTokenizer
from nanoproof.data.leangithubraw import iter_texts_batched

# Random text I got from a random website this morning
news_text = r"""
(Washington, D.C., July 9, 2025)- Yesterday, Mexico’s National Service of Agro-Alimentary Health, Safety, and Quality (SENASICA) reported a new case of New World Screwworm (NWS) in Ixhuatlan de Madero, Veracruz in Mexico, which is approximately 160 miles northward of the current sterile fly dispersal grid, on the eastern side of the country and 370 miles south of the U.S./Mexico border. This new northward detection comes approximately two months after northern detections were reported in Oaxaca and Veracruz, less than 700 miles away from the U.S. border, which triggered the closure of our ports to Mexican cattle, bison, and horses on May 11, 2025.

While USDA announced a risk-based phased port re-opening strategy for cattle, bison, and equine from Mexico beginning as early as July 7, 2025, this newly reported NWS case raises significant concern about the previously reported information shared by Mexican officials and severely compromises the outlined port reopening schedule of five ports from July 7-September 15. Therefore, in order to protect American livestock and our nation’s food supply, Secretary Rollins has ordered the closure of livestock trade through southern ports of entry effective immediately.

“The United States has promised to be vigilant — and after detecting this new NWS case, we are pausing the planned port reopening’s to further quarantine and target this deadly pest in Mexico. We must see additional progress combatting NWS in Veracruz and other nearby Mexican states in order to reopen livestock ports along the Southern border,” said U.S. Secretary of Agriculture Brooke L. Rollins. “Thanks to the aggressive monitoring by USDA staff in the U.S. and in Mexico, we have been able to take quick and decisive action to respond to the spread of this deadly pest.”
""".strip()

lean_search_text = r"""
x : ℝ
⊢ x ^ 2 - 2 * x - 24 < 0 ↔ x ∈ Set.Ioo (-4) 6
	
exact ⟨fun h ↦ by rw [Set.mem_Ioo]; constructor <;> nlinarith [h], fun h ↦ by rw [Set.mem_Ioo] at h; nlinarith⟩

⊢ ∀ (x : ℝ), 2⁻¹ + cos (2 * (2 * x)) / 2 = (1 + cos (4 * x)) / 2
	
ring

case h
ι : Type u_4
inst✝ : Fintype ι
f : ℝ → ι → ℝ
s : Set ℝ
h : LocallyBoundedVariationOn f s
A : ∀ (i : ι), LipschitzWith 1 fun x => x i
i : ι
⊢ LocallyBoundedVariationOn (fun x => f x i) s

exact LipschitzWith.comp_locallyBoundedVariationOn (A i) h

p q : Prop
⊢ p ∧ q → p

intro h

case mp.inl
p q r : Prop
hp : p
hq : q
⊢ p ∧ q ∨ p ∧ r

exact Or.inl ⟨hp, hq⟩

α : Type
P : α → Prop
inst✝ : Inhabited α
h : ∀ (x : α), P x
x0 : α := default
hx0 : P x0
⊢ ∃ x, P x

exact Exists.intro x0 hx0

¬test
"""

if len(sys.argv) != 2:
    print("Usage: python tok_show.py <tokenizer_name>")
    sys.exit(1)
tokenizer_name = sys.argv[1]
if tokenizer_name == "gpt2":
    tokenizer = HuggingFaceTokenizer.from_pretrained("gpt2")
elif tokenizer_name == "ours":
    tokenizer = get_tokenizer()
else:
    raise ValueError(f"Unknown tokenizer: {tokenizer_name}")

print(f"Vocab size: {tokenizer.get_vocab_size():,}")

for text in [("news", news_text), ("lean", lean_search_text)]:
    name, text = text
    encoded = tokenizer.encode(text)
    tokens = [tokenizer.id_to_token(id) for id in encoded]
    print(f"{name}:")
    print(' '.join(tokens))
    print()


print("Gathering character frequencies from leangithubraw train...")
char_counts = {}
# iter_texts_batched yields lists of strings (texts)
# We'll iterate through both train and val splits to be thorough
for batch_texts in iter_texts_batched(split="train", url_whitelist=["https://github.com/leanprover-community/mathlib4"]):
    for text in batch_texts:
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1

# Filter to only characters that appear at least 10 times
frequent_chars = {char for char, count in char_counts.items() if count >= 1000}
print(f"Found {len(frequent_chars)} characters that appear at least 1000 times.")

print("Checking for characters without dedicated tokens...")
chars_without_token = []
for char in tqdm(frequent_chars):
    # Encode the single character
    ids = tokenizer.encode(char)
    
    # If it takes more than 1 token to represent the character, 
    # it doesn't have a single dedicated token.
    if len(ids) != 1:
        chars_without_token.append(char)
        continue
    assert tokenizer.decode(ids) == char, f"decode({ids}) = \"{tokenizer.decode(ids)}\" != \"{char}\" (U+{ord(char):04X})"

chars_without_token.sort()
print(f"\nFound {len(chars_without_token)} characters without a dedicated token:")

print("[" + ", ".join(f"\"{c}\"" for c in chars_without_token) + "]")