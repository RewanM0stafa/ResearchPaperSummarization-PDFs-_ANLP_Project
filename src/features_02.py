import os
os.environ["HF_HOME"] = r"D:\hf_cache"

from datasets import load_from_disk
from transformers import AutoTokenizer

print("Loading processed dataset...")

# Load the cleaned data we saved in Phase 1
# No need to re-download or re-clean
train_data = load_from_disk("../data/processed/train")
test_data  = load_from_disk("../data/processed/test")

print(f"Train: {len(train_data)} examples")
print(f"Test:  {len(test_data)} examples")

# ─────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────

# Load BART tokenizer
# BART is a Seq2Seq model designed for summarization
# The tokenizer converts text → numbers (token IDs) that the model understands
# Example: "deep learning" → [1996, 4083]
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

# Quick test to understand what the tokenizer does
sample_text = "Deep learning is a subset of machine learning."
tokens = tokenizer(sample_text)
print("\n--- Tokenizer test ---")
print(f"Original text: {sample_text}")
print(f"Token IDs:     {tokens['input_ids']}")
print(f"Number of tokens: {len(tokens['input_ids'])}")

# ─────────────────────────────────────────
# TOKENIZATION SETTINGS
# ─────────────────────────────────────────

# Max lengths control how much text the model sees
# Longer = more info but uses more memory and time
# 512 for articles is a known limit for BART
MAX_INPUT_LENGTH  = 512   # article (input)
MAX_TARGET_LENGTH = 128   # abstract (output/label)

def tokenize_sample(sample):
    # Tokenize the article (model input)
    # truncation=True  → cut if longer than MAX_INPUT_LENGTH
    # padding="max_length" → pad with zeros if shorter (all inputs must be same length)
    model_inputs = tokenizer(
        sample["document"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )

    # Tokenize the abstract (target label)
    # The model will try to generate this during training
    labels = tokenizer(
        sample["summary"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length"
    )

    # Attach labels to the inputs dictionary
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("\nTokenizing dataset...")

# batched=True → process multiple examples at once (much faster)
train_tokenized = train_data.map(tokenize_sample, batched=True)
test_tokenized  = test_data.map(tokenize_sample,  batched=True)

print("Tokenization done!")

# ─────────────────────────────────────────
# VERIFY
# ─────────────────────────────────────────

print("\n--- Verification ---")
print(f"input_ids length:      {len(train_tokenized[0]['input_ids'])}")
print(f"attention_mask length: {len(train_tokenized[0]['attention_mask'])}")
print(f"labels length:         {len(train_tokenized[0]['labels'])}")

# ─────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────

os.makedirs("../data/processed", exist_ok=True)
train_tokenized.save_to_disk("../data/processed/train_tokenized")
test_tokenized.save_to_disk("../data/processed/test_tokenized")

print("\nTokenized data saved to data/processed/")
print("Phase 2 complete!")