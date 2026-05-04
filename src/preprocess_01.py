import os
os.environ["HF_HOME"] = r"D:\hf_cache"
from datasets import load_dataset
import numpy as np
import re


print("Loading dataset...")

# load_dataset → downloads the dataset from HuggingFace automatically
# "xsum" → BBC news articles with human-written summaries
# split="train[:500]" → take only 500 examples (manageable size)
dataset = load_dataset("xsum", split="train[:2000]")

print(f"Total examples loaded: {len(dataset)}")
print("\n--- Dataset structure ---")
print(dataset)

# Each example has two fields:
# "document" → the full news article (model input)
# "summary"  → the one-sentence summary (target label)
print("\n--- First example preview ---")
print("Document (input):")
print(dataset[0]["document"][:300])
print("\nSummary (target):")
print(dataset[0]["summary"][:200])

# Measure lengths BEFORE any cleaning
# Required for report: "shape of dataset before any changes"
print("\n--- Length statistics (before cleaning) ---")
doc_lengths = [len(x["document"].split()) for x in dataset]
sum_lengths = [len(x["summary"].split())  for x in dataset]

print(f"Avg document length: {np.mean(doc_lengths):.0f} words")
print(f"Avg summary length:  {np.mean(sum_lengths):.0f} words")
print(f"Max document length: {max(doc_lengths)} words")
print(f"Min document length: {min(doc_lengths)} words")

# ─────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────

def clean_text(text):
    # Remove reference numbers like [1], [23]
    text = re.sub(r'\[\d+\]', '', text)

    # Remove LaTeX math expressions like $x^2$
    text = re.sub(r'\$.*?\$', '', text)

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Collapse multiple spaces/newlines into one space
    text = re.sub(r'\s+', ' ', text)

    # Remove leading and trailing spaces
    text = text.strip()

    return text

def preprocess_sample(sample):
    return {
        "document": clean_text(sample["document"]),
        "summary":  clean_text(sample["summary"])
    }

print("\nCleaning dataset...")
dataset = dataset.map(preprocess_sample)
print("Cleaning done!")

# Measure lengths AFTER cleaning
print("\n--- Length statistics (after cleaning) ---")
doc_lengths_clean = [len(x["document"].split()) for x in dataset]
sum_lengths_clean = [len(x["summary"].split())  for x in dataset]

print(f"Avg document length: {np.mean(doc_lengths_clean):.0f} words")
print(f"Avg summary length:  {np.mean(sum_lengths_clean):.0f} words")

# ─────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────

# 80% train, 20% test
# seed=42 → same split every time
split = dataset.train_test_split(test_size=0.2, seed=42)
train_data = split["train"]
test_data  = split["test"]

print(f"\nTraining examples: {len(train_data)}")
print(f"Testing examples:  {len(test_data)}")

# ─────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────

os.makedirs("../data/processed", exist_ok=True)
train_data.save_to_disk("../data/processed/train")
test_data.save_to_disk("../data/processed/test")

print("\nDataset saved to data/processed/")
print("Phase 1 complete!")