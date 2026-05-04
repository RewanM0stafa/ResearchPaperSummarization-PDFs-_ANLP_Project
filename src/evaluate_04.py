import os
os.environ["HF_HOME"] = r"D:\hf_cache"

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration
)
import evaluate
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# ─────────────────────────────────────────
# LOAD MODEL & DATA
# ─────────────────────────────────────────

print("Loading model and data...")

MODEL_PATH = "../outputs/model/checkpoint-200"
tokenizer  = AutoTokenizer.from_pretrained("facebook/bart-base")
model      = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
model.eval()

test_data = load_from_disk("../data/processed/test")

# ─────────────────────────────────────────
# GENERATE SUMMARIES
# ─────────────────────────────────────────

print("Generating summaries...")

def generate_summary(text):
    # Tokenize the input article
    inputs = tokenizer(
        text,
        max_length=512,
        truncation=True,
        return_tensors="pt"   # return PyTorch tensors
    )

    # Generate the summary
    # max_new_tokens → maximum length of the generated summary
    # num_beams → beam search: explore multiple paths and pick the best
    #             higher = better quality but slower
    # early_stopping → stop when all beams reach the end token
    summary_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=128,
        num_beams=4,
        early_stopping=True
    )

    # Decode token IDs back to text
    # skip_special_tokens → remove tokens like <pad>, <eos>
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Generate summaries for all test examples
predictions = []
references  = []

for example in test_data:
    pred = generate_summary(example["document"])
    predictions.append(pred)
    references.append(example["summary"])

# ─────────────────────────────────────────
# ROUGE SCORE
# ─────────────────────────────────────────

# ROUGE measures how much the generated summary overlaps with the reference
# ROUGE-1 → overlap of single words
# ROUGE-2 → overlap of word pairs
# ROUGE-L → longest common subsequence
print("\nCalculating ROUGE scores...")
rouge = evaluate.load("rouge")
results = rouge.compute(predictions=predictions, references=references)

print("\n--- ROUGE Scores ---")
for key, value in results.items():
    print(f"{key}: {value:.4f}")

# Save scores to file
os.makedirs("../outputs/reports", exist_ok=True)
with open("../outputs/reports/rouge_scores.json", "w") as f:
    json.dump(results, f, indent=2)

# ─────────────────────────────────────────
# PLOT TRAINING LOSS
# ─────────────────────────────────────────

print("\nPlotting training loss...")

# Load training logs saved during training
log_file = "../outputs/model/trainer_state.json"

if os.path.exists(log_file):
    with open(log_file) as f:
        trainer_state = json.load(f)

    # Extract loss values from logs
    steps  = [x["step"] for x in trainer_state["log_history"] if "loss" in x]
    losses = [x["loss"] for x in trainer_state["log_history"] if "loss" in x]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, label="Training Loss", color="blue")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/plots/loss_curve.png")
    print("Loss curve saved to outputs/plots/loss_curve.png")

# ─────────────────────────────────────────
# SAMPLE PREDICTIONS
# ─────────────────────────────────────────

print("\n--- Sample Predictions ---")
for i in range(min(3, len(predictions))):
    print(f"\nExample {i+1}:")
    print(f"Reference:  {references[i][:200]}")
    print(f"Generated:  {predictions[i][:200]}")

print("\nEvaluation complete!")
print("Results saved to outputs/reports/")