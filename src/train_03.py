import os
os.environ["HF_HOME"] = r"D:\hf_cache"

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import torch

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────

print("Loading tokenized dataset...")

# Load the tokenized data saved in Phase 2
# No need to re-tokenize every time
train_data = load_from_disk("../data/processed/train_tokenized")
test_data  = load_from_disk("../data/processed/test_tokenized")

print(f"Train: {len(train_data)} examples")
print(f"Test:  {len(test_data)} examples")

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────

print("\nLoading BART model...")

MODEL_NAME = "facebook/bart-base"

# BartForConditionalGeneration → BART version designed for text generation
# (summarization, translation, etc.)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

# CPU only → fp16 not supported on CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device)

# ─────────────────────────────────────────
# DATA COLLATOR
# ─────────────────────────────────────────

# Pads each batch to the longest example in that batch
# Better than padding everything to max_length → saves memory
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# ─────────────────────────────────────────
# TRAINING ARGUMENTS
# ─────────────────────────────────────────

training_args = Seq2SeqTrainingArguments(
    output_dir="../outputs/model",

    num_train_epochs=3,              # 1 epoch only → CPU is slow, 1 is enough to test
    per_device_train_batch_size=2,   # small batch → BART is large
    per_device_eval_batch_size=2,

    warmup_steps=50,                 # gradual learning rate increase at start
    weight_decay=0.01,               # regularization to prevent overfitting

    logging_dir="../outputs/plots",
    logging_steps=10,                # print loss every 10 steps

    eval_strategy="epoch",           # evaluate after each epoch
    save_strategy="epoch",           # save checkpoint after each epoch
    load_best_model_at_end=True,     # keep best checkpoint at the end

    predict_with_generate=True,      # required for summarization evaluation

    fp16=False,                      # False → CPU does not support fp16
)

# ─────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────

# Handles the full training loop automatically:
# forward pass → loss calculation → backward pass → weight update
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=data_collator,
)

# ─────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────

print("\nStarting training...")
trainer.train()

print("\nTraining complete!")
print("Model saved to outputs/model/")