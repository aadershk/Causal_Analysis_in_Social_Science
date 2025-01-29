import os
import json
import random
import logging
from collections import Counter
from typing import List, Dict

import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Configuration and hyperparameters
DATA_FILE = "annotation_folder/annotated_output_final.jsonl"
MODEL_NAME = "tanfiona/unicausal-tok-baseline"
OUTPUT_DIR = "fine-tuned-model/unicausal_finetuned"
MAX_TRAIN_SAMPLES = None
BATCH_SIZE = 16
NUM_EPOCHS = 4
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
SEED = 42
DEBUG = True

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

LABEL_LIST = ["O", "B-C", "I-C", "B-E", "I-E"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


class UnicausalDataset(Dataset):
    """Loads causal data (tokens and labels) and aligns them with tokenized input."""
    def __init__(self, data_file: str, tokenizer, max_samples: int = None):
        self.tokenizer = tokenizer
        self.data = []

        with open(data_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()

        if max_samples:
            lines = lines[:max_samples]

        for line in lines:
            record = json.loads(line.strip())
            tokens = record["tokens"]
            labels = record["labels"]

            encoding = tokenizer.encode_plus(
                tokens,
                is_split_into_words=True,
                add_special_tokens=True,
                truncation=True,
                max_length=256,
                padding="max_length",
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"][0]
            attention_mask = encoding["attention_mask"][0]
            word_ids = encoding.word_ids()

            label_ids = []
            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)
                else:
                    if wid < len(labels):
                        label_ids.append(LABEL2ID[labels[wid]])
                    else:
                        label_ids.append(-100)

            self.data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": torch.tensor(label_ids, dtype=torch.long)
            })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


def inspect_dataset(data_file: str) -> None:
    """Logs label distribution for the dataset."""
    label_count = Counter()
    with open(data_file, "r", encoding="utf-8") as fin:
        for line in fin:
            record = json.loads(line.strip())
            label_count.update(record["labels"])

    logging.info("[INFO] Label Distribution:")
    for label, cnt in label_count.items():
        logging.info(f"{label}: {cnt}")
    logging.info("[INFO] Dataset Inspection Completed.")


def split_dataset(dataset: Dataset, train_ratio: float = 0.8):
    size = len(dataset)
    train_size = int(size * train_ratio)
    val_size = size - train_size
    return random_split(dataset, [train_size, val_size])


def compute_metrics(pred: EvalPrediction) -> Dict[str, float]:
    predictions = pred.predictions.argmax(-1)
    labels = pred.label_ids
    valid_preds = predictions[labels != -100]
    valid_labels = labels[labels != -100]

    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_labels, valid_preds, average="weighted"
    )
    accuracy = accuracy_score(valid_labels, valid_preds)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def fine_tune_model() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        label2id=LABEL2ID,
        id2label=ID2LABEL
    )

    logging.info("[INFO] Loading dataset...")
    dataset = UnicausalDataset(DATA_FILE, tokenizer, max_samples=MAX_TRAIN_SAMPLES)
    logging.info(f"[INFO] Loaded {len(dataset)} examples.")

    inspect_dataset(DATA_FILE)
    train_set, val_set = split_dataset(dataset)
    logging.info(f"[INFO] Train size: {len(train_set)}, Val size: {len(val_set)}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        seed=SEED
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    logging.info("[INFO] Starting training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    logging.info(f"[INFO] Fine-tuned model saved to {OUTPUT_DIR}.")

    logging.info("[INFO] Evaluating on training data...")
    results = trainer.evaluate(train_set)
    logging.info(f"[INFO] Training Data Evaluation Metrics: {results}")


def main():
    fine_tune_model()
    logging.info("[DONE] Fine-tuning completed.")


if __name__ == "__main__":
    main()
