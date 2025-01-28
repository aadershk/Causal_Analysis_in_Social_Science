import os
import json
import random
import logging
from collections import Counter
import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
from typing import List, Dict

###############################################################################
# 1) CONFIGURATION AND HYPERPARAMETERS
###############################################################################
DATA_FILE = "annotation_folder/annotated_output_final.jsonl"  # Input dataset
MODEL_NAME = "tanfiona/unicausal-tok-baseline"  # Base model
OUTPUT_DIR = "fine-tuned-model/unicausal_finetuned"  # Save fine-tuned model
MAX_TRAIN_SAMPLES = None  # Set a limit on training samples if needed
BATCH_SIZE = 8  # Batch size
NUM_EPOCHS = 5  # Increased epochs for better training
LEARNING_RATE = 3e-5  # Adjusted learning rate
SEED = 42
DEBUG = True  # Set to True for detailed logging

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

###############################################################################
# 2) SET RANDOM SEEDS FOR REPRODUCIBILITY
###############################################################################
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

###############################################################################
# 3) LABEL MAPPINGS (must match the base model)
###############################################################################
LABEL_LIST = ["O", "B-C", "I-C", "B-E", "I-E"]  # BIO tagging
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


###############################################################################
# 4) CUSTOM DATASET CLASS
###############################################################################
class UnicausalDataset(Dataset):
    def __init__(self, data_file: str, tokenizer, max_samples: int = None):
        self.tokenizer = tokenizer
        self.data = []

        with open(data_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if max_samples:
            lines = lines[:max_samples]

        for line in lines:
            record = json.loads(line.strip())
            tokens, labels = record["tokens"], record["labels"]

            # Tokenize the input
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

            # Align labels with word_ids
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    # Assign -100 for special tokens
                    label_ids.append(-100)
                else:
                    # Ensure word_idx does not exceed the labels list
                    label_ids.append(LABEL2ID[labels[word_idx]] if word_idx < len(labels) else -100)

            self.data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": torch.tensor(label_ids, dtype=torch.long)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


###############################################################################
# 5) DATASET INSPECTION
###############################################################################
def inspect_dataset(data_file):
    label_counter = Counter()
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            labels = record["labels"]
            label_counter.update(labels)

    logging.info("[INFO] Label Distribution:")
    for label, count in label_counter.items():
        logging.info(f"{label}: {count}")

    logging.info("[INFO] Dataset Inspection Completed.")


###############################################################################
# 6) SPLIT DATASET INTO TRAIN/VAL
###############################################################################
def split_dataset(dataset, train_ratio=0.8):
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])


###############################################################################
# 7) FINE-TUNING FUNCTION WITH VALIDATION AND DEBUGGING
###############################################################################
def fine_tune_model():
    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        label2id=LABEL2ID,
        id2label=ID2LABEL
    )

    # Load dataset
    logging.info("[INFO] Loading dataset...")
    dataset = UnicausalDataset(DATA_FILE, tokenizer, max_samples=MAX_TRAIN_SAMPLES)
    logging.info(f"[INFO] Loaded {len(dataset)} examples.")

    # Inspect dataset
    inspect_dataset(DATA_FILE)

    # Split dataset
    train_dataset, val_dataset = split_dataset(dataset)
    logging.info(f"[INFO] Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        fp16=True if torch.cuda.is_available() else False,
        seed=SEED
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    # Train the model
    logging.info("[INFO] Starting training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    logging.info(f"[INFO] Fine-tuned model saved to {OUTPUT_DIR}.")

    # Evaluate on training data
    logging.info("[INFO] Evaluating on training data...")
    results = trainer.evaluate(train_dataset)
    logging.info(f"[INFO] Training Data Evaluation Metrics: {results}")


###############################################################################
# 8) MAIN FUNCTION
###############################################################################
if __name__ == "__main__":
    fine_tune_model()
    logging.info("[DONE] Fine-tuning completed.")
"""

import os
import json
import random
import logging
from collections import Counter
import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from typing import List, Dict

###############################################################################
# 1) CONFIGURATION AND HYPERPARAMETERS
###############################################################################
DATA_FILE = "annotation_folder/annotated_output_final.jsonl"  # Input dataset
MODEL_NAME = "tanfiona/unicausal-tok-baseline"  # Base model
OUTPUT_DIR = "fine-tuned-model/unicausal_finetuned"  # Save fine-tuned model
MAX_TRAIN_SAMPLES = None  # Limit training samples if needed
BATCH_SIZE = 16  # Stable batch size for gradient updates
NUM_EPOCHS = 4  # Number of epochs for training
LEARNING_RATE = 5e-5  # Fine-tuning learning rate
WEIGHT_DECAY = 0.01  # Regularization parameter
WARMUP_STEPS = 500  # Warmup steps for learning rate scheduler
SEED = 42
DEBUG = True  # Enable detailed logging

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

###############################################################################
# 2) SET RANDOM SEEDS FOR REPRODUCIBILITY
###############################################################################
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

###############################################################################
# 3) LABEL MAPPINGS
###############################################################################
LABEL_LIST = ["O", "B-C", "I-C", "B-E", "I-E"]  # BIO tagging
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

###############################################################################
# 4) CUSTOM DATASET CLASS
###############################################################################
class UnicausalDataset(Dataset):
    def __init__(self, data_file: str, tokenizer, max_samples: int = None):
        self.tokenizer = tokenizer
        self.data = []

        with open(data_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if max_samples:
            lines = lines[:max_samples]

        for line in lines:
            record = json.loads(line.strip())
            tokens, labels = record["tokens"], record["labels"]

            # Tokenize the input
            encoding = tokenizer.encode_plus(
                tokens,
                is_split_into_words=True,
                add_special_tokens=True,
                truncation=True,
                max_length=256,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"][0]
            attention_mask = encoding["attention_mask"][0]
            word_ids = encoding.word_ids()

            # Align labels with word_ids
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # Ignore special tokens
                else:
                    label_ids.append(
                        LABEL2ID[labels[word_idx]] if word_idx < len(labels) else -100
                    )

            self.data.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": torch.tensor(label_ids, dtype=torch.long),
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

###############################################################################
# 5) DATASET INSPECTION
###############################################################################
def inspect_dataset(data_file):
    label_counter = Counter()
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            labels = record["labels"]
            label_counter.update(labels)

    logging.info("[INFO] Label Distribution:")
    for label, count in label_counter.items():
        logging.info(f"{label}: {count}")

    logging.info("[INFO] Dataset Inspection Completed.")

###############################################################################
# 6) SPLIT DATASET INTO TRAIN/VAL
###############################################################################
def split_dataset(dataset, train_ratio=0.8):
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])

###############################################################################
# 7) CUSTOM METRICS FUNCTION
###############################################################################
def compute_metrics(p: EvalPrediction):
    predictions = p.predictions.argmax(-1)
    labels = p.label_ids
    predictions = predictions[labels != -100]
    labels = labels[labels != -100]

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    accuracy = accuracy_score(labels, predictions)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

###############################################################################
# 8) FINE-TUNING FUNCTION
###############################################################################
def fine_tune_model():
    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    # Load dataset
    logging.info("[INFO] Loading dataset...")
    dataset = UnicausalDataset(DATA_FILE, tokenizer, max_samples=MAX_TRAIN_SAMPLES)
    logging.info(f"[INFO] Loaded {len(dataset)} examples.")

    # Inspect dataset
    inspect_dataset(DATA_FILE)

    # Split dataset
    train_dataset, val_dataset = split_dataset(dataset)
    logging.info(f"[INFO] Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Training arguments
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
        fp16=True if torch.cuda.is_available() else False,
        seed=SEED,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    logging.info("[INFO] Starting training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    logging.info(f"[INFO] Fine-tuned model saved to {OUTPUT_DIR}.")

    # Evaluate on training data
    logging.info("[INFO] Evaluating on training data...")
    results = trainer.evaluate(train_dataset)
    logging.info(f"[INFO] Training Data Evaluation Metrics: {results}")

###############################################################################
# 9) MAIN FUNCTION
###############################################################################
if __name__ == "__main__":
    fine_tune_model()
    logging.info("[DONE] Fine-tuning completed.")
"""