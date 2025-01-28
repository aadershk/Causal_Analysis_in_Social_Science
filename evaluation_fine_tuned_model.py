import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import List, Dict

###############################################################################
# 1) PATHS & HYPERPARAMETERS
###############################################################################
TEST_DATA_FILE = "data (annotated)/social_data_final.jsonl"  # Path to the test dataset
MODEL_PATH = "fine-tuned-model/unicausal_finetuned"  # Fine-tuned model path
BATCH_SIZE = 8
MAX_SEQ_LENGTH = 256

###############################################################################
# 2) LABEL MAPPINGS
###############################################################################
LABEL_LIST = ["O", "B-C", "I-C", "B-E", "I-E"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

###############################################################################
# 3) CUSTOM DATASET CLASS
###############################################################################
class TestDataset(Dataset):
    def __init__(self, data_file: str, tokenizer, max_length: int = MAX_SEQ_LENGTH):
        self.tokenizer = tokenizer
        self.data = []

        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                tokens, labels = record["tokens"], record["labels"]

                # Tokenize and align labels
                encoding = tokenizer(
                    tokens,
                    is_split_into_words=True,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt"
                )

                input_ids = encoding["input_ids"][0]
                attention_mask = encoding["attention_mask"][0]
                word_ids = encoding.word_ids()

                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)  # Special tokens
                    else:
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
# 4) EVALUATION FUNCTION
###############################################################################
def evaluate_model():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    model.eval()

    # Load test dataset
    test_dataset = TestDataset(TEST_DATA_FILE, tokenizer)
    print(f"[INFO] Loaded {len(test_dataset)} examples for evaluation.")

    # Prepare DataLoader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    all_preds, all_labels = [], []

    # Iterate over test data
    for batch in test_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Get predictions
        predictions = torch.argmax(logits, dim=-1)

        # Align predictions and labels
        for pred, true_labels in zip(predictions, labels):
            filtered_preds = [p.item() for p, t in zip(pred, true_labels) if t != -100]
            filtered_labels = [t.item() for t in true_labels if t != -100]

            all_preds.extend(filtered_preds)
            all_labels.extend(filtered_labels)

    # Convert IDs to labels
    all_preds_labels = [ID2LABEL[p] for p in all_preds]
    all_labels_labels = [ID2LABEL[l] for l in all_labels]

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")

    print("[RESULTS] Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

###############################################################################
# 5) MAIN FUNCTION
###############################################################################
if __name__ == "__main__":
    metrics = evaluate_model()
    print("[DONE] Evaluation completed.")
