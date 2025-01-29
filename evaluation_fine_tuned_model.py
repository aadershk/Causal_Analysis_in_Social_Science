import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import List, Dict

# Global settings
TEST_DATA_FILE = "data (annotated)/social_data_final.jsonl"
MODEL_PATH = "fine-tuned-model/unicausal_finetuned"
BATCH_SIZE = 8
MAX_SEQ_LENGTH = 256

LABEL_LIST = ["O", "B-C", "I-C", "B-E", "I-E"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

class SocialScienceTestDataset(Dataset):
    """
    Dataset for social science sentences with cause-effect labels.
    Each sample is pre-tokenized and aligned with label IDs.
    """
    def __init__(self, data_file: str, tokenizer, max_length: int = MAX_SEQ_LENGTH) -> None:
        self.samples = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                tokens = record["tokens"]
                raw_labels = record["labels"]

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
                for w_id in word_ids:
                    if w_id is None:
                        label_ids.append(-100)
                    else:
                        label_ids.append(LABEL2ID[raw_labels[w_id]] if w_id < len(raw_labels) else -100)

                self.samples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": torch.tensor(label_ids, dtype=torch.long)
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


class CausalModelEvaluator:
    """
    Evaluates a fine-tuned causal extraction model on a social science dataset.
    """
    def __init__(self, model_path: str = MODEL_PATH, test_file: str = TEST_DATA_FILE) -> None:
        self.model_path = model_path
        self.test_file = test_file
        self.model = None
        self.tokenizer = None

    def load_model_and_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.model.eval()

    def evaluate(self) -> Dict[str, float]:
        if not self.model or not self.tokenizer:
            self.load_model_and_tokenizer()

        dataset = SocialScienceTestDataset(self.test_file, self.tokenizer, MAX_SEQ_LENGTH)
        print(f"[INFO] Loaded {len(dataset)} examples for evaluation.")

        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        all_preds, all_labels = [], []

        for batch in data_loader:
            with torch.no_grad():
                outputs = self.model(batch["input_ids"], attention_mask=batch["attention_mask"])
                logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            for pred_seq, true_seq in zip(preds, batch["labels"]):
                filtered_preds = [p.item() for p, t in zip(pred_seq, true_seq) if t != -100]
                filtered_labels = [t.item() for t in true_seq if t != -100]
                all_preds.extend(filtered_preds)
                all_labels.extend(filtered_labels)

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
            "f1": f1
        }


def main():
    evaluator = CausalModelEvaluator()
    evaluator.evaluate()
    print("[DONE] Evaluation completed.")


if __name__ == "__main__":
    main()
