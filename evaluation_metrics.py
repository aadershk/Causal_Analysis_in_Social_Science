import os
import csv
import json
import re
import numpy as np
from typing import List, Dict, Any

from seqeval.metrics import (
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.metrics import confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)

class BaseEvaluator:
    """
    Shared functionalities for token classification pipeline,
    label prediction, and label unification.
    """
    MODEL_NAME = "tanfiona/unicausal-tok-baseline"
    DEVICE_ID = 0
    LABEL_MAPPING = {
        "LABEL_0": "B-C",
        "LABEL_1": "B-E",
        "LABEL_2": "I-C",
        "LABEL_3": "I-E",
    }

    def load_pipeline(self):
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        model = AutoModelForTokenClassification.from_pretrained(self.MODEL_NAME)
        return pipeline(
            task="token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=self.DEVICE_ID
        )

    def predict_labels_on_tokens(self, tokens: List[str], pipe) -> List[str]:
        text = " ".join(tokens)
        predictions = pipe(text)
        char_labels = ["O"] * len(text)

        for pred in predictions:
            group = pred["entity_group"]
            mapped_label = self.LABEL_MAPPING.get(group, "O")
            start, end = pred["start"], pred["end"]
            if not (0 <= start < end <= len(text)):
                continue
            if mapped_label in ("B-C", "I-C"):
                for idx in range(start, end):
                    char_labels[idx] = "C"
            elif mapped_label in ("B-E", "I-E"):
                for idx in range(start, end):
                    char_labels[idx] = "E"

        final_labels = []
        pointer = 0
        for token in tokens:
            segment = char_labels[pointer:pointer + len(token)]
            if all(lbl == "O" for lbl in segment):
                final_labels.append("O")
            else:
                if "C" in segment:
                    final_labels.append("B-C" if segment[0] == "C" else "I-C")
                elif "E" in segment:
                    final_labels.append("B-E" if segment[0] == "E" else "I-E")
                else:
                    final_labels.append("O")
            pointer += len(token) + 1

        return final_labels

    def unify_label(self, label: str) -> str:
        if label in ("B-C", "I-C"):
            return "C"
        if label in ("B-E", "I-E"):
            return "E"
        return "O"


class ArgBasedEvaluator(BaseEvaluator):
    """
    Evaluator for ARG-based CSV dataset.
    """
    DATA_FILE = (
        "data (annotated)/general_data.csv"
    )

    def read_data_file(self) -> List[Dict[str, str]]:
        if not os.path.exists(self.DATA_FILE):
            raise FileNotFoundError(f"No CSV found at: {self.DATA_FILE}")

        entries = []
        with open(self.DATA_FILE, "r", encoding="utf-8-sig") as fin:
            reader = csv.DictReader(fin)
            fieldnames = reader.fieldnames or []
            required = {"text", "text_w_pairs"}
            missing = required - set(fieldnames)
            if missing:
                raise KeyError(f"Missing columns: {missing}")

            for row in reader:
                raw_text = row["text"]
                paired_text = row["text_w_pairs"]
                if not raw_text or not paired_text:
                    continue
                if ("<ARG0>" not in paired_text) and ("<ARG1>" not in paired_text):
                    continue
                entries.append({"text": raw_text, "text_w_pairs": paired_text})
        return entries

    def build_gold_token_labels(self, text: str, text_w_pairs: str):
        raw_tokens = text.split()
        pair_tokens = text_w_pairs.split()
        gold_labels = []
        current_label = "O"

        def strip_tags(token: str) -> str:
            token = re.sub(r"</?ARG[01]>", "", token)
            token = re.sub(r"</?SIG\d*>", "", token)
            return token.strip()

        cleaned_tokens = []
        for ptok in pair_tokens:
            lbl = current_label
            if "<ARG0>" in ptok:
                lbl = "B-C"
                current_label = "I-C"
            elif "</ARG0>" in ptok and current_label in ("I-C", "B-C"):
                lbl = "I-C"
                current_label = "O"
            elif "<ARG1>" in ptok:
                lbl = "B-E"
                current_label = "I-E"
            elif "</ARG1>" in ptok and current_label in ("I-E", "B-E"):
                lbl = "I-E"
                current_label = "O"

            stripped = strip_tags(ptok)
            if stripped:
                cleaned_tokens.append(stripped)
                gold_labels.append(lbl)

        limit = min(len(raw_tokens), len(cleaned_tokens))
        return raw_tokens[:limit], gold_labels[:limit]


class SocialScienceEvaluator(BaseEvaluator):
    """
    Evaluator for Social Science JSONL dataset.
    """
    JSON_FILES = [
        "data (annotated)/social_data_1.jsonl",
        "data (annotated)/social_data_2.jsonl"
    ]

    def read_json_files(self) -> List[Dict[str, Any]]:
        all_data = []
        for path in self.JSON_FILES:
            if not os.path.exists(path):
                raise FileNotFoundError(f"JSON file not found: {path}")
            with open(path, "r", encoding="utf-8") as fin:
                for line in fin:
                    obj = json.loads(line.strip())
                    text = obj.get("text", "")
                    entities = obj.get("entities", [])
                    cause_effect = [e for e in entities if e["label"].lower() in ("cause", "effect")]
                    if not cause_effect:
                        continue
                    all_data.append({"id": obj["id"], "text": text, "entities": cause_effect})
        return all_data

    def build_gold_labels(self, text: str, entities: List[Dict[str, Any]]):
        tokens = text.split()
        char_map = []
        idx = 0
        for tok in tokens:
            start_i, end_i = idx, idx + len(tok)
            char_map.append((start_i, end_i))
            idx = end_i + 1

        c_spans, e_spans = [], []
        for ent in entities:
            lb = ent["label"].lower()
            st, en = ent["start_offset"], ent["end_offset"]
            if lb == "cause":
                c_spans.append((st, en))
            elif lb == "effect":
                e_spans.append((st, en))

        labels = ["O"] * len(tokens)

        def label_spans(span_list, btag, itag):
            in_span = False
            for start, end in span_list:
                for i, (cstart, cend) in enumerate(char_map):
                    if cend > start and cstart < end:
                        if not in_span:
                            labels[i] = btag
                            in_span = True
                        else:
                            if labels[i] in ("O", btag):
                                labels[i] = itag
                    elif cstart >= end:
                        in_span = False

        label_spans(c_spans, "B-C", "I-C")
        label_spans(e_spans, "B-E", "I-E")
        return tokens, labels


def compute_confusion_matrix(gold: List[List[str]], pred: List[List[str]]):
    flattened_gold = [g for seq in gold for g in seq]
    flattened_pred = [p for seq in pred for p in seq]
    return confusion_matrix(flattened_gold, flattened_pred, labels=["O", "C", "E"])


def main():
    print("\n===== ARG-BASED EVALUATION =====")
    arg_eval = ArgBasedEvaluator()
    arg_pipe = arg_eval.load_pipeline()
    arg_data = arg_eval.read_data_file()

    arg_gold, arg_pred = [], []
    for row in arg_data:
        tokens, gold_labels = arg_eval.build_gold_token_labels(row["text"], row["text_w_pairs"])
        if not tokens:
            continue
        pred_labels = arg_eval.predict_labels_on_tokens(tokens, arg_pipe)
        limit = min(len(gold_labels), len(pred_labels))
        arg_gold.append([arg_eval.unify_label(g) for g in gold_labels[:limit]])
        arg_pred.append([arg_eval.unify_label(p) for p in pred_labels[:limit]])

    print(f"Accuracy:  {accuracy_score(arg_gold, arg_pred):.4f}")
    print(f"Precision: {precision_score(arg_gold, arg_pred):.4f}")
    print(f"Recall:    {recall_score(arg_gold, arg_pred):.4f}")
    print(f"F1 Score:  {f1_score(arg_gold, arg_pred):.4f}")

    arg_matrix = compute_confusion_matrix(arg_gold, arg_pred)
    print("Confusion Matrix (ARG-Based):")
    print(arg_matrix)

    print("\n===== SOCIAL SCIENCE EVALUATION =====")
    soc_eval = SocialScienceEvaluator()
    soc_pipe = soc_eval.load_pipeline()
    soc_data = soc_eval.read_json_files()

    soc_gold, soc_pred = [], []
    for entry in soc_data:
        tokens, gold_labels = soc_eval.build_gold_labels(entry["text"], entry["entities"])
        if not tokens:
            continue
        pred_labels = soc_eval.predict_labels_on_tokens(tokens, soc_pipe)
        limit = min(len(gold_labels), len(pred_labels))
        soc_gold.append([soc_eval.unify_label(g) for g in gold_labels[:limit]])
        soc_pred.append([soc_eval.unify_label(p) for p in pred_labels[:limit]])

    print(f"Accuracy:  {accuracy_score(soc_gold, soc_pred):.4f}")
    print(f"Precision: {precision_score(soc_gold, soc_pred):.4f}")
    print(f"Recall:    {recall_score(soc_gold, soc_pred):.4f}")
    print(f"F1 Score:  {f1_score(soc_gold, soc_pred):.4f}")

    soc_matrix = compute_confusion_matrix(soc_gold, soc_pred)
    print("Confusion Matrix (Social Science):")
    print(soc_matrix)


if __name__ == "__main__":
    main()
