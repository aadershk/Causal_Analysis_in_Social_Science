import os
import csv
import json
import re
from random import sample
from typing import List, Dict, Any

from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)

class BaseCausalIndicatorEvaluator:
    """
    Base class providing:
      - Pipeline loading for token classification
      - Common token labeling and unification logic
    """
    MODEL_NAME = "tanfiona/unicausal-tok-baseline"
    DEVICE_ID = 0
    CAUSAL_INDICATORS = ["because", "therefore", "thus", "since", "hence"]
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
        preds = pipe(text)
        char_array = ["O"] * len(text)

        for pr in preds:
            label_grp = pr["entity_group"]
            mapped_label = self.LABEL_MAPPING.get(label_grp, "O")
            start, end = pr["start"], pr["end"]
            if not (0 <= start < end <= len(text)):
                continue

            if mapped_label in ("B-C", "I-C"):
                for idx in range(start, end):
                    char_array[idx] = "C"
            elif mapped_label in ("B-E", "I-E"):
                for idx in range(start, end):
                    char_array[idx] = "E"

        final_labels = []
        pointer = 0
        for token in tokens:
            segment = char_array[pointer:pointer + len(token)]
            if all(l == "O" for l in segment):
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

    def unify_label(self, lab: str) -> str:
        if lab in ("B-C", "I-C"):
            return "C"
        if lab in ("B-E", "I-E"):
            return "E"
        return "O"


class ArgBasedCausalIndicatorEvaluator(BaseCausalIndicatorEvaluator):
    """
    Evaluator for a CSV dataset with potential causal indicators
    (ARG-based approach).
    """
    DATA_FILE = "data (annotated)/general_data.csv"

    def read_data_file(self) -> List[dict]:
        if not os.path.exists(self.DATA_FILE):
            raise FileNotFoundError(f"No CSV found at: {self.DATA_FILE}")

        entries = []
        with open(self.DATA_FILE, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_text = row.get("text", "")
                paired_text = row.get("text_w_pairs", "")
                if not raw_text or not paired_text:
                    continue
                if not any(cue in raw_text.lower() for cue in self.CAUSAL_INDICATORS):
                    continue
                entries.append({"text": raw_text, "text_w_pairs": paired_text})
        return entries

    def balance_data(self, data: List[dict], max_size: int = 100) -> List[dict]:
        return sample(data, min(len(data), max_size))

    def build_gold_token_labels(self, text: str, text_w_pairs: str):
        raw_tokens = text.split()
        pair_tokens = text_w_pairs.split()
        gold_labels = []
        current_label = "O"

        def strip_tags(tok: str) -> str:
            tok = re.sub(r"</?ARG[01]>", "", tok)
            tok = re.sub(r"</?SIG\d*>", "", tok)
            return tok.strip()

        cleaned_tokens = []
        for ptok in pair_tokens:
            label_for_tok = current_label
            if "<ARG0>" in ptok:
                label_for_tok = "B-C"
                current_label = "I-C"
            elif "</ARG0>" in ptok and current_label in ("I-C", "B-C"):
                label_for_tok = "I-C"
                current_label = "O"
            elif "<ARG1>" in ptok:
                label_for_tok = "B-E"
                current_label = "I-E"
            elif "</ARG1>" in ptok and current_label in ("I-E", "B-E"):
                label_for_tok = "I-E"
                current_label = "O"

            cleaned = strip_tags(ptok)
            if cleaned:
                cleaned_tokens.append(cleaned)
                gold_labels.append(label_for_tok)

        aligned_len = min(len(raw_tokens), len(cleaned_tokens))
        return raw_tokens[:aligned_len], gold_labels[:aligned_len]


class SocialScienceCausalIndicatorEvaluator(BaseCausalIndicatorEvaluator):
    """
    Evaluator for a JSONL-based social science dataset where
    text might contain implicit or explicit causal indicators.
    """
    JSON_FILES = [
        "data (annotated)/social_data_1.jsonl",
        "data (annotated)/social_data_2.jsonl"
    ]

    def read_json_files(self) -> List[Dict[str, Any]]:
        all_entries = []
        for fpath in self.JSON_FILES:
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"JSON file not found: {fpath}")

            with open(fpath, "r", encoding="utf-8") as fin:
                for line in fin:
                    obj = json.loads(line.strip())
                    text = obj.get("text", "")
                    if not any(ci in text.lower() for ci in self.CAUSAL_INDICATORS):
                        continue
                    entities = obj.get("entities", [])
                    c_e = [e for e in entities if e["label"].lower() in ["cause", "effect"]]
                    if not c_e:
                        continue
                    all_entries.append({"id": obj["id"], "text": text, "entities": c_e})
        return all_entries

    def balance_data(self, data: List[Dict[str, Any]], max_size: int = 100) -> List[Dict[str, Any]]:
        return sample(data, min(len(data), max_size))

    def build_gold_labels(self, text: str, entities: List[Dict[str, Any]]):
        tokens = text.split()
        joined = " ".join(tokens)
        char_positions = []
        idx = 0
        for token in tokens:
            cstart, cend = idx, idx + len(token)
            char_positions.append((cstart, cend))
            idx = cend + 1

        cause_spans, effect_spans = [], []
        for ent in entities:
            label = ent["label"].lower()
            st, en = ent["start_offset"], ent["end_offset"]
            if label == "cause":
                cause_spans.append((st, en))
            elif label == "effect":
                effect_spans.append((st, en))

        labels = ["O"] * len(tokens)

        def label_interval(spans, b_label, i_label):
            in_span = False
            for (start, end) in spans:
                for i, (cstart, cend) in enumerate(char_positions):
                    if cend > start and cstart < end:
                        if not in_span:
                            labels[i] = b_label
                            in_span = True
                        else:
                            if labels[i] in ("O", b_label):
                                labels[i] = i_label
                    elif cstart >= end:
                        in_span = False

        label_interval(cause_spans, "B-C", "I-C")
        label_interval(effect_spans, "B-E", "I-E")
        return tokens, labels


def main():
    print("\n===== ARG-BASED EVALUATION =====")
    arg_evaluator = ArgBasedCausalIndicatorEvaluator()
    arg_pipe = arg_evaluator.load_pipeline()
    arg_data = arg_evaluator.read_data_file()
    arg_data = arg_evaluator.balance_data(arg_data)

    gold_seq_arg, pred_seq_arg = [], []
    for row in arg_data:
        tokens, gold = arg_evaluator.build_gold_token_labels(row["text"], row["text_w_pairs"])
        if not tokens:
            continue
        preds = arg_evaluator.predict_labels_on_tokens(tokens, arg_pipe)
        length = min(len(gold), len(preds))
        gold_seq_arg.append([arg_evaluator.unify_label(g) for g in gold[:length]])
        pred_seq_arg.append([arg_evaluator.unify_label(p) for p in preds[:length]])

    print(f"Precision: {precision_score(gold_seq_arg, pred_seq_arg):.4f}")
    print(f"Recall:    {recall_score(gold_seq_arg, pred_seq_arg):.4f}")
    print(f"F1 Score:  {f1_score(gold_seq_arg, pred_seq_arg):.4f}")
    print(f"Accuracy:  {accuracy_score(gold_seq_arg, pred_seq_arg):.4f}")

    print("\n===== SOCIAL SCIENCE EVALUATION =====")
    social_eval = SocialScienceCausalIndicatorEvaluator()
    social_pipe = social_eval.load_pipeline()
    social_data = social_eval.read_json_files()
    social_data = social_eval.balance_data(social_data)

    gold_seq_soc, pred_seq_soc = [], []
    for entry in social_data:
        tokens, gold = social_eval.build_gold_labels(entry["text"], entry["entities"])
        if not tokens:
            continue
        preds = social_eval.predict_labels_on_tokens(tokens, social_pipe)
        length = min(len(gold), len(preds))
        gold_seq_soc.append([social_eval.unify_label(g) for g in gold[:length]])
        pred_seq_soc.append([social_eval.unify_label(p) for p in preds[:length]])

    print(f"Precision: {precision_score(gold_seq_soc, pred_seq_soc):.4f}")
    print(f"Recall:    {recall_score(gold_seq_soc, pred_seq_soc):.4f}")
    print(f"F1 Score:  {f1_score(gold_seq_soc, pred_seq_soc):.4f}")
    print(f"Accuracy:  {accuracy_score(gold_seq_soc, pred_seq_soc):.4f}")


if __name__ == "__main__":
    main()
