import os
import csv
import json
import re
from typing import List, Dict, Any
from random import sample
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

##############################################################################
# BASE CLASS FOR EVALUATORS
##############################################################################

class BaseCausalIndicatorEvaluator:
    """
    Base class to handle common functionalities for both ARG-based and Social Science evaluators:
      - Model pipeline loading.
      - Token label prediction.
      - Label unification.
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
        """
        Load the model pipeline for token classification.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        model = AutoModelForTokenClassification.from_pretrained(self.MODEL_NAME)
        pipe = pipeline(
            task="token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=self.DEVICE_ID
        )
        return pipe

    def predict_labels_on_tokens(self, tokens: List[str], pipe) -> List[str]:
        """
        Predict token labels using the model pipeline.
        """
        model_text = " ".join(tokens)
        preds = pipe(model_text)
        char_labels = ["O"] * len(model_text)

        for pr in preds:
            grp = pr["entity_group"]
            mapped_label = self.LABEL_MAPPING.get(grp, "O")
            s_i, e_i = pr["start"], pr["end"]
            if s_i < 0 or e_i > len(model_text):
                continue
            if mapped_label in ("B-C", "I-C"):
                for c_idx in range(s_i, e_i):
                    char_labels[c_idx] = "C"
            elif mapped_label in ("B-E", "I-E"):
                for c_idx in range(s_i, e_i):
                    char_labels[c_idx] = "E"

        pred_labels = []
        char_ptr = 0
        for t in tokens:
            seg = char_labels[char_ptr: char_ptr + len(t)]
            if all(x == "O" for x in seg):
                pred_labels.append("O")
            else:
                if "C" in seg:
                    if seg[0] == "C":
                        pred_labels.append("B-C")
                    else:
                        pred_labels.append("I-C")
                elif "E" in seg:
                    if seg[0] == "E":
                        pred_labels.append("B-E")
                    else:
                        pred_labels.append("I-E")
                else:
                    pred_labels.append("O")
            char_ptr += len(t) + 1

        return pred_labels

    def unify_label(self, lab: str) -> str:
        """
        Simplify token labels to "C", "E", or "O".
        """
        if lab in ("B-C", "I-C"):
            return "C"
        elif lab in ("B-E", "I-E"):
            return "E"
        else:
            return "O"

##############################################################################
# ARG-BASED CAUSAL INDICATOR EVALUATOR
##############################################################################

class ArgBasedCausalIndicatorEvaluator(BaseCausalIndicatorEvaluator):
    """
    Evaluator for ARG-based CSV dataset with causal indicators.
    """
    GENERAL_DATA_FILE = r"C:\Users\aader\PycharmProjects\Causal_Analysis_in_Social_Science\data (annotated)\general_data.csv"

    def read_data_file(self) -> List[dict]:
        if not os.path.exists(self.GENERAL_DATA_FILE):
            raise FileNotFoundError(f"No CSV found at: {self.GENERAL_DATA_FILE}")

        data = []
        with open(self.GENERAL_DATA_FILE, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_text = row["text"]
                text_w_pairs = row["text_w_pairs"]
                if not raw_text or not text_w_pairs:
                    continue
                if not any(ind in raw_text.lower() for ind in self.CAUSAL_INDICATORS):
                    continue
                data.append({
                    "text": raw_text,
                    "text_w_pairs": text_w_pairs
                })
        return data

    def balance_data(self, data: List[dict]) -> List[dict]:
        return sample(data, min(len(data), 100))

    def build_gold_token_labels(self, text: str, text_w_pairs: str):
        raw_tokens = text.split()
        pair_tokens = text_w_pairs.split()

        gold_labels = []
        label_state = "O"

        def strip_tags(tok: str) -> str:
            tok_clean = re.sub(r"</?ARG[01]>", "", tok)
            tok_clean = re.sub(r"</?SIG\d*>", "", tok)
            return tok_clean.strip()

        cleaned_tokens = []
        for p_tok in pair_tokens:
            local_label = label_state

            if "<ARG0>" in p_tok:
                local_label = "B-C"
                label_state = "I-C"
            elif "</ARG0>" in p_tok:
                if label_state in ("I-C", "B-C"):
                    local_label = "I-C"
                label_state = "O"
            elif "<ARG1>" in p_tok:
                local_label = "B-E"
                label_state = "I-E"
            elif "</ARG1>" in p_tok:
                if label_state in ("I-E", "B-E"):
                    local_label = "I-E"
                label_state = "O"

            cleaned_tok = strip_tags(p_tok)
            if not cleaned_tok:
                continue
            cleaned_tokens.append(cleaned_tok)
            gold_labels.append(local_label)

        aligned_len = min(len(raw_tokens), len(cleaned_tokens))
        tokens = raw_tokens[:aligned_len]
        labels = gold_labels[:aligned_len]
        return tokens, labels

##############################################################################
# SOCIAL SCIENCE CAUSAL INDICATOR EVALUATOR
##############################################################################

class SocialScienceCausalIndicatorEvaluator(BaseCausalIndicatorEvaluator):
    """
    Evaluator for Social Science JSONL dataset with causal indicators.
    """
    JSON_FILES = [
        r"C:\Users\aader\PycharmProjects\Causal_Analysis_in_Social_Science\data (annotated)\social_data_1.jsonl",
        r"C:\Users\aader\PycharmProjects\Causal_Analysis_in_Social_Science\data (annotated)\social_data_2.jsonl",
    ]

    def read_json_files(self) -> List[Dict[str, Any]]:
        data = []
        for fpath in self.JSON_FILES:
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"JSON file not found: {fpath}")

            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line.strip())
                    text = obj["text"]
                    entities = obj.get("entities", [])
                    if not any(ind in text.lower() for ind in self.CAUSAL_INDICATORS):
                        continue
                    ce_entities = [e for e in entities if e["label"].lower() in ["cause", "effect"]]
                    if not ce_entities:
                        continue
                    data.append({"id": obj["id"], "text": text, "entities": ce_entities})
        return data

    def balance_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sample(data, min(len(data), 100))

    def build_gold_labels(self, text: str, entities: List[Dict[str, Any]]):
        raw_tokens = text.split()
        joined_text = " ".join(raw_tokens)
        char_positions = []
        idx = 0
        for rt in raw_tokens:
            start_i = idx
            end_i = idx + len(rt)
            char_positions.append((start_i, end_i))
            idx = end_i + 1

        cause_intervals = []
        effect_intervals = []
        for e in entities:
            lab = e["label"].lower()
            st = e["start_offset"]
            en = e["end_offset"]
            if lab == "cause":
                cause_intervals.append((st, en))
            elif lab == "effect":
                effect_intervals.append((st, en))

        labels = ["O"] * len(raw_tokens)

        def label_span(span_list, b_label, i_label):
            in_span = False
            for (st, en) in span_list:
                for t_i, (c_start, c_end) in enumerate(char_positions):
                    if c_end > st and c_start < en:
                        if not in_span:
                            labels[t_i] = b_label
                            in_span = True
                        else:
                            if labels[t_i] == "O" or labels[t_i].startswith("B-"):
                                labels[t_i] = i_label
                    else:
                        if c_start >= en:
                            in_span = False

        label_span(cause_intervals, "B-C", "I-C")
        label_span(effect_intervals, "B-E", "I-E")

        return raw_tokens, labels

##############################################################################
# MAIN
##############################################################################

def main():
    print("\n===== ARG-BASED EVALUATION =====")
    arg_evaluator = ArgBasedCausalIndicatorEvaluator()
    arg_pipe = arg_evaluator.load_pipeline()
    arg_data = arg_evaluator.read_data_file()
    balanced_arg_data = arg_evaluator.balance_data(arg_data)

    arg_gold_seq = []
    arg_pred_seq = []

    for row in balanced_arg_data:
        tokens, gold_labels = arg_evaluator.build_gold_token_labels(row["text"], row["text_w_pairs"])
        if not tokens:
            continue
        pred_labels = arg_evaluator.predict_labels_on_tokens(tokens, arg_pipe)
        L = min(len(gold_labels), len(pred_labels))
        arg_gold_seq.append([arg_evaluator.unify_label(g) for g in gold_labels[:L]])
        arg_pred_seq.append([arg_evaluator.unify_label(p) for p in pred_labels[:L]])

    print(f"Precision: {precision_score(arg_gold_seq, arg_pred_seq):.4f}")
    print(f"Recall:    {recall_score(arg_gold_seq, arg_pred_seq):.4f}")
    print(f"F1 Score:  {f1_score(arg_gold_seq, arg_pred_seq):.4f}")
    print(f"Accuracy:  {accuracy_score(arg_gold_seq, arg_pred_seq):.4f}")

    print("\n===== SOCIAL SCIENCE EVALUATION =====")
    social_evaluator = SocialScienceCausalIndicatorEvaluator()
    social_pipe = social_evaluator.load_pipeline()
    social_data = social_evaluator.read_json_files()
    balanced_social_data = social_evaluator.balance_data(social_data)

    social_gold_seq = []
    social_pred_seq = []

    for entry in balanced_social_data:
        tokens, gold_labels = social_evaluator.build_gold_labels(entry["text"], entry["entities"])
        if not tokens:
            continue
        pred_labels = social_evaluator.predict_labels_on_tokens(tokens, social_pipe)
        L = min(len(gold_labels), len(pred_labels))
        social_gold_seq.append([social_evaluator.unify_label(g) for g in gold_labels[:L]])
        social_pred_seq.append([social_evaluator.unify_label(p) for p in pred_labels[:L]])

    print(f"Precision: {precision_score(social_gold_seq, social_pred_seq):.4f}")
    print(f"Recall:    {recall_score(social_gold_seq, social_pred_seq):.4f}")
    print(f"F1 Score:  {f1_score(social_gold_seq, social_pred_seq):.4f}")
    print(f"Accuracy:  {accuracy_score(social_gold_seq, social_pred_seq):.4f}")


if __name__ == "__main__":
    main()
