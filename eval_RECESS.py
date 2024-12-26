import os
import csv
import re
from typing import List
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

##############################################################################
# ARG-BASED NO-SIGNAL EVALUATOR (CLASS-BASED)
##############################################################################

class ArgBasedNoSignalEvaluator:
    """
    This class handles:
      1) Reading an ARG-based CSV file that includes <ARG0>/<ARG1> tags for cause/effect.
      2) Skipping any lines with NO <ARG0> or <ARG1> (i.e., non-causal).
      3) Ignoring signals (e.g., <SIG0>), focusing only on cause/effect.
      4) Generating token-level B-C/I-C or B-E/I-E labels.
      5) Using 'tanfiona/unicausal-tok-baseline' to predict cause/effect tokens.
      6) Computing and displaying precision, recall, and F1.
    """

    ############################################################################
    # CONFIG
    ############################################################################
    MODEL_NAME = "tanfiona/unicausal-tok-baseline"

    GENERAL_DATA_FILE = r"C:\Users\aader\PycharmProjects\Causal_Analysis_in_Social_Science\data (annotated)\general_data.csv"

    DEVICE_ID = 0

    LABEL_MAPPING = {
        "LABEL_0": "B-C",
        "LABEL_1": "B-E",
        "LABEL_2": "I-C",
        "LABEL_3": "I-E",
    }

    ############################################################################
    # STEP A: INITIALIZATION
    ############################################################################
    def __init__(self):

        pass

    ############################################################################
    # STEP B: LOAD MODEL PIPELINE
    ############################################################################
    def load_pipeline(self):

        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        model = AutoModelForTokenClassification.from_pretrained(self.MODEL_NAME)
        pipe = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=self.DEVICE_ID
        )
        return pipe

    ############################################################################
    # STEP C: READ LOCAL CSV
    ############################################################################
    def read_data_file(self) -> List[dict]:

        if not os.path.exists(self.GENERAL_DATA_FILE):
            raise FileNotFoundError(f"No CSV found at: {self.GENERAL_DATA_FILE}")

        data = []
        with open(self.GENERAL_DATA_FILE, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if not fieldnames:
                raise ValueError("No columns found in general_data.csv. Possibly empty?")

            required_cols = {"text", "text_w_pairs"}
            missing = required_cols - set(fieldnames)
            if missing:
                raise KeyError(
                    f"Missing columns {missing} in general_data.csv. Found columns: {fieldnames}"
                )

            for row in reader:
                raw_text = row["text"]
                text_w_pairs = row["text_w_pairs"]
                if not raw_text or not text_w_pairs:
                    continue
                # skip non-causal (no <ARG0> or <ARG1>)
                if ("<ARG0>" not in text_w_pairs) and ("<ARG1>" not in text_w_pairs):
                    continue

                data.append({
                    "text": raw_text,
                    "text_w_pairs": text_w_pairs
                })
        return data

    ############################################################################
    # STEP D: BUILD GOLD TOKEN LABELS (IGNORE SIGNALS)
    ############################################################################
    def build_gold_token_labels(self, text: str, text_w_pairs: str):

        raw_tokens = text.split()
        pair_tokens = text_w_pairs.split()

        gold_labels = []
        label_state = "O"

        def strip_tags(tok: str) -> str:
            tok_clean = re.sub(r"</?ARG[01]>", "", tok)
            tok_clean = re.sub(r"</?SIG\d*>", "", tok_clean)
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

    ############################################################################
    # STEP E: PREDICT MODEL TOKEN LABELS
    ############################################################################
    def predict_token_labels(self, pipe, tokens: List[str]) -> List[str]:

        model_text = " ".join(tokens)
        preds = pipe(model_text)
        char_labels = ["O"] * len(model_text)

        for pr in preds:
            grp = pr["entity_group"]
            mapped = self.LABEL_MAPPING.get(grp, "O")
            s_i, e_i = pr["start"], pr["end"]
            if s_i < 0 or e_i > len(model_text):
                continue
            if mapped in ("B-C", "I-C"):
                for idx in range(s_i, e_i):
                    char_labels[idx] = "C"
            elif mapped in ("B-E", "I-E"):
                for idx in range(s_i, e_i):
                    char_labels[idx] = "E"

        pred_labels = []
        char_ptr = 0
        for t in tokens:
            seg = char_labels[char_ptr : char_ptr + len(t)]
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

    ############################################################################
    # STEP F: UNIFY & EVAL
    ############################################################################
    def unify_label(self, lab: str) -> str:

        if lab in ("B-C","I-C"):
            return "C"
        elif lab in ("B-E","I-E"):
            return "E"
        else:
            return "O"

    def evaluate(self):

        pipe = self.load_pipeline()
        dataset = self.read_data_file()

        gold_seq = []
        pred_seq = []

        for row in dataset:
            raw_text = row["text"]
            twp = row["text_w_pairs"]
            tokens, gold_labels = self.build_gold_token_labels(raw_text, twp)
            if not tokens:
                continue

            pred_labels = self.predict_token_labels(pipe, tokens)
            L = min(len(gold_labels), len(pred_labels))
            if L == 0:
                continue
            gold_labels = gold_labels[:L]
            pred_labels = pred_labels[:L]

            gold_seq.append([self.unify_label(g) for g in gold_labels])
            pred_seq.append([self.unify_label(p) for p in pred_labels])

        prec = precision_score(gold_seq, pred_seq, suffix=False)
        rec = recall_score(gold_seq, pred_seq, suffix=False)
        f1 = f1_score(gold_seq, pred_seq, suffix=False)

        print("\n===== ARG-BASED TOKEN EVAL (NO SIGNALS, SKIP NON-CAUSAL) [CLASS EDITION] =====")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}\n")

        print("[DETAIL REPORT]")
        print(classification_report(gold_seq, pred_seq, suffix=False))


##############################################################################
# MAIN
##############################################################################
if __name__ == "__main__":
    evaluator = ArgBasedNoSignalEvaluator()
    evaluator.evaluate()
