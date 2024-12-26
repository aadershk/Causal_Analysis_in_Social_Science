import os
import json
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

##############################################################################
# SOCIAL SCIENCE NO-SIGNAL EVALUATOR (CLASS-BASED)
##############################################################################

class SocialScienceNoSignalEvaluator:
    """
    This class handles:
      1) Reading two JSONL files
      2) Skipping lines that have no cause/effect entities at all.
      3) Ignoring signals/polarities entirely.
      4) Generating B-C/I-C or B-E/I-E token labels from 'start_offset','end_offset'.
      5) Using 'tanfiona/unicausal-tok-baseline' to predict cause/effect tokens.
      6) Computing final P/R/F1 for domain bias analysis.
    """

    MODEL_NAME = "tanfiona/unicausal-tok-baseline"
    DEVICE_ID = 0

    JSON_FILES = [
        "/data (annotated)/social_data_1.jsonl",
        "/data (annotated)/social_data_1.jsonl",
    ]

    LABEL_MAPPING = {
        "LABEL_0": "B-C",
        "LABEL_1": "B-E",
        "LABEL_2": "I-C",
        "LABEL_3": "I-E",
    }

    ############################################################################
    # STEP A: INIT
    ############################################################################
    def __init__(self):
        """
        Constructs the evaluator, set file paths if needed.
        """
        pass

    ############################################################################
    # STEP B: LOAD MODEL PIPELINE
    ############################################################################
    def load_pipeline(self):
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

    ############################################################################
    # STEP C: READ JSON FILES, SKIP NON-CAUSAL
    ############################################################################
    def read_json_files(self) -> List[Dict[str,Any]]:
        """
        Merges lines from self.JSON_FILES, skipping lines that have no cause/effect.
        Returns a list of dicts:
          { "id":..., "text":"...", "entities":[ cause/effect only ] }
        """
        data = []
        for fpath in self.JSON_FILES:
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"JSON file not found: {fpath}")

            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = obj["text"]
                    entities = obj.get("entities", [])
                    # filter cause/effect only
                    ce_entities = [e for e in entities if e["label"].lower() in ["cause","effect"]]
                    if not ce_entities:
                        # skip non-causal
                        continue

                    data.append({
                        "id": obj["id"],
                        "text": text,
                        "entities": ce_entities
                    })
        return data

    ############################################################################
    # STEP D: BUILD GOLD TOKEN LABELS
    ############################################################################
    def build_gold_labels(self, text: str, entities: List[Dict[str,Any]]):
        """
        Splits text into tokens (whitespace).
        Entities have { "label":"cause"/"effect", "start_offset":..., "end_offset":... }.

        Output:
          tokens: e.g. ["There","is","a",...]
          labels: e.g. ["O","O","O","B-C","I-C",...]
        """
        raw_tokens = text.split()
        joined_text = " ".join(raw_tokens)
        # track each token's (start,end) in joined_text
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
                cause_intervals.append((st,en))
            elif lab == "effect":
                effect_intervals.append((st,en))

        labels = ["O"]*len(raw_tokens)

        def label_span(span_list, b_label, i_label):
            in_span = False
            for (st,en) in span_list:
                for t_i,(c_start,c_end) in enumerate(char_positions):
                    if c_end>st and c_start<en:
                        if not in_span:
                            labels[t_i] = b_label
                            in_span=True
                        else:
                            if labels[t_i]=="O" or labels[t_i].startswith("B-"):
                                labels[t_i] = i_label
                    else:
                        if c_start>=en:
                            in_span=False

        label_span(cause_intervals, "B-C","I-C")
        label_span(effect_intervals, "B-E","I-E")

        return raw_tokens, labels

    ############################################################################
    # STEP E: PREDICT LABELS
    ############################################################################
    def predict_labels_on_tokens(self, tokens: List[str], pipe) -> List[str]:
        model_text = " ".join(tokens)
        preds = pipe(model_text)
        char_labels = ["O"] * len(model_text)

        for pr in preds:
            grp = pr["entity_group"]
            mapped_label = self.LABEL_MAPPING.get(grp, "O")
            s_i, e_i = pr["start"], pr["end"]
            if s_i<0 or e_i>len(model_text):
                continue
            if mapped_label in ("B-C","I-C"):
                for c_idx in range(s_i,e_i):
                    char_labels[c_idx]="C"
            elif mapped_label in ("B-E","I-E"):
                for c_idx in range(s_i,e_i):
                    char_labels[c_idx]="E"

        pred_labels=[]
        char_ptr=0
        for t in tokens:
            seg=char_labels[char_ptr: char_ptr+len(t)]
            if all(x=="O" for x in seg):
                pred_labels.append("O")
            else:
                if "C" in seg:
                    if seg[0]=="C":
                        pred_labels.append("B-C")
                    else:
                        pred_labels.append("I-C")
                elif "E" in seg:
                    if seg[0]=="E":
                        pred_labels.append("B-E")
                    else:
                        pred_labels.append("I-E")
                else:
                    pred_labels.append("O")
            char_ptr+=len(t)+1

        return pred_labels

    def unify_label(self, lab:str)->str:
        if lab in ("B-C","I-C"):
            return "C"
        elif lab in ("B-E","I-E"):
            return "E"
        else:
            return "O"

    ############################################################################
    # STEP F: EVALUATE
    ############################################################################
    def evaluate(self):
        from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

        pipe = self.load_pipeline()
        data = self.read_json_files()

        gold_seq=[]
        pred_seq=[]

        for entry in data:
            text = entry["text"]
            ce_entities = entry["entities"]
            tokens, gold_labels = self.build_gold_labels(text, ce_entities)
            if not tokens:
                continue
            pred_labels = self.predict_labels_on_tokens(tokens, pipe)
            L = min(len(gold_labels), len(pred_labels))
            if L==0:
                continue
            gold_labels=gold_labels[:L]
            pred_labels=pred_labels[:L]

            gold_seq.append([self.unify_label(g) for g in gold_labels])
            pred_seq.append([self.unify_label(p) for p in pred_labels])

        prec=precision_score(gold_seq, pred_seq, suffix=False)
        rec=recall_score(gold_seq, pred_seq, suffix=False)
        f1=f1_score(gold_seq, pred_seq, suffix=False)

        print("\n===== SOCIAL SCIENCE TOKEN-LEVEL EVAL (NO SIGNALS, SKIP NON-CAUSAL) [CLASS EDITION] =====")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}\n")

        print("[DETAIL REPORT]")
        print(classification_report(gold_seq, pred_seq, suffix=False))


##############################################################################
# MAIN (OPTIONAL)
##############################################################################
if __name__=="__main__":
    evaluator = SocialScienceNoSignalEvaluator()
    evaluator.evaluate()
