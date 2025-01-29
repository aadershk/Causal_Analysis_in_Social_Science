import os
import csv
import json
import re
import random
import warnings
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict

import spacy
from tqdm import tqdm
from prefixspan import PrefixSpan
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)

warnings.filterwarnings("ignore", category=UserWarning)


def strip_tags(token: str) -> str:
    """Remove <ARGx> and <SIGx> tags from a token."""
    token = re.sub(r"</?ARG[01]>", "", token)
    token = re.sub(r"</?SIG\d*>", "", token)
    return token.strip()

def unify_label(label: str) -> str:
    """Convert B-C/I-C => 'C', B-E/I-E => 'E', else 'O'."""
    if label in ("B-C", "I-C"):
        return "C"
    if label in ("B-E", "I-E"):
        return "E"
    return "O"

def map_pipeline_label_to_BIO(label_group: str) -> str:
    """Map pipeline label groups to BIO-style tags."""
    mapping = {
        "LABEL_0": "B-C",
        "LABEL_1": "B-E",
        "LABEL_2": "I-C",
        "LABEL_3": "I-E"
    }
    return mapping.get(label_group, "O")

def build_char_label_array(text: str, predictions, label_mapping) -> List[str]:
    """Create a char-level label array; fill with 'C' or 'E' for cause/effect spans."""
    char_arr = ["O"] * len(text)
    for pr in predictions:
        group = pr["entity_group"]
        mapped_bio = label_mapping(group)
        start, end = pr["start"], pr["end"]
        if 0 <= start < end <= len(text):
            if mapped_bio in ("B-C", "I-C"):
                for idx in range(start, end):
                    char_arr[idx] = "C"
            elif mapped_bio in ("B-E", "I-E"):
                for idx in range(start, end):
                    char_arr[idx] = "E"
    return char_arr

def predict_labels_on_tokens(pipe, tokens: List[str]) -> List[str]:
    """Predict token-level cause/effect labels using Hugging Face pipeline."""
    text = " ".join(tokens)
    predictions = pipe(text)
    char_labels = build_char_label_array(text, predictions, map_pipeline_label_to_BIO)

    pred_bio = []
    pointer = 0
    for tok in tokens:
        seg = char_labels[pointer:pointer + len(tok)]
        if all(lbl == "O" for lbl in seg):
            pred_bio.append("O")
        else:
            if "C" in seg:
                pred_bio.append("B-C" if seg[0] == "C" else "I-C")
            elif "E" in seg:
                pred_bio.append("B-E" if seg[0] == "E" else "I-E")
            else:
                pred_bio.append("O")
        pointer += len(tok) + 1

    return [unify_label(lbl) for lbl in pred_bio]

def align_spacy_with_tokens(raw_text: str, token_list: List[str], nlp):
    """Align each token with spaCy's POS. If mismatch, fallback is 'X'."""
    doc = nlp(raw_text)
    pos_map = defaultdict(list)
    for token in doc:
        pos_map[token.text].append(token.pos_)

    aligned = []
    for tok in token_list:
        if pos_map[tok]:
            aligned.append((tok, pos_map[tok].pop(0)))
        else:
            aligned.append((tok, "X"))
    return aligned


class TokenLevelErrorAnalysis:
    """
    Analyzes token-level cause/effect errors, mines POS-based patterns,
    and stores tokens labeled as 'X' for manual review.
    """

    def __init__(
        self,
        general_csv_path: str,
        social_jsonl_paths: List[str],
        model_name: str = "tanfiona/unicausal-tok-baseline",
        output_dir: str = "analysis_outputs",
        random_seed: int = 42
    ):
        self.general_csv_path = general_csv_path
        self.social_jsonl_paths = social_jsonl_paths
        self.model_name = model_name
        self.output_dir = output_dir
        self.random_seed = random_seed

        random.seed(self.random_seed)
        self.all_data = []

        # Subset storages
        self.correct_tokens_general = []
        self.incorrect_tokens_general = []
        self.correct_tokens_social = []
        self.incorrect_tokens_social = []
        self.x_tokens = []

        self.nlp = None
        self.pipe = None
        self._init_spacy()
        self._init_pipeline()

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _init_spacy(self) -> None:
        print("[INFO] Loading spaCy (en_core_web_sm)...")
        self.nlp = spacy.load("en_core_web_sm")
        try:
            self.nlp.to_gpu()
            print("[INFO] spaCy on GPU.")
        except Exception:
            print("[INFO] spaCy on CPU.")

    def _init_pipeline(self) -> None:
        print(f"[INFO] Loading HF pipeline from model: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self.pipe = pipeline(
            task="token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=0  # or -1 for CPU
        )

    def load_general_data(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.general_csv_path):
            raise FileNotFoundError(f"General CSV not found: {self.general_csv_path}")

        data = []
        with open(self.general_csv_path, "r", encoding="utf-8-sig") as fin:
            reader = csv.DictReader(fin)
            fields = reader.fieldnames
            if not fields:
                raise ValueError("No columns in the CSV.")
            required = {"text", "text_w_pairs"}
            missing = required - set(fields)
            if missing:
                raise KeyError(f"Missing columns {missing} in {self.general_csv_path}.")

            for row in reader:
                raw_text = row["text"]
                pairs_text = row["text_w_pairs"]
                if not raw_text or not pairs_text:
                    continue
                if ("<ARG0>" not in pairs_text) and ("<ARG1>" not in pairs_text):
                    continue
                tokens, gold_bio = self._build_gold_labels_general(raw_text, pairs_text)
                unified = [unify_label(x) for x in gold_bio]
                data.append({
                    "domain": "general",
                    "text": raw_text,
                    "tokens": tokens,
                    "gold_labels": unified
                })
        return data

    def _build_gold_labels_general(self, raw_text: str, text_w_pairs: str) -> Tuple[List[str], List[str]]:
        raw_tokens = raw_text.split()
        pair_tokens = text_w_pairs.split()
        gold_labels = []
        label_state = "O"
        cleaned_tokens = []

        for p_tok in pair_tokens:
            local_label = label_state
            if "<ARG0>" in p_tok:
                local_label = "B-C"
                label_state = "I-C"
            elif "</ARG0>" in p_tok and label_state in ("I-C", "B-C"):
                local_label = "I-C"
                label_state = "O"
            elif "<ARG1>" in p_tok:
                local_label = "B-E"
                label_state = "I-E"
            elif "</ARG1>" in p_tok and label_state in ("I-E", "B-E"):
                local_label = "I-E"
                label_state = "O"

            cleaned = strip_tags(p_tok)
            if cleaned:
                cleaned_tokens.append(cleaned)
                gold_labels.append(local_label)

        limit = min(len(raw_tokens), len(cleaned_tokens))
        return raw_tokens[:limit], gold_labels[:limit]

    def load_social_data(self) -> List[Dict[str, Any]]:
        data = []
        for path in self.social_jsonl_paths:
            if not os.path.exists(path):
                print(f"[WARNING] File not found: {path}")
                continue
            with open(path, "r", encoding="utf-8") as fin:
                for line in fin:
                    obj = json.loads(line.strip())
                    text = obj["text"]
                    entities = obj.get("entities", [])
                    ce_ents = [e for e in entities if e["label"].lower() in ["cause", "effect"]]
                    if not ce_ents:
                        continue
                    tokens, gold_bio = self._build_gold_labels_social(text, ce_ents)
                    unified = [unify_label(x) for x in gold_bio]
                    data.append({
                        "domain": "social",
                        "text": text,
                        "tokens": tokens,
                        "gold_labels": unified
                    })
        return data

    def _build_gold_labels_social(self, text: str, entities: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        tokens = text.split()
        char_map = []
        idx = 0
        for tok in tokens:
            start_i, end_i = idx, idx + len(tok)
            char_map.append((start_i, end_i))
            idx = end_i + 1

        labels = ["O"] * len(tokens)
        cause_spans, effect_spans = [], []

        for ent in entities:
            lb = ent["label"].lower()
            st, en = ent["start_offset"], ent["end_offset"]
            if lb == "cause":
                cause_spans.append((st, en))
            elif lb == "effect":
                effect_spans.append((st, en))

        def label_span(span_list, b_tag, i_tag):
            in_span = False
            for st, en in span_list:
                for i, (cstart, cend) in enumerate(char_map):
                    if cend > st and cstart < en:
                        if not in_span:
                            labels[i] = b_tag
                            in_span = True
                        else:
                            if labels[i] in ("O", b_tag):
                                labels[i] = i_tag
                    elif cstart >= en:
                        in_span = False

        label_span(cause_spans, "B-C", "I-C")
        label_span(effect_spans, "B-E", "I-E")
        return tokens, labels

    def run_analysis(self, sample_size: int = 60) -> None:
        general_data = self.load_general_data()
        social_data = self.load_social_data()
        combined = general_data + social_data

        gen_items = [x for x in combined if x["domain"] == "general"]
        soc_items = [x for x in combined if x["domain"] == "social"]
        random.shuffle(gen_items)
        random.shuffle(soc_items)

        if len(gen_items) > sample_size:
            gen_items = gen_items[:sample_size]
        if len(soc_items) > sample_size:
            soc_items = soc_items[:sample_size]
        final_data = gen_items + soc_items
        random.shuffle(final_data)

        for item in tqdm(final_data, desc="Model inference"):
            pred_labels = predict_labels_on_tokens(self.pipe, item["tokens"])
            item["pred_labels"] = pred_labels

        gold_sequences, pred_sequences = [], []
        for item in final_data:
            gold_sequences.append(item["gold_labels"])
            pred_sequences.append(item["pred_labels"])

        prec = precision_score(gold_sequences, pred_sequences)
        rec = recall_score(gold_sequences, pred_sequences)
        f1v = f1_score(gold_sequences, pred_sequences)

        print("\n===== TOKEN-LEVEL EVALUATION =====")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1v:.4f}\n")
        print("[DETAIL REPORT]")
        print(classification_report(gold_sequences, pred_sequences))

        self.all_data = []
        for item in final_data:
            domain = item["domain"]
            text = item["text"]
            toks = item["tokens"]
            golds = item["gold_labels"]
            preds = item["pred_labels"]
            pos_aligned = align_spacy_with_tokens(text, toks, self.nlp)

            record = {
                "domain": domain,
                "text": text,
                "tokens": [],
                "pos_tags": [],
                "gold_labels": [],
                "pred_labels": [],
                "correct_flags": []
            }
            for i, token in enumerate(toks):
                g = golds[i]
                p = preds[i]
                correct_flag = (g == p)
                assigned_pos = pos_aligned[i][1] if i < len(pos_aligned) else "X"

                record["tokens"].append(token)
                record["pos_tags"].append(assigned_pos)
                record["gold_labels"].append(g)
                record["pred_labels"].append(p)
                record["correct_flags"].append(correct_flag)

            self.all_data.append(record)

        self.final_error_analysis()

    def final_error_analysis(self) -> None:
        """
        Splits data into correct/incorrect subsets for each domain,
        runs prefixspan on tokens & POS sequences, saves X-labeled tokens.
        """
        gc_tokpos, gi_tokpos = [], []
        sc_tokpos, si_tokpos = [], []
        gc_posonly, gi_posonly = [], []
        sc_posonly, si_posonly = [], []
        self.x_tokens = []

        for rec in self.all_data:
            domain = rec["domain"]
            tokens = rec["tokens"]
            pos_tags = rec["pos_tags"]
            correct_flags = rec["correct_flags"]
            golds = rec["gold_labels"]
            preds = rec["pred_labels"]

            correct_seq, incorrect_seq = [], []
            correct_pos_seq, incorrect_pos_seq = [], []

            for tok, pos, cflag in zip(tokens, pos_tags, correct_flags):
                if pos in ("SPACE", "PUNCT"):
                    continue
                if pos == "X":
                    self.x_tokens.append((tok, domain, cflag, rec["text"]))

                if cflag:
                    correct_seq.append((tok, pos))
                    correct_pos_seq.append(pos)
                else:
                    incorrect_seq.append((tok, pos))
                    incorrect_pos_seq.append(pos)

            if domain == "general":
                gc_tokpos.append(correct_seq)
                gi_tokpos.append(incorrect_seq)
                gc_posonly.append(correct_pos_seq)
                gi_posonly.append(incorrect_pos_seq)
            else:
                sc_tokpos.append(correct_seq)
                si_tokpos.append(incorrect_seq)
                sc_posonly.append(correct_pos_seq)
                si_posonly.append(incorrect_pos_seq)

        print("\n==================== TOKEN-LEVEL ERROR ANALYSIS ====================")

        def compute_pos_dist(list_of_seqs):
            cnt = Counter()
            for seq in list_of_seqs:
                for _, pos in seq:
                    cnt[pos] += 1
            return cnt

        gc_dist = compute_pos_dist(gc_tokpos)
        gi_dist = compute_pos_dist(gi_tokpos)
        sc_dist = compute_pos_dist(sc_tokpos)
        si_dist = compute_pos_dist(si_tokpos)

        def print_top_pos(label, dist):
            top5 = dist.most_common(5)
            print(f"{label} top 5 POS:")
            for pos_, freq_ in top5:
                print(f"  {pos_}: {freq_}")

        print_top_pos("General-Correct", gc_dist)
        print_top_pos("General-Incorrect", gi_dist)
        print_top_pos("Social-Correct", sc_dist)
        print_top_pos("Social-Incorrect", si_dist)

        def run_prefixspan_and_print(label, sequences, topk=10):
            ps = PrefixSpan(sequences)
            patterns = ps.topk(topk, closed=True)
            print(f"\n[TOP (token, pos) PATTERNS for {label}]")
            for freq, pat in patterns:
                print(f"  freq={freq}, pattern={pat}")

        run_prefixspan_and_print("General - Correct", gc_tokpos, 10)
        run_prefixspan_and_print("General - Incorrect", gi_tokpos, 10)
        run_prefixspan_and_print("Social - Correct", sc_tokpos, 10)
        run_prefixspan_and_print("Social - Incorrect", si_tokpos, 10)

        def run_prefixspan_pos_only(label, sequences, topk=10):
            ps = PrefixSpan(sequences)
            patterns = ps.topk(topk, closed=True)
            print(f"\n[TOP POS-ONLY PATTERNS for {label}]")
            for freq, pat in patterns:
                print(f"  freq={freq}, pattern={pat}")

        run_prefixspan_pos_only("General - Correct", gc_posonly, 10)
        run_prefixspan_pos_only("General - Incorrect", gi_posonly, 10)
        run_prefixspan_pos_only("Social - Correct", sc_posonly, 10)
        run_prefixspan_pos_only("Social - Incorrect", si_posonly, 10)

        x_outfile = os.path.join(self.output_dir, "x_tokens.txt")
        with open(x_outfile, "w", encoding="utf-8") as fout:
            for tok, domain, cflag, full_text in self.x_tokens:
                correctness = "CORRECT" if cflag else "INCORRECT"
                fout.write(f"Token='{tok}' Domain={domain} Correct={correctness}\n")
                fout.write(f"  Full sentence: {full_text}\n\n")

        print(f"\n[INFO] Wrote {len(self.x_tokens)} 'X' tokens to: {x_outfile}")
        print("==================== END OF ERROR ANALYSIS ====================")


if __name__ == "__main__":
    """
    1) Provide general_data.csv with 'text' and 'text_w_pairs' columns
    2) Provide social_data.jsonl files with 'text' and 'entities'
       containing cause/effect spans
    3) The script:
       - Builds token-level gold labels
       - Predicts with HF pipeline
       - Evaluates with seqeval
       - Splits correct vs. incorrect tokens
       - Runs prefixspan on (token, pos) & POS-only sequences
       - Outputs tokens assigned pos='X' to x_tokens.txt for manual inspection
    """
    GENERAL_CSV = r"data (annotated)/general_data.csv"
    SOCIAL_JSONL_PATHS = [
        r"data (annotated)/social_data_1.jsonl",
        r"data (annotated)/social_data_2.jsonl"
    ]

    analyzer = TokenLevelErrorAnalysis(
        general_csv_path=GENERAL_CSV,
        social_jsonl_paths=SOCIAL_JSONL_PATHS,
        model_name="tanfiona/unicausal-tok-baseline",
        output_dir="analysis_outputs",
        random_seed=42
    )
    analyzer.run_analysis(sample_size=60)
