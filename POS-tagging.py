import os
import csv
import json
import re
import random
from typing import List, Dict, Any
from collections import Counter, defaultdict

from tqdm import tqdm
import spacy
from prefixspan import PrefixSpan
import warnings

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

warnings.filterwarnings("ignore", category=UserWarning)

##############################################################################
#                         HELPER FUNCTIONS                                  #
##############################################################################

def strip_tags(tok: str) -> str:
    """
    Remove <ARGx> and <SIGx> tags from a token.
    """
    tok_clean = re.sub(r"</?ARG[01]>", "", tok)
    tok_clean = re.sub(r"</?SIG\d*>", "", tok_clean)
    return tok_clean.strip()

def unify_label(lab: str) -> str:
    """
    Convert B-C/I-C => 'C', B-E/I-E => 'E', else 'O'.
    """
    if lab in ("B-C", "I-C"):
        return "C"
    elif lab in ("B-E", "I-E"):
        return "E"
    else:
        return "O"

def map_pipeline_label_to_BIO(label_group: str) -> str:
    """
    Model's label mapping:
       LABEL_0 -> B-C
       LABEL_1 -> B-E
       LABEL_2 -> I-C
       LABEL_3 -> I-E
    """
    mapping = {
        "LABEL_0": "B-C",
        "LABEL_1": "B-E",
        "LABEL_2": "I-C",
        "LABEL_3": "I-E"
    }
    return mapping.get(label_group, "O")

def build_char_label_array(text: str, preds, label_group_mapping) -> List[str]:
    """
    Create a char-level label array, init 'O'. Fill with 'C' or 'E' for cause/effect spans.
    """
    char_labels = ["O"] * len(text)
    for pr in preds:
        grp = pr["entity_group"]
        mapped_bio = label_group_mapping(grp)
        s_i, e_i = pr["start"], pr["end"]
        if s_i < 0 or e_i > len(text):
            continue
        if mapped_bio in ("B-C", "I-C"):
            for c_idx in range(s_i, e_i):
                if 0 <= c_idx < len(char_labels):
                    char_labels[c_idx] = "C"
        elif mapped_bio in ("B-E", "I-E"):
            for c_idx in range(s_i, e_i):
                if 0 <= c_idx < len(char_labels):
                    char_labels[c_idx] = "E"
    return char_labels

def predict_labels_on_tokens(pipe, tokens: List[str]) -> List[str]:
    """
    Predict token-level cause/effect labels using HF pipeline output, returning C/E/O labels.
    """
    joined_text = " ".join(tokens)
    preds = pipe(joined_text)
    char_labels = build_char_label_array(joined_text, preds, map_pipeline_label_to_BIO)

    # Map back to tokens (B-C, I-C, B-E, I-E)
    pred_bio_labels = []
    char_ptr = 0
    for tok in tokens:
        seg = char_labels[char_ptr : char_ptr + len(tok)]
        if all(x == "O" for x in seg):
            pred_bio_labels.append("O")
        else:
            if "C" in seg:
                if seg[0] == "C":
                    pred_bio_labels.append("B-C")
                else:
                    pred_bio_labels.append("I-C")
            elif "E" in seg:
                if seg[0] == "E":
                    pred_bio_labels.append("B-E")
                else:
                    pred_bio_labels.append("I-E")
            else:
                pred_bio_labels.append("O")
        char_ptr += len(tok) + 1  # +1 for space

    # unify
    pred_unified = [unify_label(x) for x in pred_bio_labels]
    return pred_unified

def align_spacy_with_tokens(raw_text: str, token_list: List[str], nlp):
    """
    Attempt to align each 'token' with spaCy's POS. If spaCy splits differently, fallback is 'X'.
    """
    doc = nlp(raw_text)
    pos_map = defaultdict(list)
    # Collect each doc token text->pos in a list to handle duplicates
    # e.g. if doc has 'the' multiple times
    for t in doc:
        pos_map[t.text].append(t.pos_)

    aligned = []
    for tok in token_list:
        if pos_map[tok]:
            aligned.append((tok, pos_map[tok].pop(0)))  # take the first available pos
        else:
            aligned.append((tok, "X"))
    return aligned

##############################################################################
#                        MAIN ANALYSIS CLASS                                #
##############################################################################

class TokenLevelErrorAnalysis:
    """
    Token-level cause/effect error analysis.
    Now extended to:
      - Also mine top POS-only sequences (besides (token, pos)).
      - Output all tokens labeled pos='X' to a text file for manual inspection.
    """

    def __init__(self,
                 general_csv_path: str,
                 social_jsonl_paths: List[str],
                 model_name: str = "tanfiona/unicausal-tok-baseline",
                 output_dir: str = "analysis_outputs",
                 random_seed: int = 42):
        self.general_csv_path = general_csv_path
        self.social_jsonl_paths = social_jsonl_paths
        self.model_name = model_name
        self.output_dir = output_dir
        self.random_seed = random_seed

        random.seed(self.random_seed)

        self.all_data = []  # final token-level data

        # We store subsets for correct vs. incorrect tokens
        self.correct_tokens_general = []
        self.incorrect_tokens_general = []
        self.correct_tokens_social = []
        self.incorrect_tokens_social = []

        # We'll store tokens that have pos='X'
        self.x_tokens = []  # a list of tuples: (token_text, domain, correctness_flag, context?)

        # spaCy + HF pipeline
        self.nlp = None
        self.pipe = None
        self._init_spacy()
        self._init_pipeline()

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    ###########################################################################
    # INITIALIZATION
    ###########################################################################
    def _init_spacy(self):
        print("[INFO] Loading spaCy (en_core_web_sm)...")
        self.nlp = spacy.load("en_core_web_sm")
        try:
            self.nlp.to_gpu()
            print("[INFO] spaCy to_gpu successful (if GPU available).")
        except Exception:
            print("[INFO] spaCy on CPU.")

    def _init_pipeline(self):
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

    ###########################################################################
    # GOLD LABEL BUILDING (GENERAL)
    ###########################################################################
    def load_general_data(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.general_csv_path):
            raise FileNotFoundError(f"General CSV not found: {self.general_csv_path}")

        data = []
        with open(self.general_csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            if not fields:
                raise ValueError("No columns in general CSV.")
            required = {"text", "text_w_pairs"}
            missing = required - set(fields)
            if missing:
                raise KeyError(f"Missing columns {missing} in {self.general_csv_path}.")

            for row in reader:
                raw_text = row["text"]
                text_w_pairs = row["text_w_pairs"]
                if not raw_text or not text_w_pairs:
                    continue
                # Must have <ARG0> or <ARG1>
                if ("<ARG0>" not in text_w_pairs) and ("<ARG1>" not in text_w_pairs):
                    continue
                tokens, gold_bio = self._build_gold_labels_general(raw_text, text_w_pairs)
                gold_unified = [unify_label(x) for x in gold_bio]
                data.append({
                    "domain": "general",
                    "text": raw_text,
                    "tokens": tokens,
                    "gold_labels": gold_unified
                })
        return data

    def _build_gold_labels_general(self, raw_text: str, text_w_pairs: str):
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

            ctok = strip_tags(p_tok)
            if not ctok:
                continue
            cleaned_tokens.append(ctok)
            gold_labels.append(local_label)

        L = min(len(raw_tokens), len(cleaned_tokens))
        tokens = raw_tokens[:L]
        labs = gold_labels[:L]
        return tokens, labs

    ###########################################################################
    # GOLD LABEL BUILDING (SOCIAL)
    ###########################################################################
    def load_social_data(self) -> List[Dict[str, Any]]:
        data = []
        for path in self.social_jsonl_paths:
            if not os.path.exists(path):
                print(f"[WARNING] Social file not found: {path}")
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line.strip())
                    text = obj["text"]
                    entities = obj.get("entities", [])
                    ce_entities = [e for e in entities if e["label"].lower() in ["cause", "effect"]]
                    if not ce_entities:
                        continue
                    tokens, gold_bio = self._build_gold_labels_social(text, ce_entities)
                    gold_unified = [unify_label(x) for x in gold_bio]
                    data.append({
                        "domain": "social",
                        "text": text,
                        "tokens": tokens,
                        "gold_labels": gold_unified
                    })
        return data

    def _build_gold_labels_social(self, text: str, entities: List[Dict[str, Any]]):
        raw_tokens = text.split()
        # build char positions
        char_positions = []
        idx = 0
        for rt in raw_tokens:
            start_i = idx
            end_i = idx + len(rt)
            char_positions.append((start_i, end_i))
            idx = end_i + 1

        labels = ["O"] * len(raw_tokens)

        cause_spans = []
        effect_spans = []
        for e in entities:
            lb = e["label"].lower()
            st = e["start_offset"]
            en = e["end_offset"]
            if lb == "cause":
                cause_spans.append((st, en))
            elif lb == "effect":
                effect_spans.append((st, en))

        def label_span(span_list, B_tag, I_tag):
            in_span = False
            for (st, en) in span_list:
                for i, (cst, cen) in enumerate(char_positions):
                    if cen > st and cst < en:  # overlap
                        if not in_span:
                            labels[i] = B_tag
                            in_span = True
                        else:
                            if labels[i] == "O" or labels[i].startswith("B-"):
                                labels[i] = I_tag
                    else:
                        if cst >= en:
                            in_span = False

        label_span(cause_spans, "B-C", "I-C")
        label_span(effect_spans, "B-E", "I-E")

        return raw_tokens, labels

    ###########################################################################
    # MAIN RUN
    ###########################################################################
    def run_analysis(self, sample_size=60):
        # 1) Load general + social
        general_data = self.load_general_data()
        social_data = self.load_social_data()
        combined = general_data + social_data

        # 2) sample
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

        # 3) Predict
        for item in tqdm(final_data, desc="Model inference"):
            pred_unified = predict_labels_on_tokens(self.pipe, item["tokens"])
            item["pred_labels"] = pred_unified

        # 4) Evaluate (seqeval)
        gold_sequences = []
        pred_sequences = []
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

        # 5) Align spaCy POS + Mark correct/incorrect
        self.all_data = []
        for item in final_data:
            domain = item["domain"]
            text = item["text"]
            toks = item["tokens"]
            golds = item["gold_labels"]
            preds = item["pred_labels"]

            # Align pos
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
            for i, tok in enumerate(toks):
                g = golds[i]
                p = preds[i]
                cflag = (g == p)
                # If spaCy split mismatch, fallback is 'X'
                assigned_pos = pos_aligned[i][1] if i < len(pos_aligned) else "X"
                record["tokens"].append(tok)
                record["pos_tags"].append(assigned_pos)
                record["gold_labels"].append(g)
                record["pred_labels"].append(p)
                record["correct_flags"].append(cflag)
            self.all_data.append(record)

        # 6) Final error analysis
        self.final_error_analysis()

    ###########################################################################
    # ERROR ANALYSIS
    ###########################################################################
    def final_error_analysis(self):
        """
        - Create 4 subsets: general-correct, general-incorrect, social-correct, social-incorrect
        - For each subset, we do:
          (1) prefixspan on (token, pos)
          (2) prefixspan on POS-only sequences
        - Collect all pos='X' tokens in a separate list, then write them to x_tokens.txt
        """
        # Subsets as lists of lists (one list per sentence):
        gc_tokpos = []
        gi_tokpos = []
        sc_tokpos = []
        si_tokpos = []

        # Also build parallel POS-only sequences
        gc_posonly = []
        gi_posonly = []
        sc_posonly = []
        si_posonly = []

        # We'll gather all X tokens for post-run
        self.x_tokens = []

        for record in self.all_data:
            domain = record["domain"]
            tokens = record["tokens"]
            poss = record["pos_tags"]
            cflags = record["correct_flags"]

            correct_seq = []
            incorrect_seq = []
            correct_pos_seq = []
            incorrect_pos_seq = []

            for i, (tok, pos, cflag) in enumerate(zip(tokens, poss, cflags)):
                if pos == "SPACE" or pos == "PUNCT":
                    continue

                # Capture X tokens for debugging
                if pos == "X":
                    # store (token, domain, correctness) in a list
                    self.x_tokens.append((tok, domain, cflag, record["text"]))

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

        # Print distributions or run prefixspan on them
        print("\n==================== TOKEN-LEVEL ERROR ANALYSIS ====================")

        # Quick POS distribution function
        def compute_pos_dist(list_of_sequences):
            cnt = Counter()
            for seq in list_of_sequences:
                for (_, pos) in seq:
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

        # Now run prefixspan
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

        # POS-only prefixspan
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

        # Finally, write out X tokens to a txt file
        x_outfile = os.path.join(self.output_dir, "x_tokens.txt")
        with open(x_outfile, "w", encoding="utf-8") as f:
            for x_tok, domain, cflag, full_text in self.x_tokens:
                c_str = "CORRECT" if cflag else "INCORRECT"
                f.write(f"Token='{x_tok}' Domain={domain} Correct={c_str}\n")
                f.write(f"  Full sentence: {full_text}\n\n")

        print(f"\n[INFO] Wrote {len(self.x_tokens)} 'X' tokens to: {x_outfile}")
        print("==================== END OF ERROR ANALYSIS ====================")


##############################################################################
#                                MAIN                                        #
##############################################################################

if __name__ == "__main__":
    """
    Example usage:
     - Provide your general_data.csv with 'text' and 'text_w_pairs' columns.
     - Provide a list of social_data.jsonl files with 'text' and 'entities' 
       containing cause/effect spans.
     - The script:
       1) Builds token-level gold labels
       2) Predicts with HF pipeline
       3) Evaluates with seqeval
       4) Splits correct vs. incorrect tokens 
       5) Runs prefixspan on (token, pos) and POS-only sequences
       6) Outputs all tokens assigned pos='X' to x_tokens.txt for manual inspection
    """
    # Adjust these paths as appropriate:
    GENERAL_CSV = r"data (annotated)/general_data.csv"
    SOCIAL_JSONL_PATHS = [
        r"data (annotated)/social_data_1.jsonl",
        r"data (annotated)/social_data_2.jsonl"
    ]

    analysis = TokenLevelErrorAnalysis(
        general_csv_path=GENERAL_CSV,
        social_jsonl_paths=SOCIAL_JSONL_PATHS,
        model_name="tanfiona/unicausal-tok-baseline",
        output_dir="analysis_outputs",
        random_seed=42
    )
    analysis.run_analysis(sample_size=60)
