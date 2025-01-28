import csv
import json
import openai
import sys
import time
from typing import List, Tuple

###############################################################################
# 1) OPENAI API KEY
###############################################################################
# Mention the API key below
#openai.api_key = ""

###############################################################################
# 2) ANNOTATION GUIDELINES (VERBATIM)
###############################################################################
ANNOTATION_GUIDELINES = r"""
Annotation Guideline Structure for Causal Sentence Labeling
1. Introduction
1.1 Purpose
Capturing causal relationships in social science is essential to understanding the underlying mechanism behind social phenomena and developing robust theories (Hedström & Ylikoski, 2010). The causal relations are linked with a Directed Acyclic Graph, a graphical representation of causal relations in the form of cause-effect (Digitale et al., 2022). The graph could serve as a tool for researchers to hypothesize and test the mechanisms linking cause to effect. Researchers can also explore the relationship between variables and, as a result, enhance their causal claims (Gross, 2018). Empirical researchers can use DAG to find patterns between variables, find potential biases, and refine their assumptions about the underlying mechanisms of social phenomena (Shrier & Platt, 2008).

1.2 Definitions
A cause is defined as a condition or event that can lead to a change or have an effect. In social science, the cause could manifest itself as a factor, variable, or manipulation that affects or modifies behavior or produces an outcome or social phenomenon (Halpern, 2016). On the other hand, effects are consequences, outcomes, or results that are produced by the presence or manipulation of the cause. In social science, effects take various forms, such as changes in social behavior or economic conditions (Moustakas, 2023).
A causal relationship is a connection between two variables, where a change in the state of one variable (cause) can change the state of the other variable (effects) (Halpern, 2016). Put differently: the effect "listens to" its causes, and a change in the cause results in a change in the effect (Pearl, 1995).
The polarity of connection between cause and effect refers to the sign, or valence, of the relationship, whether modifying a cause will increase or decrease the effect regardless of our internal sentimental judgment. For example, the sentence "smoking causes lung cancer" invokes a negative sentiment – lung cancer is a terrible disease. However, the polarity of the connection is neutral as there is not any explicit information about the polarity in the sentence.

2. Annotation goals
This project aims to annotate a dataset of causal statements, identifying causes and effects. This dataset will be used to train machine learning models to perform the same task (identifying causes, effects). Training such models will make it possible to automatically extract DAGs from written text.

3. Guidelines for Identifying Causes, Effects, and Polarity of Connections
To capture a causal sentence and detecting its elements like cause, effect, and polarity, one needs to understand the sentence. You should look for a variable (cause) whose presence or modification leads to an outcome or changes the state of another variable (effect).

3.1 Watch out for Causality Indicators
Causality is often indicated by words such as “lead”, “cause”, “because”, “could lead”, “increase”, “decrease”. These explicit indicators are easily detected. But sometimes causal sentences lack these indicators. For example, compare the sentence, “Celebrating the victories increases team morale!” to the sentence “Celebrating team victories is essential for maintaining high morale.’’ In both sentences, the cause is “Celebrating the victories”, the effect is “team morale,” and the polarity of their connection is “positive” because the cause increases the effect. However, the second example does not include an explicit causality indicator. Such implicitly causal statements are relatively common in social science texts, so watch out for them.

3.2 Correlation does not equal Causation
Just because two variables change together, it does not mean that one is the cause of the other. The following example is not a causal sentence but shows a correlation: “There is a strong and significant correlation that those subjects who burn money tend to be also those who expect their counterpart to burn theirs.”

3.3 Be Objective
We should not let our personal judgment and assumptions influence our coding of a sentence. Limit the coding to the information contained in the sentence. For example, consider the sentence “smoking causes lung cancer”. We know that "smoking" is the cause, “lung cancer” is the effect, and that the sentiment of sentence is negative. But the sentence does not explicitly mention positive or negative polarity; it's a neutral sentence. If the sentence was "smoking increases the likelihood of lung cancer", we would code polarity as positive.

3.4 Order does not Matter
Although causes always precede effects, sentences are not always written that way. For example, the sentence "Last night's storm resulted in the destroyed house" and the sentence “The destroyed house” was the result of the “last night's storm.” are both identically coded: the cause is "storm". At this causal sentence, the effect is “destroyed house”, and the polarity is neutral. The fact that one sentence mentions the effect before the cause is irrelevant.

3.5 Code all Cause-Effect Pairs
A sentence could have more than one cause and effect. A cause can have more than one effect; vice versa, an effect could be connected with multiple causes in a single sentence. In such cases, code all cause-effect pairs separately.
Read the following sentence:
“In models based on income differences, if responders care for the utility of the other responder, then in addition to receiving disutility because the take authority has a higher income than they do, they will also receive disutility because the take authority has higher income than their friend does.”
In the above sentence, the effect, “disutility”, has “positive” connections with the following causes: “take authority has a higher income than they do”, “take authority has higher income than their friend does”, and “care for the utility of the other responder”.

5. Checklist
• If there is a cause in a sentence, at least there will be an effect, and vice-versa
• Do not include your personal feelings and judgment for annotation. If you feel a sentence sentiment is negative, it does not necessarily imply that the connection polarity is negative!
• Read the full sentence and then start to highlight the sentence elements. In some complicated sentences, a cause can be the effect of another cause simultaneously!
• Always follow the definitions and guidelines!
"""

###############################################################################
# 3) SINGLE PROMPT FOR BATCH ANNOTATION
###############################################################################
BATCH_CALL_PROMPT = r"""
You have two tasks for a list of sentences:

1.) For each sentence in the batch, check if it meets any of these conditions:
   - Grammatically incorrect, contains random characters, or lacks coherence.
   - Contains subwords resulting from improper preprocessing (e.g., tokens split unnaturally such as "ex plo it").
   If a sentence meets any of these, include it in the "discarded" list.

2.) For each valid sentence, assign each token a label:
   - `B-C` for the beginning of causes
   - `I-C` for tokens continuing a cause
   - `B-E` for the beginning of effects
   - `I-E` for tokens continuing an effect
   - `O` for tokens outside any causal relationship.

Return a JSON object with two keys:
{
  "valid": [
    {"sentence": "<sentence>", "tokens": ["<token1>", "<token2>", ...], "labels": ["<label1>", "<label2>", ...]},
    ...
  ],
  "discarded": ["<sentence1>", "<sentence2>", ...]
}
"""

###############################################################################
# 4) BUILD PROMPT FOR A BATCH
###############################################################################
def build_batch_prompt(sentences: List[str]) -> str:
    """
    Combines multiple sentences into a single prompt for batch processing.
    """
    sentences_text = "\n".join([f"- {s}" for s in sentences])
    return f"{BATCH_CALL_PROMPT}\n\n{ANNOTATION_GUIDELINES}\nSentences:\n{sentences_text}"

###############################################################################
# 5) CALL OPENAI API (BATCH MODE)
###############################################################################
def call_openai_api(prompt: str, max_retries: int = 5) -> str:
    """
    Calls the OpenAI API. Retries on rate limit errors.
    """
    retries = 0
    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                temperature=0.0
            )
            return response["choices"][0]["message"]["content"].strip()
        except openai.error.RateLimitError:
            retries += 1
            wait_time = 10 * retries
            print(f"[WARN] Rate limit exceeded. Retrying in {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"[ERROR] API call failed: {e}", file=sys.stderr)
            return ""
    return ""

###############################################################################
# 6) PARSE RESPONSE
###############################################################################
def parse_batch_response(resp_text: str) -> Tuple[List[str], List[dict]]:
    """
    Parses the API response for a batch of sentences.
    Returns two lists:
    - discarded: List of sentences marked as "DISCARD"
    - valid: List of valid sentence results with tokens and labels
    """
    try:
        results = json.loads(resp_text)
        discarded = results.get("discarded", [])
        valid = results.get("valid", [])
        return discarded, valid
    except json.JSONDecodeError:
        print("[ERROR] Failed to parse JSON response. Saving batch for review.", file=sys.stderr)
        with open("../../OneDrive/Desktop/Causal_Analysis_in_Social_Science/failed_batch_response.txt", "w", encoding="utf-8") as f:
            f.write(resp_text)  # Save the problematic response for debugging
        return [], []

###############################################################################
# 7) PROCESS CSV (BATCH MODE)
###############################################################################
def process_csv(
    input_csv: str,
    output_jsonl: str,
    sentence_column: str,
    max_rows: int,
    batch_size: int = 4
):
    """
    Reads input CSV, processes sentences in batches, and writes labeled outputs to a JSONL file.
    """
    count = 0
    batch = []

    with open(input_csv, "r", encoding="utf-8-sig") as fin, open(output_jsonl, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        for row_idx, row in enumerate(reader, start=1):
            if count >= max_rows:
                break

            sentence = row.get(sentence_column, "").strip()
            if not sentence:
                continue

            batch.append(sentence)
            if len(batch) == batch_size:
                # Process the batch
                prompt = build_batch_prompt(batch)
                response = call_openai_api(prompt)
                discarded, valid = parse_batch_response(response)

                for valid_result in valid:
                    fout.write(json.dumps(valid_result, ensure_ascii=False) + "\n")
                    count += 1
                    print(f"[INFO] Annotated: {valid_result['sentence']}")

                if discarded:
                    for disc_sentence in discarded:
                        print(f"[INFO] Discarded: {disc_sentence}")

                # Reset batch
                batch = []

        # Process any remaining sentences in the batch
        if batch:
            prompt = build_batch_prompt(batch)
            response = call_openai_api(prompt)
            discarded, valid = parse_batch_response(response)

            for valid_result in valid:
                fout.write(json.dumps(valid_result, ensure_ascii=False) + "\n")
                count += 1
                print(f"[INFO] Annotated: {valid_result['sentence']}")

            if discarded:
                for disc_sentence in discarded:
                    print(f"[INFO] Discarded: {disc_sentence}")

    print(f"\n[INFO] Finished. Annotated {count} sentences (max {max_rows}).")
    print(f"[INFO] Output in => {output_jsonl}")

###############################################################################
# 8) MAIN
###############################################################################
if __name__ == "__main__":
    input_csv_file = "../../OneDrive/Desktop/Causal_Analysis_in_Social_Science/annotation_folder/causal_sentences.csv"
    output_jsonl_file = "../../OneDrive/Desktop/Causal_Analysis_in_Social_Science/annotation_folder/annotated_output.jsonl"
    process_csv(
        input_csv=input_csv_file,
        output_jsonl=output_jsonl_file,
        sentence_column="causal_sentence",
        max_rows=3500,
        batch_size=4  # Process 4 sentences per batch
    )
    print("[DONE] Annotation complete.")
