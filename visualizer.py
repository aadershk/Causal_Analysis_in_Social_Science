import os
import json
import numpy as np
import torch
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForTokenClassification

###############################################################################
# CONFIGURATION
###############################################################################
MODEL_NAME = "tanfiona/unicausal-tok-baseline"
DATA_FILE = "data (annotated)/social_data_final.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_LIST = ["O", "B-C", "I-C", "B-E", "I-E"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
LABEL_COLORS = {
    "O": "black",
    "B-C": "red",
    "I-C": "orange",
    "B-E": "blue",
    "I-E": "cyan"
}

###############################################################################
# LOAD MODEL AND DATA
###############################################################################
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME).to(DEVICE)
    return model, tokenizer

def load_annotated_data(data_file):
    with open(data_file, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]
    # Remove `[CLS]` and `[SEP]` tokens
    for record in data:
        record["tokens"] = [token for token in record["tokens"] if token not in ["[CLS]", "[SEP]"]]
        record["labels"] = record["labels"][: len(record["tokens"])]
    return data

###############################################################################
# PREDICT FUNCTION
###############################################################################
def predict_and_visualize(sentence_tokens, model, tokenizer):
    encoding = tokenizer.encode_plus(
        sentence_tokens,
        is_split_into_words=True,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0]

    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    predicted_labels = np.argmax(probs, axis=-1)

    return probs, predicted_labels

###############################################################################
# COLOR-CODED SENTENCE
###############################################################################
def color_code_sentence(tokens, labels):
    """
    Create a list of styled html.Span elements for color-coded tokens.
    """
    return [
        html.Span(
            token,
            style={"color": LABEL_COLORS.get(label, "black"), "marginRight": "5px"}
        )
        for token, label in zip(tokens, labels)
    ]

###############################################################################
# DASH APP SETUP
###############################################################################
model, tokenizer = load_model_and_tokenizer()
data = load_annotated_data(DATA_FILE)

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1(
            "Causal Relationship Token Classification Visualizer",
            style={"textAlign": "center", "marginBottom": "20px"},
        ),
        dcc.Slider(
            id="sentence-slider",
            min=0,
            max=len(data) - 1,
            step=1,
            value=0,
            marks={
                i: f"Sentence {i+1}"
                for i in range(0, len(data), max(1, len(data) // 10))
            },
        ),
        html.Div(
            id="sentence-display",
            style={"textAlign": "center", "margin": "20px", "fontSize": "18px"},
        ),
        dcc.Graph(id="heatmap"),
        html.Div(
            id="label-comparison",
            style={"textAlign": "center", "marginTop": "20px", "fontSize": "18px"},
        ),
        html.Div(
            children=[
                html.P("Label Color Coding:", style={"fontWeight": "bold"}),
                html.Div(
                    [
                        html.Span("O: Outside", style={"color": "black", "marginRight": "20px"}),
                        html.Span("B-C: Begin Cause", style={"color": "red", "marginRight": "20px"}),
                        html.Span("I-C: Inside Cause", style={"color": "orange", "marginRight": "20px"}),
                        html.Span("B-E: Begin Effect", style={"color": "blue", "marginRight": "20px"}),
                        html.Span("I-E: Inside Effect", style={"color": "cyan", "marginRight": "20px"}),
                    ],
                    style={"textAlign": "center"},
                ),
            ],
            style={"marginTop": "20px"},
        ),
    ]
)

###############################################################################
# CALLBACKS
###############################################################################
@app.callback(
    [Output("sentence-display", "children"),
     Output("heatmap", "figure"),
     Output("label-comparison", "children")],
    [Input("sentence-slider", "value")],
)
def update_visualization(index):
    example = data[index]
    sentence = example["sentence"]
    tokens = example["tokens"]
    true_labels = example["labels"]

    probs, predicted_labels = predict_and_visualize(tokens, model, tokenizer)
    predicted_label_names = [ID2LABEL[label] for label in predicted_labels]

    probs = probs[:len(tokens), :]  # Adjust probabilities to match token length

    fig = px.imshow(
        probs.T,
        labels=dict(x="Tokens", y="Labels", color="Probability"),
        x=tokens,
        y=LABEL_LIST,
        title="Token-to-Label Probability Heatmap",
        aspect="auto",
    )
    fig.update_xaxes(tickangle=45)

    predicted_colored = color_code_sentence(tokens, predicted_label_names)
    true_colored = color_code_sentence(tokens, true_labels)

    comparison = html.Div(
        [
            html.Div(
                ["Predicted: "] + predicted_colored,
                style={"marginBottom": "10px"},
            ),
            html.Div(["True: "] + true_colored),
        ],
        style={"textAlign": "center", "marginTop": "20px"},
    )

    return sentence, fig, comparison

###############################################################################
# RUN APP
###############################################################################
if __name__ == "__main__":
    app.run_server(debug=True)
