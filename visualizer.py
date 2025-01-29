import os
import json
import numpy as np
import torch
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Model and data paths
MODEL_NAME = "tanfiona/unicausal-tok-baseline"
DATA_FILE = "data (annotated)/social_data_final.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_LIST = ["O", "B-C", "I-C", "B-E", "I-E"]
ID2LABEL = {i: lbl for i, lbl in enumerate(LABEL_LIST)}
LABEL_COLORS = {
    "O": "black",
    "B-C": "red",
    "I-C": "orange",
    "B-E": "blue",
    "I-E": "cyan"
}


class TokenClassificationVisualizer:
    """Loads a token-classification model, annotated data, and sets up a Dash app for visualization."""

    def __init__(self, model_name: str, data_file: str) -> None:
        self.model_name = model_name
        self.data_file = data_file
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.data = self._load_annotated_data()
        self.app = Dash(__name__)
        self._configure_app_layout()

    def _load_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForTokenClassification.from_pretrained(self.model_name).to(DEVICE)
        return model, tokenizer

    def _load_annotated_data(self):
        with open(self.data_file, "r", encoding="utf-8") as fin:
            records = [json.loads(line.strip()) for line in fin]

        # Remove [CLS], [SEP] tokens if any
        for rec in records:
            rec["tokens"] = [t for t in rec["tokens"] if t not in ["[CLS]", "[SEP]"]]
            rec["labels"] = rec["labels"][: len(rec["tokens"])]
        return records

    def _predict_and_visualize(self, tokens):
        enc = self.tokenizer.encode_plus(
            tokens,
            is_split_into_words=True,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            out = self.model(input_ids, attention_mask=attention_mask)
            logits = out.logits[0]

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_ids = np.argmax(probs, axis=-1)
        return probs, pred_ids

    def _color_code_sentence(self, tokens, labels):
        return [
            html.Span(
                token,
                style={"color": LABEL_COLORS.get(lbl, "black"), "marginRight": "5px"}
            )
            for token, lbl in zip(tokens, labels)
        ]

    def _configure_app_layout(self):
        self.app.layout = html.Div(
            [
                html.H1(
                    "Causal Relationship Token Classification Visualizer",
                    style={"textAlign": "center", "marginBottom": "20px"},
                ),
                dcc.Slider(
                    id="sentence-slider",
                    min=0,
                    max=len(self.data) - 1,
                    step=1,
                    value=0,
                    marks={
                        i: f"Sentence {i+1}"
                        for i in range(0, len(self.data), max(1, len(self.data) // 10))
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

        @self.app.callback(
            [
                Output("sentence-display", "children"),
                Output("heatmap", "figure"),
                Output("label-comparison", "children"),
            ],
            [Input("sentence-slider", "value")],
        )
        def update_visualization(index):
            rec = self.data[index]
            sentence = rec["sentence"]
            tokens = rec["tokens"]
            true_labels = rec["labels"]

            probs, pred_ids = self._predict_and_visualize(tokens)
            pred_labels = [ID2LABEL[i] for i in pred_ids]
            probs = probs[: len(tokens), :]

            fig = px.imshow(
                probs.T,
                labels=dict(x="Tokens", y="Labels", color="Probability"),
                x=tokens,
                y=LABEL_LIST,
                title="Token-to-Label Probability Heatmap",
                aspect="auto",
            )
            fig.update_xaxes(tickangle=45)

            predicted_colored = self._color_code_sentence(tokens, pred_labels)
            true_colored = self._color_code_sentence(tokens, true_labels)
            comparison = html.Div(
                [
                    html.Div(["Predicted: "] + predicted_colored, style={"marginBottom": "10px"}),
                    html.Div(["True: "] + true_colored),
                ],
                style={"textAlign": "center", "marginTop": "20px"},
            )
            return sentence, fig, comparison

    def run(self, debug=True):
        self.app.run_server(debug=debug)


def main():
    visualizer = TokenClassificationVisualizer(MODEL_NAME, DATA_FILE)
    visualizer.run(debug=True)

if __name__ == "__main__":
    main()
