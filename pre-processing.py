import os
import re
import csv
import fitz
import spacy
import torch
import wordninja
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from langdetect import detect, LangDetectException
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

class SSCBertModel:
    """
    Loads an SSC-BERT model for classifying sentences as causal or not.
    """
    def __init__(self, model_name: str = "rasoultilburg/ssc_bert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def classify_sentences(self, sentences: List[str], batch_size: int = 128) -> List[str]:
        causal_sents = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = self.model(**tokens)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            for sent, pred in zip(batch, preds):
                if pred == 1:
                    causal_sents.append(sent)

            del tokens, outputs
            torch.cuda.empty_cache()

        return causal_sents


class PDFPipeline:
    """
    Extracts text from PDFs, preprocesses it, classifies sentences with SSC-BERT, and writes to CSV.
    """
    spacy_nlp = spacy.load("en_core_web_sm")

    def __init__(self, ssc_model: SSCBertModel):
        self.ssc_model = ssc_model

    @staticmethod
    def extract_pdf_text(pdf_path: str) -> str:
        with fitz.open(pdf_path) as doc:
            return "".join(page.get_text() for page in doc)

    @classmethod
    def split_into_sentences(cls, text: str) -> List[str]:
        doc = cls.spacy_nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    @staticmethod
    def preprocess_text(text: str) -> str:
        text = re.sub(r'[\n\t\003]', ' ', text)
        if len(text.strip()) < 50:
            return None
        try:
            if detect(text) != 'en':
                return None
        except LangDetectException:
            return None

        ligatures = {"ﬁ": "fi", "ﬂ": "fl"}
        for lig, rep in ligatures.items():
            text = text.replace(lig, rep)

        text = ' '.join(wordninja.split(text))
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) < 30:
            return None
        return text

    @staticmethod
    def preprocess_sentences_parallel(sentences: List[str]) -> List[str]:
        with ProcessPoolExecutor() as executor:
            processed = list(executor.map(PDFPipeline.preprocess_text, sentences))
        return [p for p in processed if p]

    def process_pdf(self, dataset_id: str, pdf_path: str, writer: csv.writer) -> None:
        doc_id = os.path.basename(pdf_path)
        raw_text = self.extract_pdf_text(pdf_path)
        sentences = self.split_into_sentences(raw_text)
        preprocessed = self.preprocess_sentences_parallel(sentences)
        causal_sents = self.ssc_model.classify_sentences(preprocessed)

        for sent in causal_sents:
            sent = sent.encode('utf-8', 'replace').decode('utf-8')
            writer.writerow([dataset_id, doc_id, sent])


def main():
    input_folders = [
        "/data (pre-processing)/ThirdDataset_PDF",
        "/data (pre-processing)/Xavier_PDF",
        "/data (pre-processing)/ThirdDataset_PDF"
    ]
    output_folder = "annotation_folder/causal_sentences.csv"
    os.makedirs(output_folder, exist_ok=True)

    output_csv = os.path.join(output_folder, "causal_sentences.csv")
    ssc_model = SSCBertModel()
    pdf_processor = PDFPipeline(ssc_model=ssc_model)

    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as fout:
        writer = csv.writer(fout)
        writer.writerow(["dataset_id", "document_id", "causal_sentence"])

        for folder in input_folders:
            dataset_id = os.path.basename(folder)
            for pdf_file in os.listdir(folder):
                if pdf_file.endswith(".pdf"):
                    pdf_path = os.path.join(folder, pdf_file)
                    pdf_processor.process_pdf(dataset_id, pdf_path, writer)

    print("Pipeline execution completed. Check results in 'processed_csv_causal'.")

if __name__ == "__main__":
    main()
