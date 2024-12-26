import os
import fitz  # PyMuPDF
import spacy
import re
import wordninja
import csv
import torch
from concurrent.futures import ProcessPoolExecutor
from langdetect import detect, LangDetectException
from transformers import AutoTokenizer, AutoModelForSequenceClassification

##############################################################################
# 1) PIPELINE CONFIG & GLOBAL SETUP
##############################################################################

class SSCBertModel:
    """
    Loads the 'ssc_bert' model and tokenizer on GPU if available.
    Provides a method to classify batches of sentences (causal vs. non-causal).
    """
    def __init__(self, model_name="rasoultilburg/ssc_bert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def classify_sentences(self, sentences, batch_size=128):
        """
        Classify each sentence as 0 (non-causal) or 1 (causal).
        Returns a list of only the causal sentences.
        """
        causal_sentences = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            tokens = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            tokens = tokens.to(self.device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = self.model(**tokens)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1).cpu().numpy()

            # collect only those sentences predicted as causal (pred==1)
            for sentence, pred in zip(batch, predictions):
                if pred == 1:
                    causal_sentences.append(sentence)

            del tokens, outputs, logits
            torch.cuda.empty_cache()

        return causal_sentences


class PDFPipeline:
    """
    Handles:
      - Extracting text from PDF
      - Splitting text into sentences
      - Preprocessing sentences (parallel)
      - Writing causal sentences to CSV
    """

    spacy_nlp = spacy.load("en_core_web_sm")

    def __init__(self, ssc_model: SSCBertModel):
        """
        ssc_model: an instance of SSCBertModel
        """
        self.ssc_model = ssc_model

    ##########################################################################
    # EXTRACT & SPLIT
    ##########################################################################
    @staticmethod
    def extract_pdf_text(pdf_path):
        """
        Opens the PDF with PyMuPDF and concatenates text from each page.
        """
        with fitz.open(pdf_path) as doc:
            text = "".join(page.get_text() for page in doc)
        return text

    @classmethod
    def split_into_sentences(cls, text):
        """
        Uses spaCy to split text into sentences, removing empty lines.
        """
        doc = cls.spacy_nlp(text)
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]

    ##########################################################################
    # PREPROCESS
    ##########################################################################
    @staticmethod
    def preprocess_text(text):
        """
        - Removes newlines/tabs
        - Skips short (<50 chars) or non-English
        - Replaces ligatures
        - Splits words with wordninja
        - Strips
        - Skips if <30 chars post-split
        Returns preprocessed string or None.
        """
        text = re.sub(r'[\n\t\003]', ' ', text)

        if len(text.strip()) < 50:
            return None

        try:
            if detect(text) != 'en':
                return None
        except LangDetectException:
            return None

        ligatures = {"ﬁ": "fi", "ﬂ": "fl"}
        for ligature, replacement in ligatures.items():
            text = text.replace(ligature, replacement)

        text = ' '.join(wordninja.split(text))
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) < 30:
            return None

        return text

    @staticmethod
    def preprocess_sentences_parallel(sentences):
        """
        Parallel map of preprocess_text across all sentences.
        Returns only non-None results.
        """
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(PDFPipeline.preprocess_text, sentences))
        return [r for r in results if r is not None]

    ##########################################################################
    # PROCESSING PIPELINE
    ##########################################################################
    def process_pdf(self, dataset_id, pdf_path, writer):
        """
        1) Extract text
        2) Split into sentences
        3) Parallel-preprocess
        4) Classify cause vs. non-cause
        5) Write causal sentences to CSV
        """
        document_id = os.path.basename(pdf_path)
        text = self.extract_pdf_text(pdf_path)
        sentences = self.split_into_sentences(text)
        preprocessed = self.preprocess_sentences_parallel(sentences)

        # Use SSCBertModel to classify
        causal_sentences = self.ssc_model.classify_sentences(preprocessed)

        for sentence in causal_sentences:
            # Safely encode to avoid weird chars in CSV
            sentence = sentence.encode('utf-8', 'replace').decode('utf-8')
            writer.writerow([dataset_id, document_id, sentence])


##############################################################################
# MAIN
##############################################################################
def main():
    input_folders = [
        "/data (pre-processing)/ThirdDataset_PDF",
        "/data (pre-processing)/Xavier_PDF",
        "/data (pre-processing)/ThirdDataset_PDF"
    ]
    output_folder = "/content/drive/MyDrive/processed_csv_causal"
    os.makedirs(output_folder, exist_ok=True)

    output_csv = os.path.join(output_folder, "causal_sentences.csv")

    # Instantiate the model
    ssc_model = SSCBertModel()
    # Instantiate pipeline
    pipeline_processor = PDFPipeline(ssc_model=ssc_model)

    with open(output_csv, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(["dataset_id", "document_id", "causal_sentence"])

        for folder in input_folders:
            dataset_id = os.path.basename(folder)
            for pdf_file in os.listdir(folder):
                if pdf_file.endswith(".pdf"):
                    pdf_path = os.path.join(folder, pdf_file)
                    pipeline_processor.process_pdf(dataset_id, pdf_path, writer)

    print("Pipeline execution completed. Check the 'processed_csv_causal' directory for results.")


if __name__ == "__main__":
    main()
