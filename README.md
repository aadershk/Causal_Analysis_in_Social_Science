# Causal Analysis in Social Science

This repository contains Python scripts and notebooks aimed at analyzing, annotating, and visualizing causal relationships within social science text data. The project leverages large language models, fine-tuned token classifiers, and a custom pipeline to discover cause-effect pairs from a specialized corpus.

---

## Table of Contents
1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Dependencies](#dependencies)
4. [Usage](#usage)

---

## Overview

Social science texts often contain implicit causal relationships, demanding tailored methods for detection and interpretation. This project aims to bridge that gap by:

- **Preprocessing** large numbers of academic PDFs,
- **Automatically annotating** sentences via OpenAI’s GPT-based APIs,
- **Providing** interpretability tools (token-level attention heatmaps, POS tagging, and pattern mining).
- **Fine-tuning** token classification models (e.g., UniCausal, SSC-BERT),
- **Evaluating** performance with metrics (precision, recall, F1),


---

## Core Components

1. **Annotation Scripts**  
   - Automated labeling of social science sentences using OpenAI’s GPT-3.5-turbo, guided by domain-specific causality guidelines.

2. **Visualization Dash App**  
   - Interactive dashboard showing token-level predictions, confidence heatmaps, and comparisons of gold vs. predicted labels.

3. **POS & Pattern Mining**  
   - PrefixSpan-based pattern analysis for correct/incorrect tokens, highlighting domain-specific “X” tags in specialized text.

4. **Fine-Tuning Pipelines**  
   - Training and validation of Hugging Face Transformer models on specialized social science data, preserving original logic and hyperparameters.

3. **Evaluation Tools**  
   - Metrics demonstrating model performances (_Unicausal_tok_baseline_ and the fine-tuned model) on respective datasets ( general purpose data vs. social science literature)

---

## Dependencies

This project uses Python 3.8+ and a GPU-enabled environment (optional but recommended). A `requirements.txt` includes the necessary packages to run the code effectively.

> **Note**: Adjust versions as necessary for your environment, especially if you already have a specific GPU-compatible PyTorch installed.

---

## Usage

1. **Install Requirements**  
   ```bash
   pip install -r requirements.txt
   ```
2. **Data Pre-processing**  
   ```bash
   python pre-processing.py
   ```
3. **Annotation**  
   Modify and run the annotation script to process a CSV of social science sentences:
   ```bash
   # Firstly create a data (annotated sub folder)
   python annotation_openai.py
   ```
   This generates a `.jsonl` output with GPT-labeled cause/effect tokens.
   
4. **Preliminary evaluation**  
   ```bash
   # You will first need to download the general-purpose dataset from following link: https://github.com/tanfiona/CausalNewsCorpus/blob/master/data/V2/train_subtask2.csv
   # Create the following sub folder before running the files: annotation_folder
   python evaluation_metrics.py
   python evaluation_(causal_indicators).py
   ```
5. **Visualization**  
   Launch the Dash app for interactive token-level analysis:
   ```bash
   python seq-to-seq_BERT_visualization.py
   ```
   Access the local server link to browse heatmaps, gold vs. predicted labels, and overall classification performance.

4. **POS sequence tagging analysis**  
   ```bash
   python POS-tagging.py
   ```
   Besides the POS analysis results in the terminal, access the x_tokens.txt file in the analysis_outputs subfolder to view the tokens labeled as "X" which are found to be domain-specific terminologies pertaining to social science.

   
5. **Fine-Tuning**  
   Provide the annotated dataset to the training scripts (e.g., `fine_tune_model.py`), adjusting hyperparameters or file paths if needed:
   ```bash
   python fine_tune_model.py
   ```
   The resulting model is saved in the specified output directory.

4. **Evaluation - Fine Tuned model**  
   Evaluate the model performance with:
   ```bash
   python evaluation_fine_tuned_model.py
   ```
   The respective evaluation metrics are reported



