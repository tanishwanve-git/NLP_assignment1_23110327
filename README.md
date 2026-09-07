# Toxic Comment Classification and Moderation System

A multi-stage NLP pipeline that detects toxic comments using a fine-tuned DistilBERT model, identifies the intent behind flagged comments, and generates a cleaned paraphrase where applicable. Built for NLP Assignment 1 (Roll No: 23110327).

---

## System Architecture

The system works in three stages:

```
User Input --> Stage 1: DistilBERT Toxicity Classifier
                           |
              (if flagged as toxic)
                           |
                           v
              Stage 2: Intent Detector (Regex + MNLI fallback)
                           |
                           v
              Stage 3: T5 Text Detoxifier --> Clean Output
```

1. Stage 1 runs the input through the fine-tuned DistilBERT model, which outputs a probability for each of the six toxicity labels.
2. Stage 2 identifies the underlying intent (hostile attack, venting, critique, or general).
3. Stage 3 rewrites flagged comments using a regex substitution map or a T5 paraphrase model if the comment is not a direct personal attack.

---

## Key Features

- Multi-label classification: fine-tuned distilbert-base-uncased across 6 non-exclusive toxicity labels
- Focal Loss with gamma=2.0 to handle extreme class imbalance (threat label has a 1:334 ratio)
- Per-label threshold calibration on the dev set using Precision-Recall curve sweep
- Interactive Gradio web app for live testing

---

## Repository Structure

```
.
├── app.py                       # Gradio web UI for testing the full pipeline
├── inspect_app.py               # Diagnostic tool for inspecting model behavior
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── report.md                    # Assignment report
├── thresholds.json              # Calibrated decision thresholds per label
├── gantt_chart.png              # Project timeline chart
├── clean_vs_toxic.png           # EDA plot: clean vs toxic distribution
├── comment_length.png           # EDA plot: comment length histogram
├── label_distribution.png       # EDA plot: per-label counts
├── scripts/
│   ├── run_eda.py               # Generates EDA plots from train.csv
│   ├── evaluate_bias.py         # Measures false positive rate by identity group
│   └── generate_gantt.py        # Generates the Gantt chart
└── src/
    ├── preprocessing.py         # Text cleaning, label list, CSV loading utility
    ├── data_pipeline.py         # Downloads dataset, deduplicates, splits 80/10/10
    ├── train_baseline.py        # TF-IDF + Logistic Regression baseline
    ├── train_llm.py             # DistilBERT fine-tuning with Focal Loss
    ├── train_intent_classifier.py  # Intent classifier fine-tuning (optional)
    ├── train_detoxifier.py      # T5 detoxifier fine-tuning (optional)
    ├── calibrate_thresholds.py  # Finds optimal threshold per label on dev set
    ├── moderation_layer.py      # Connects all three stages into one pipeline
    └── evaluate.py              # Runs evaluation on test set, prints results table
```

---

## Setup Instructions

### 1. Clone and Install

```bash
git clone https://github.com/tanishwanve-git/NLP_assignment1_23110327.git
cd NLP_assignment1_23110327
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset and Model Preparation

**Dataset:** Download the Jigsaw Toxic Comment Classification dataset (train.csv) from Kaggle and place it in the project root. Then run:

```bash
python src/data_pipeline.py
```

This downloads the dataset from HuggingFace (google/jigsaw_toxicity_pred), deduplicates it, runs a leakage audit, and saves train.csv, dev.csv, and test.csv.

**Model Weights:** The fine-tuned DistilBERT model is too large for GitHub (267MB). Download it from Google Drive and extract it into the project root.

- [Download fine_tuned_llm.zip from Google Drive](https://drive.google.com/file/d/1AB97KD9Zss9OQKdTaG1-yVNO8FwNgeO4/view?usp=drive_link)

After extracting, the folder structure should look like:

```
nlp_toxic_comment_project/
└── fine_tuned_llm/
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── model.safetensors
```

### 3. Run the Pipeline Step by Step

```bash
# Step A: Run EDA to generate plots
python scripts/run_eda.py

# Step B: Train the classical baseline
python src/train_baseline.py

# Step C: Fine-tune DistilBERT (skip if using downloaded weights)
python src/train_llm.py

# Step D: Calibrate thresholds on the dev set
python src/calibrate_thresholds.py

# Step E: Evaluate on the test set
python src/evaluate.py
```

### 4. Run the Web App

```bash
python app.py
```

Open http://127.0.0.1:7860 in a browser to test the system interactively.

---

## License and Acknowledgments

Dataset: Jigsaw Toxic Comment Classification Challenge (Kaggle / Google Conversation AI)
