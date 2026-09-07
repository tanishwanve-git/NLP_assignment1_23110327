import pandas as pd
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import os
import sys

# Add project root to sys.path so we can import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import LABELS, load_labeled_csv

import json

def evaluate_models(test_path="test.csv"):
    test_df = load_labeled_csv(test_path, require_labels=False)
    if test_df is None:
        print("Data files not found. Run src/data_pipeline.py first.")
        return

    print("Loading test data...")
    X_test_raw = test_df["comment_text"]
    has_labels = all(lbl in test_df.columns for lbl in LABELS)
    if not has_labels:
        print("Error: test dataset missing ground truth labels.")
        return
    
    y_test = test_df[LABELS].values

    # Load thresholds if available
    thresholds = {lbl: 0.5 for lbl in LABELS}
    if os.path.exists("thresholds.json"):
        with open("thresholds.json", "r") as f:
            thresholds = json.load(f)
        print(f"Loaded calibrated thresholds: {thresholds}")

    results = []
    per_label_f1 = {}

    # 1. Evaluate Baseline (TF-IDF + LR)
    print("\n--- Evaluating Classical Baseline ---")
    try:
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        lr_model = joblib.load('baseline_lr_model.pkl')
        
        X_test_tfidf = vectorizer.transform(X_test_raw)
        y_pred_lr = lr_model.predict(X_test_tfidf)
        y_pred_proba_lr = lr_model.predict_proba(X_test_tfidf)
        
        results.append({
            "Model": "TF-IDF + Logistic Regression",
            "Macro F1": round(f1_score(y_test, y_pred_lr, average='macro'), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_pred_proba_lr, average='macro'), 4),
            "Exact Match Accuracy": round(accuracy_score(y_test, y_pred_lr), 4)
        })
        
        lr_f1s = [f1_score(y_test[:, j], y_pred_lr[:, j], zero_division=0) for j in range(len(LABELS))]
        per_label_f1["Baseline LR"] = [round(f, 4) for f in lr_f1s]
        print("Baseline evaluation complete.")
    except FileNotFoundError:
        print("Baseline model not found. Run src/train_baseline.py first.")

    # 2. Evaluate LLM (DistilBERT)
    print("\n--- Evaluating Fine-Tuned LLM ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_llm")
        model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_llm")
        model.eval()

        batch_size = 32
        all_probs = []
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        print(f"Running inference on {device}...")
        for i in range(0, len(X_test_raw), batch_size):
            batch_texts = X_test_raw.iloc[i:i+batch_size].tolist()
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                
        all_probs = np.array(all_probs)
        
        # A) Default 0.5 Threshold
        y_pred_llm_def = (all_probs > 0.5).astype(int)
        results.append({
            "Model": "DistilBERT (Default tau=0.5)",
            "Macro F1": round(f1_score(y_test, y_pred_llm_def, average='macro'), 4),
            "ROC-AUC": round(roc_auc_score(y_test, all_probs, average='macro'), 4),
            "Exact Match Accuracy": round(accuracy_score(y_test, y_pred_llm_def), 4)
        })
        llm_def_f1s = [f1_score(y_test[:, j], y_pred_llm_def[:, j], zero_division=0) for j in range(len(LABELS))]
        per_label_f1["DistilBERT (tau=0.5)"] = [round(f, 4) for f in llm_def_f1s]

        # B) Calibrated Thresholds
        t_vec = np.array([thresholds.get(lbl, 0.5) for lbl in LABELS])
        y_pred_llm_cal = (all_probs > t_vec).astype(int)
        results.append({
            "Model": "DistilBERT (Calibrated Thresholds)",
            "Macro F1": round(f1_score(y_test, y_pred_llm_cal, average='macro'), 4),
            "ROC-AUC": round(roc_auc_score(y_test, all_probs, average='macro'), 4),
            "Exact Match Accuracy": round(accuracy_score(y_test, y_pred_llm_cal), 4)
        })
        llm_cal_f1s = [f1_score(y_test[:, j], y_pred_llm_cal[:, j], zero_division=0) for j in range(len(LABELS))]
        per_label_f1["DistilBERT (Calibrated)"] = [round(f, 4) for f in llm_cal_f1s]
        
        print("LLM evaluation complete.")
    except Exception as e:
        print(f"LLM model error: {e}. Run src/train_llm.py first.")

    # 3. Print Results Tables
    if results:
        print("\n================ GLOBAL PERFORMANCE TABLE ================")
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        print("==========================================================")
        results_df.to_csv("final_results.csv", index=False)

    if per_label_f1:
        print("\n================ PER-LABEL F1 SCORES TABLE ================")
        per_label_df = pd.DataFrame(per_label_f1, index=LABELS)
        print(per_label_df.to_string())
        print("==========================================================")
        per_label_df.to_csv("per_label_f1.csv")

if __name__ == "__main__":
    evaluate_models()

