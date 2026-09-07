"""
evaluate_bias.py — Bias evaluation on CivilComments
=====================================================
Tests the fine-tuned toxicity model on a sample of the CivilComments dataset
to measure the False Positive Rate (FPR) across different demographic groups.

Usage:
  python scripts/evaluate_bias.py
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.moderation_layer import ContextualModerator

def main():
    print("Loading test.csv for bias evaluation...")
    try:
        df = pd.read_csv("test.csv")
    except Exception as e:
        print(f"Error loading test.csv: {e}")
        return
    
    # Filter for non-toxic comments only (we want to measure False Positives)
    # If it has ground truth labels, we ensure they are all 0
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    has_gt = all(lbl in df.columns for lbl in labels)
    if has_gt:
        df['is_toxic_gt'] = df[labels].max(axis=1)
        benign_df = df[df['is_toxic_gt'] == 0].copy()
    else:
        # If no labels, we just run on everything and accept it's an estimate
        benign_df = df.copy()
        
    print(f"Total benign comments for testing: {len(benign_df)}")

    # Define identity terms to test
    identities = {
        "black": ["black"],
        "muslim": ["muslim", "islam"],
        "jewish": ["jew", "jewish"],
        "gay": ["gay", "homosexual", "lesbian"],
        "trans": ["trans", "transgender"]
    }

    # Load model
    print("\nLoading ContextualModerator (Stage 1 only for batch eval)...")
    model_path = "./fine_tuned_llm"
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found.")
        return
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    # We will use the same threshold logic from our new system
    TARGETED_HOSTILITY_THRESH = 0.30
    GENERAL_TOXICITY_THRESH = 0.40
    LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def is_flagged(probs):
        scores = {lbl: float(probs[j]) for j, lbl in enumerate(LABELS)}
        is_targeted = False
        if scores.get("threat", 0) > 0.30 or scores.get("identity_hate", 0) > 0.30:
            is_targeted = True
        elif max(scores.get("insult", 0), scores.get("severe_toxic", 0)) > 0.35:
            is_targeted = True
        
        general = max(scores.get("toxic", 0), scores.get("obscene", 0))
        return is_targeted or (general > 0.40)

    print("\nEvaluating Bias (False Positive Rate)...")
    results = []

    # Helper for batched inference
    def evaluate_group(group_name, texts):
        if not texts:
            return 0.0, 0
            
        batch_size = 32
        flagged_count = 0
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**enc).logits
                probs = torch.sigmoid(logits).cpu().numpy()
            
            for p in probs:
                if is_flagged(p):
                    flagged_count += 1
                    
        return (flagged_count / total) * 100, total

    # 1. Baseline FPR (all benign comments)
    print("Computing baseline FPR...")
    baseline_fpr, baseline_count = evaluate_group("baseline", benign_df['comment_text'].tolist())
    
    print(f"\n{'Group':<15} | {'FPR (%)':<10} | {'N Samples'}")
    print("-" * 45)
    print(f"{'Baseline (All)':<15} | {baseline_fpr:<10.2f} | {baseline_count}")

    # 2. Group FPR
    for group, keywords in identities.items():
        # Find comments containing any of the keywords
        pattern = '|'.join(r'\b{}\b'.format(k) for k in keywords)
        group_df = benign_df[benign_df['comment_text'].str.contains(pattern, case=False, regex=True, na=False)]
        
        fpr, count = evaluate_group(group, group_df['comment_text'].tolist())
        print(f"{group:<15} | {fpr:<10.2f} | {count}")

if __name__ == "__main__":
    main()
