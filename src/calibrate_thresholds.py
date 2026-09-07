import pandas as pd
import numpy as np
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score
import sys
import os

# Add project root to sys.path so we can import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import LABELS, load_labeled_csv

def calibrate_thresholds(model_path="./fine_tuned_llm", dev_path="dev.csv"):
    print("Loading Dev Data...")
    df = load_labeled_csv(dev_path)
    if df is None:
        print("Error: dev.csv not found.")
        sys.exit(1)

    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"Running inference on {device}...")
    texts = df["comment_text"].tolist()
    y_true = df[LABELS].values
    
    batch_size = 32
    all_probs = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            all_probs.extend(probs)
            
    all_probs = np.array(all_probs)
    
    print("\nOptimizing Thresholds...")
    best_thresholds = {}
    
    for j, label in enumerate(LABELS):
        true_labels = y_true[:, j]
        label_probs = all_probs[:, j]
        
        best_t = 0.5
        best_f1 = 0.0
        
        for t in np.arange(0.05, 0.95, 0.05):
            preds = (label_probs > t).astype(int)
            f1 = f1_score(true_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
                
        best_thresholds[label] = round(float(best_t), 2)
        print(f"  {label:<15}: Optimal Threshold = {best_t:.2f} (F1 = {best_f1:.4f})")
        
    out_file = "thresholds.json"
    with open(out_file, "w") as f:
        json.dump(best_thresholds, f, indent=4)
        
    print(f"\nOptimal thresholds saved to {out_file}")

if __name__ == "__main__":
    calibrate_thresholds()
