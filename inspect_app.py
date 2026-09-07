"""
inspect_app.py — Batch Evaluation Script (Mac-safe, single-model)
==================================================================
Evaluates the fine-tuned DistilBERT toxicity model against test.csv
using efficient batch inference — no Gradio server, no sentiment
model, no Flan-T5.  Only ONE model is loaded at a time, which
avoids the SIGBUS / memory crash on macOS when multiple large
transformers are resident simultaneously.

Verdict logic mirrors moderation_layer.py Stage 1 thresholds:
  targeted_hostility > 0.35  → Hostile Cyberbullying  (binary 1)
  general_toxicity   > 0.40  → Flagged Profanity       (binary 1)
  else                       → Clean                   (binary 0)

Outputs
-------
  inspection_results.csv          — Per-row predictions + ground-truth
  inspection_metrics_summary.csv  — Binary & multi-label metrics + latency

Usage
-----
  python inspect_app.py                          # full test set
  python inspect_app.py --sample 500            # N-row random sample
  python inspect_app.py --batch-size 64         # tweak batch size
  python inspect_app.py --test-csv path/to.csv  # custom CSV path
"""

import os
import sys
import time
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    classification_report, confusion_matrix,
)

# Add project root to path so src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import LABELS, load_labeled_csv

# ── Environment ───────────────────────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_PATH = "./fine_tuned_llm"

# Mirror thresholds from moderation_layer.py
TARGETED_HOSTILITY_THRESH = 0.30
GENERAL_TOXICITY_THRESH   = 0.35


# ─── Stage-1 verdict (no sentiment model needed for batch eval) ───────────────

def scores_to_verdict(scores: dict) -> tuple[str, int]:
    """
    Replicate moderation_layer Stage-1 decision using only toxicity scores.
    Returns (verdict_str, binary_int)
    """
    # Use lower thresholds for rare classes (threat, identity_hate)
    is_targeted_hostile = False
    if scores.get("threat", 0) > 0.15 or scores.get("identity_hate", 0) > 0.18:
        is_targeted_hostile = True
    elif max(scores.get("insult", 0), scores.get("severe_toxic", 0)) > 0.30:
        is_targeted_hostile = True

    general = max(scores.get("toxic", 0), scores.get("obscene", 0))

    if is_targeted_hostile:
        return "🚨 Hostile Cyberbullying", 1
    elif general > GENERAL_TOXICITY_THRESH:
        # Stage 2 (sentiment) would disambiguate here in live app.
        # For batch eval we conservatively flag as Flagged Profanity (toxic).
        return "⚠️ Flagged Profanity", 1
    else:
        return "✅ Clean", 0


# ─── Metrics printer ─────────────────────────────────────────────────────────

def print_metrics(m: dict) -> None:
    s = "=" * 65
    print(f"\n{s}")
    print("  EVALUATION METRICS SUMMARY")
    print(s)
    print("  Binary (any-toxic vs clean):")
    print(f"    Accuracy           : {m['binary_accuracy']}")
    print(f"    Macro  F1          : {m['binary_macro_f1']}")
    print(f"    Weighted F1        : {m['binary_weighted_f1']}")
    print("\n  Multi-Label (scores @ 0.5 threshold vs ground truth):")
    print(f"    Exact Match Acc    : {m['multilabel_exact_match_accuracy']}")
    print(f"    Macro  F1          : {m['multilabel_macro_f1']}")
    print(f"    Micro  F1          : {m['multilabel_micro_f1']}")
    print(f"    Macro  ROC-AUC     : {m['multilabel_macro_roc_auc']}")
    print(f"    Micro  ROC-AUC     : {m['multilabel_micro_roc_auc']}")
    print("\n  Per-Label F1:")
    for lbl in LABELS:
        print(f"    {lbl:<18} : {m[f'f1_{lbl}']}")
    print("\n  Latency (batch inference, ms/row):")
    print(f"    Mean               : {m['latency_mean_ms']}")
    print(f"    Median             : {m['latency_median_ms']}")
    print(f"    p95                : {m['latency_p95_ms']}")
    print(f"    Max                : {m['latency_max_ms']}")
    print(s)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_inspection(
    test_csv: str = "test.csv",
    sample: int = None,
    batch_size: int = 32,
) -> None:

    # 1. Load data ──────────────────────────────────────────────────────────────
    df = load_labeled_csv(test_csv, require_labels=False)
    if df is None:
        print(f"[ERROR] CSV not found or unreadable: {test_csv}")
        sys.exit(1)

    print(f"\n{'='*65}")
    print("  NLP Toxic Comment Project — Batch Inspection (Mac-safe)")
    print(f"{'='*65}")

    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)
        print(f"  Mode       : SAMPLE  ({len(df)} rows)")
    else:
        print(f"  Mode       : FULL    ({len(df)} rows)")

    print(f"  Batch size : {batch_size}")
    print(f"  Model      : {MODEL_PATH}")

    has_gt = all(lbl in df.columns for lbl in LABELS)
    if not has_gt:
        print("  [WARNING] Ground-truth columns missing — metrics will be skipped.")

    # 2. Load ONE model only (avoids dual-model SIGBUS on Mac) ─────────────────
    print(f"\n{'─'*65}")
    print("  Loading tokenizer + fine-tuned DistilBERT …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    # CPU only — MPS skipped for stability on macOS
    device = torch.device("cpu")
    model.to(device)
    print(f"  Model loaded on {device}. Parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")
    print(f"{'─'*65}\n")

    # 3. Batched inference ──────────────────────────────────────────────────────
    texts  = df["comment_text"].tolist()
    total  = len(texts)
    n_batches = (total + batch_size - 1) // batch_size

    all_probs    = []   # (total, 6) sigmoid probabilities
    latencies_ms = []   # per-batch ms, later averaged per row

    print(f"  Running batched inference ({n_batches} batches of ≤{batch_size}) …\n")

    for b_idx in range(n_batches):
        start  = b_idx * batch_size
        end    = min(start + batch_size, total)
        batch  = texts[start:end]

        t0 = time.perf_counter()
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**enc).logits
            probs  = torch.sigmoid(logits).cpu().numpy()   # (batch, 6)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_row_ms = elapsed_ms / len(batch)
        latencies_ms.extend([round(per_row_ms, 2)] * len(batch))
        all_probs.extend(probs.tolist())

        pct = round(end / total * 100, 1)
        print(f"  [{pct:5.1f}%] batch {b_idx+1:>4}/{n_batches}  "
              f"rows {start+1:>6}–{end:<6}  "
              f"{round(per_row_ms, 1)} ms/row")

    all_probs = np.array(all_probs)   # (total, 6)

    # 4. Build per-row records ──────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Building per-row result records …")

    records = []
    for i in range(total):
        row    = df.iloc[i]
        scores = {lbl: float(all_probs[i, j]) for j, lbl in enumerate(LABELS)}
        verdict, pred_binary = scores_to_verdict(scores)

        rec = {
            "row_index"    : i,
            "id"           : row.get("id", i),
            "comment_text" : str(row["comment_text"])[:300],
            "verdict"      : verdict,
            "pred_binary"  : pred_binary,
            "latency_ms"   : latencies_ms[i],
        }
        for lbl in LABELS:
            rec[f"score_{lbl}"] = round(scores[lbl], 4)

        if has_gt:
            for lbl in LABELS:
                rec[f"gt_{lbl}"] = int(row[lbl])
            rec["gt_binary"] = int(max(int(row[lbl]) for lbl in LABELS))

        records.append(rec)

    results_df = pd.DataFrame(records)

    # 5. Save per-row CSV ───────────────────────────────────────────────────────
    out_rows = "inspection_results.csv"
    results_df.to_csv(out_rows, index=False)
    print(f"  ✓ Per-row results  → {out_rows}  ({len(results_df)} rows)")

    # 6. Metrics ────────────────────────────────────────────────────────────────
    if has_gt:
        print(f"\n{'─'*65}")
        print("  Computing evaluation metrics …")

        y_true_bin = results_df["gt_binary"].values
        y_pred_bin = results_df["pred_binary"].values
        y_scores   = all_probs                          # (N, 6) float
        y_pred_ml  = (y_scores > 0.5).astype(int)
        y_true_ml  = df[LABELS].values                 # (N, 6) int

        lat = np.array(latencies_ms)

        m = {}

        # Binary
        m["binary_accuracy"]    = round(accuracy_score(y_true_bin, y_pred_bin), 4)
        m["binary_macro_f1"]    = round(f1_score(y_true_bin, y_pred_bin, average="macro",    zero_division=0), 4)
        m["binary_weighted_f1"] = round(f1_score(y_true_bin, y_pred_bin, average="weighted", zero_division=0), 4)

        # Multi-label
        m["multilabel_exact_match_accuracy"] = round(accuracy_score(y_true_ml, y_pred_ml), 4)
        m["multilabel_macro_f1"] = round(f1_score(y_true_ml, y_pred_ml, average="macro",  zero_division=0), 4)
        m["multilabel_micro_f1"] = round(f1_score(y_true_ml, y_pred_ml, average="micro",  zero_division=0), 4)
        try:
            m["multilabel_macro_roc_auc"] = round(roc_auc_score(y_true_ml, y_scores, average="macro"), 4)
            m["multilabel_micro_roc_auc"] = round(roc_auc_score(y_true_ml, y_scores, average="micro"), 4)
        except ValueError:
            m["multilabel_macro_roc_auc"] = "N/A"
            m["multilabel_micro_roc_auc"] = "N/A"

        # Per-label F1
        for j, lbl in enumerate(LABELS):
            m[f"f1_{lbl}"] = round(f1_score(y_true_ml[:, j], y_pred_ml[:, j], zero_division=0), 4)

        # Latency
        m["latency_mean_ms"]   = round(float(np.mean(lat)), 2)
        m["latency_median_ms"] = round(float(np.median(lat)), 2)
        m["latency_p95_ms"]    = round(float(np.percentile(lat, 95)), 2)
        m["latency_max_ms"]    = round(float(np.max(lat)), 2)

        print_metrics(m)

        # Confusion matrix
        cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        print(f"\n  Binary Confusion Matrix:")
        print(f"    TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
        print(f"    FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")

        # Per-label classification report
        print(f"\n  Per-Label Classification Report:")
        print(classification_report(y_true_ml, y_pred_ml, target_names=LABELS, zero_division=0))

        # Save metrics CSV
        out_metrics = "inspection_metrics_summary.csv"
        pd.DataFrame([m]).to_csv(out_metrics, index=False)
        print(f"  ✓ Metrics summary  → {out_metrics}")

    else:
        print("\n  [INFO] No ground-truth labels — metrics skipped.")
        print(f"  Verdict counts:\n{results_df['verdict'].value_counts().to_string()}")

    print(f"\n  Done. {total} rows processed.\n{'='*65}\n")

    # 7. Intent Model Evaluation (if fine_tuned_intent_model exists) ────────────
    INTENT_MODEL_PATH = "./fine_tuned_intent_model"
    if not os.path.isdir(INTENT_MODEL_PATH):
        print("  [INFO] fine_tuned_intent_model/ not found — skipping intent evaluation.")
        return

    print(f"\n{'='*65}")
    print("  INTENT MODEL EVALUATION")
    print(f"{'='*65}")

    from transformers import pipeline as hf_pipeline

    intent_sample_size = min(500, len(df))
    df_intent = df.sample(n=intent_sample_size, random_state=42).reset_index(drop=True)

    print(f"  Loading intent classifier from {INTENT_MODEL_PATH} ...")
    intent_clf = hf_pipeline(
        "text-classification",
        model=INTENT_MODEL_PATH,
        tokenizer=INTENT_MODEL_PATH,
        device=-1,
        truncation=True,
        max_length=128,
    )

    print(f"  Running intent inference on {intent_sample_size} rows ...")
    intent_texts  = df_intent["comment_text"].fillna(" ").tolist()
    intent_preds  = []
    intent_scores = []

    for text in intent_texts:
        res = intent_clf(text)[0]
        intent_preds.append(res["label"])
        intent_scores.append(round(res["score"], 4))

    df_intent = df_intent.copy()
    df_intent["predicted_intent"]       = intent_preds
    df_intent["intent_confidence"]      = intent_scores

    # If ground-truth toxicity columns exist, show intent breakdown vs toxicity
    print(f"\n  Intent Distribution ({intent_sample_size} rows):")
    dist = pd.Series(intent_preds).value_counts()
    for intent, count in dist.items():
        pct = round(count / intent_sample_size * 100, 1)
        print(f"    {intent:<25} : {count:>5} ({pct}%)")

    if has_gt:
        df_intent["gt_binary"] = df_intent[LABELS].max(axis=1)
        print(f"\n  Intent vs Toxicity Ground-Truth:")
        print(f"  {'Intent':<25} {'Total':>6} {'Toxic%':>8} {'Clean%':>8}")
        print(f"  {'-'*55}")
        for intent in dist.index:
            subset = df_intent[df_intent["predicted_intent"] == intent]
            toxic_pct = round(subset["gt_binary"].mean() * 100, 1)
            clean_pct = round(100 - toxic_pct, 1)
            print(f"  {intent:<25} {len(subset):>6} {toxic_pct:>7}% {clean_pct:>7}%")

    # Save intent results CSV
    intent_out = "intent_evaluation_results.csv"
    cols = ["id", "comment_text", "predicted_intent", "intent_confidence"]
    if has_gt:
        cols += LABELS + ["gt_binary"]
    df_intent[cols].to_csv(intent_out, index=False)
    print(f"\n  ✓ Intent results → {intent_out}  ({len(df_intent)} rows)")
    print(f"{'='*65}\n")



# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch-evaluate the fine-tuned toxicity model on test.csv (Mac-safe)"
    )
    parser.add_argument("--test-csv",   default="test.csv", help="Path to test CSV")
    parser.add_argument("--sample",     type=int, default=None, help="Random N-row subset")
    parser.add_argument("--batch-size", type=int, default=32,   help="Inference batch size (default 32)")
    args = parser.parse_args()
    run_inspection(test_csv=args.test_csv, sample=args.sample, batch_size=args.batch_size)
