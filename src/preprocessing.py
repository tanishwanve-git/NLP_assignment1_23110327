"""
preprocessing.py
Shared utilities used across the project.
Provides the label list, a text cleaning function, and a CSV loader.
"""

import unicodedata
import pandas as pd

# The six toxicity labels in the Jigsaw dataset, in a fixed order
LABELS: list = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def clean_text(text) -> str:
    """
    Basic text normalization.
    Handles None/NaN values, strips whitespace, normalizes unicode,
    and collapses multiple spaces into one.
    """
    if not isinstance(text, str):
        return " "
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text.strip() or " "


def load_labeled_csv(path: str, require_labels: bool = True):
    """
    Load a CSV file containing comment_text and optionally the 6 label columns.
    Applies clean_text to all comments. Returns None if file is missing.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"[preprocessing] File not found: {path}")
        return None
    except Exception as e:
        print(f"[preprocessing] Failed to load {path}: {e}")
        return None

    if "comment_text" not in df.columns:
        print(f"[preprocessing] 'comment_text' column missing in {path}")
        return None

    df["comment_text"] = df["comment_text"].apply(clean_text)

    has_labels = all(lbl in df.columns for lbl in LABELS)
    if require_labels and not has_labels:
        missing = [l for l in LABELS if l not in df.columns]
        print(f"[preprocessing] Warning: label columns missing: {missing}")
    if has_labels:
        for lbl in LABELS:
            df[lbl] = df[lbl].fillna(0).astype(int)

    return df


def compute_binary_label(df):
    """Returns 1 if any toxicity label is 1, else 0."""
    return df[LABELS].max(axis=1).astype(int)


def report_class_distribution(df, split_name: str = "") -> None:
    """Prints per-label positive count and imbalance ratio for a given split."""
    label = f" [{split_name}]" if split_name else ""
    total = len(df)
    print(f"\n{'─'*55}")
    print(f"  Class Distribution{label}  (N={total:,})")
    print(f"  {'Label':<18} {'Pos':>6} {'Neg':>6} {'Pos%':>6} {'Ratio (neg:pos)':>16}")
    print(f"  {'─'*54}")
    for lbl in LABELS:
        if lbl not in df.columns:
            continue
        pos = int(df[lbl].sum())
        neg = total - pos
        pct = f"{pos / total * 100:.2f}%"
        ratio = f"1:{neg // max(pos, 1)}"
        print(f"  {lbl:<18} {pos:>6} {neg:>6} {pct:>6} {ratio:>16}")
    print(f"{'─'*55}")
