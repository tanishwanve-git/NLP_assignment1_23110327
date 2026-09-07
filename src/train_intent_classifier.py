"""
Fine-tune a DistilBERT Intent Classifier on:
  1. CLINC150 (Hugging Face) — 150 intent categories remapped to our 4 classes
  2. GameTox  (GitHub: shucoll/GameTox) — chat utterances with toxic intent labels
  3. Curated fallback examples (always included to guarantee Hostile_Attack coverage)

Output: ./fine_tuned_intent_model  (used by IntentDetector as a fast local classifier)

Tested on Colab with transformers>=4.40, >=4.41, >=4.45 (version-safe).
"""

import os
import inspect
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# ─── Label Definitions ────────────────────────────────────────────────────────
INTENT_LABELS = [
    "Critique_Feedback",    # 0
    "Emotional_Venting",    # 1
    "Hostile_Attack",       # 2
    "General_Discussion",   # 3
]
LABEL2ID = {l: i for i, l in enumerate(INTENT_LABELS)}
ID2LABEL = {i: l for i, l in enumerate(INTENT_LABELS)}

# ─── CLINC150 → 4-class mapping ───────────────────────────────────────────────
CLINC150_REMAP = {
    "bill_balance":              "Critique_Feedback",
    "bill_due":                  "Critique_Feedback",
    "account_blocked":           "Critique_Feedback",
    "card_declined":             "Critique_Feedback",
    "freeze_account":            "Critique_Feedback",
    "report_lost_card":          "Critique_Feedback",
    "damaged_card":              "Critique_Feedback",
    "transaction_dispute":       "Critique_Feedback",
    "cancel_reservation":        "Critique_Feedback",
    "confirm_reservation":       "Critique_Feedback",
    "flight_status":             "Critique_Feedback",
    "lost_luggage":              "Critique_Feedback",
    "travel_alert":              "Critique_Feedback",
    "symptom_search":            "Critique_Feedback",
    "find_pharmacy":             "Critique_Feedback",
    "shopping_list":             "Critique_Feedback",
    "pto_request":               "Critique_Feedback",
    "report_fraud":              "Critique_Feedback",
    "service_fee":               "Critique_Feedback",
    "replacement_card_duration": "Critique_Feedback",
    "interest_rate":             "Critique_Feedback",
    "credit_limit_change":       "Critique_Feedback",
    "credit_score":              "Critique_Feedback",
    "rewards_balance":           "Critique_Feedback",
    "pay_bill":                  "Critique_Feedback",
    "redeem_rewards":            "Critique_Feedback",
    "repeat_phrase":             "Emotional_Venting",
    "meaning_of_life":           "Emotional_Venting",
    "tell_joke":                 "Emotional_Venting",
    "joke":                      "Emotional_Venting",
    "oos":                       "Emotional_Venting",
}


# ─── Dataset Loaders ──────────────────────────────────────────────────────────

def load_clinc150():
    """Load CLINC150 from Hugging Face, try multiple dataset IDs."""
    print("\n[1/3] Loading CLINC150 from Hugging Face...")
    from datasets import load_dataset

    raw = None
    for dataset_id in ["clinc/clinc_oos", "clinc_oos"]:
        try:
            print(f"  Trying: '{dataset_id}'...")
            raw = load_dataset(dataset_id, "plus")
            print(f"  ✓ Loaded '{dataset_id}'")
            break
        except Exception as e:
            print(f"  ✗ Failed '{dataset_id}': {e}")

    if raw is None:
        print("  ✗ CLINC150 unavailable. Using fallback only.")
        return pd.DataFrame(columns=["text", "label"])

    rows = []
    for split_name, split_data in raw.items():
        features = split_data.features
        # Safely get label name — handle both ClassLabel and Value types
        for example in split_data:
            text = str(example["text"]).strip()
            try:
                intent_str = features["intent"].int2str(int(example["intent"]))
            except Exception:
                intent_str = str(example["intent"])
            our_label = CLINC150_REMAP.get(intent_str, "General_Discussion")
            if len(text) > 3:
                rows.append({"text": text, "label": LABEL2ID[our_label]})

    df = pd.DataFrame(rows)
    print(f"  ✓ {len(df):,} examples. Distribution:\n{df['label'].map(ID2LABEL).value_counts().to_string()}\n")
    return df


def load_hate_speech():
    """
    Load hate_speech_offensive dataset from Hugging Face.
    24,000+ labeled tweets with 3 classes:
      0 = hate speech      → Hostile_Attack
      1 = offensive lang   → Emotional_Venting
      2 = neither          → General_Discussion
    Repo: https://huggingface.co/datasets/tdavidson/hate_speech_offensive
    """
    print("[2/3] Loading hate_speech_offensive from Hugging Face...")
    from datasets import load_dataset
    try:
        raw = load_dataset("tdavidson/hate_speech_offensive", split="train")
        print(f"  ✓ Loaded {len(raw):,} examples.")
    except Exception as e:
        print(f"  ✗ Failed to load hate_speech_offensive: {e}. Using curated fallback.")
        return _curated_fallback()

    rows = []
    for example in raw:
        text = str(example.get("tweet", "")).strip()
        cls  = int(example.get("class", 2))   # 0=hate, 1=offensive, 2=neither
        if cls == 0:
            our_label = "Hostile_Attack"
        elif cls == 1:
            our_label = "Emotional_Venting"
        else:
            our_label = "General_Discussion"
        if len(text) > 3:
            rows.append({"text": text, "label": LABEL2ID[our_label]})

    df = pd.DataFrame(rows)
    print(f"  ✓ {len(df):,} GameTox examples. Distribution:\n{df['label'].map(ID2LABEL).value_counts().to_string()}\n")
    return df


def _curated_fallback():
    """Curated hostile/venting/critique examples — always included to guarantee class coverage."""
    print("  [Fallback] Using curated hostile/venting/critique examples...")
    data = [
        # Hostile Attack (most important — not in CLINC150)
        {"text": "You are a complete idiot and a total loser",              "label": LABEL2ID["Hostile_Attack"]},
        {"text": "Shut the fuck up you ugly worthless piece of trash",      "label": LABEL2ID["Hostile_Attack"]},
        {"text": "I hate you and everything about you, go die",             "label": LABEL2ID["Hostile_Attack"]},
        {"text": "You stupid moron, you don't deserve to exist",            "label": LABEL2ID["Hostile_Attack"]},
        {"text": "Screw you and your whole family you racist coward",       "label": LABEL2ID["Hostile_Attack"]},
        {"text": "You are so dumb it is actually embarrassing for you",     "label": LABEL2ID["Hostile_Attack"]},
        {"text": "I will find you and make you regret saying that",         "label": LABEL2ID["Hostile_Attack"]},
        {"text": "Go kill yourself nobody wants you here",                  "label": LABEL2ID["Hostile_Attack"]},
        {"text": "You filthy piece of garbage get out of here",             "label": LABEL2ID["Hostile_Attack"]},
        {"text": "Your kind does not belong in this country",               "label": LABEL2ID["Hostile_Attack"]},
        {"text": "Get out of my sight you disgusting person",               "label": LABEL2ID["Hostile_Attack"]},
        {"text": "Nobody likes you, you are worthless trash",               "label": LABEL2ID["Hostile_Attack"]},
        {"text": "I despise everything you stand for",                      "label": LABEL2ID["Hostile_Attack"]},
        {"text": "You are the dumbest person I have ever seen",             "label": LABEL2ID["Hostile_Attack"]},
        {"text": "Stop breathing you waste of oxygen",                      "label": LABEL2ID["Hostile_Attack"]},
        # Emotional Venting
        {"text": "Ugh I am so damn frustrated with everything today",       "label": LABEL2ID["Emotional_Venting"]},
        {"text": "What the hell, this is so incredibly annoying!",          "label": LABEL2ID["Emotional_Venting"]},
        {"text": "I am so freaking sick and tired of all this crap",        "label": LABEL2ID["Emotional_Venting"]},
        {"text": "Holy crap this traffic is absolutely insane today",       "label": LABEL2ID["Emotional_Venting"]},
        {"text": "Damn it, I missed my alarm again this morning!",          "label": LABEL2ID["Emotional_Venting"]},
        {"text": "Oh for goodness sake why does this always happen to me",  "label": LABEL2ID["Emotional_Venting"]},
        {"text": "What a complete disaster this whole day has been",        "label": LABEL2ID["Emotional_Venting"]},
        # Critique / Feedback
        {"text": "The app crashes every single time I try to log in",       "label": LABEL2ID["Critique_Feedback"]},
        {"text": "This damn login page is totally broken and won't load",   "label": LABEL2ID["Critique_Feedback"]},
        {"text": "Your customer service is absolutely terrible and useless","label": LABEL2ID["Critique_Feedback"]},
        {"text": "The checkout button does not work on any browser",        "label": LABEL2ID["Critique_Feedback"]},
        {"text": "Why is your website so incredibly slow and unusable?",    "label": LABEL2ID["Critique_Feedback"]},
        {"text": "I have been charged twice and no one will respond",       "label": LABEL2ID["Critique_Feedback"]},
        # General Discussion
        {"text": "Can you tell me what time the store closes today?",       "label": LABEL2ID["General_Discussion"]},
        {"text": "I think the new update has some really nice features",    "label": LABEL2ID["General_Discussion"]},
        {"text": "How do I reset my account password on this platform?",    "label": LABEL2ID["General_Discussion"]},
        {"text": "I really enjoyed reading that article you published",     "label": LABEL2ID["General_Discussion"]},
    ]
    df = pd.DataFrame(data)
    print(f"  ✓ {len(df)} curated fallback examples.\n")
    return df


# ─── Balancing ────────────────────────────────────────────────────────────────

def balance_dataset(df, max_per_class=3000):
    """Oversample minority classes to balance training data."""
    parts = []
    for label_id in range(len(INTENT_LABELS)):
        subset = df[df["label"] == label_id]
        if len(subset) == 0:
            print(f"  ⚠ No examples found for class {ID2LABEL[label_id]}!")
            continue
        if len(subset) < max_per_class:
            subset = subset.sample(n=max_per_class, replace=True, random_state=42)
        else:
            subset = subset.sample(n=max_per_class, random_state=42)
        parts.append(subset)
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42)


# ─── Training ─────────────────────────────────────────────────────────────────

def train_intent_classifier(
    output_dir="./fine_tuned_intent_model",
    base_model="distilbert-base-uncased",
    epochs=4,
    batch_size=16,
    max_per_class=3000,
):
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
    )

    print("=" * 60)
    print(" Intent Classifier Fine-Tuning")
    print(f" Base Model : {base_model}")
    print(f" Epochs     : {epochs}   Batch: {batch_size}   Max/class: {max_per_class}")
    print("=" * 60)

    # ── 1. Load & merge datasets ───────────────────────────────────────────────
    df_clinc   = load_clinc150()
    df_game    = load_hate_speech()
    df_curated = _curated_fallback()   # always include hostile examples

    df_all = pd.concat([df_clinc, df_game, df_curated], ignore_index=True)
    df_all = df_all.dropna(subset=["text", "label"])
    df_all["text"]  = df_all["text"].astype(str).str.strip()
    df_all["label"] = df_all["label"].astype(int)
    df_all = df_all[df_all["text"].str.len() > 3]

    print(f"\n[3/3] Combined: {len(df_all):,} examples")
    print(f"  Before balancing:\n{df_all['label'].map(ID2LABEL).value_counts().to_string()}\n")

    # ── 2. Balance ─────────────────────────────────────────────────────────────
    df_balanced = balance_dataset(df_all, max_per_class=max_per_class)
    print(f"  After balancing:\n{df_balanced['label'].map(ID2LABEL).value_counts().to_string()}\n")

    # ── 3. Split ───────────────────────────────────────────────────────────────
    train_df, val_df = train_test_split(
        df_balanced, test_size=0.1, random_state=42, stratify=df_balanced["label"]
    )
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))

    # ── 4. Tokenize ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128, padding=False)

    cols_to_remove = [c for c in ["text", "__index_level_0__"] if c in train_ds.column_names]
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
    cols_to_remove = [c for c in ["text", "__index_level_0__"] if c in val_ds.column_names]
    val_ds   = val_ds.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)

    # ── 5. Model ───────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(INTENT_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # ── 6. TrainingArguments (version-safe) ────────────────────────────────────
    use_gpu = torch.cuda.is_available()
    ta_params = inspect.signature(TrainingArguments.__init__).parameters

    ta_kwargs = {
        "output_dir":                    output_dir,
        "num_train_epochs":              epochs,
        "per_device_train_batch_size":   batch_size,
        "per_device_eval_batch_size":    batch_size,
        "save_strategy":                 "epoch",
        "load_best_model_at_end":        True,
        "metric_for_best_model":         "eval_loss",
        "logging_steps":                 50,
        "learning_rate":                 3e-5,
        "weight_decay":                  0.01,
        "fp16":                          use_gpu,
        "report_to":                     "none",
    }

    # eval_strategy (>=4.41) vs evaluation_strategy (<4.41)
    if "eval_strategy" in ta_params:
        ta_kwargs["eval_strategy"] = "epoch"
    else:
        ta_kwargs["evaluation_strategy"] = "epoch"

    # warmup_ratio might not exist in very old versions
    if "warmup_ratio" in ta_params:
        ta_kwargs["warmup_ratio"] = 0.1

    training_args = TrainingArguments(**ta_kwargs)

    # ── 7. Trainer (version-safe) ──────────────────────────────────────────────
    t_params = inspect.signature(Trainer.__init__).parameters

    trainer_kwargs = {
        "model":         model,
        "args":          training_args,
        "train_dataset": train_ds,
        "eval_dataset":  val_ds,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
    }
    # processing_class (>=4.45) vs tokenizer (<4.45)
    if "processing_class" in t_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    # ── 8. Train ───────────────────────────────────────────────────────────────
    print(f"Starting training on {'GPU ✓' if use_gpu else 'CPU (slow)'}...")
    trainer.train()

    # ── 9. Save ────────────────────────────────────────────────────────────────
    print(f"\n✓ Saving model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✓ Training complete! Download fine_tuned_intent_model/ and place in project root.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",    default="./fine_tuned_intent_model")
    parser.add_argument("--base_model",    default="distilbert-base-uncased")
    parser.add_argument("--epochs",        default=4,    type=int)
    parser.add_argument("--batch_size",    default=16,   type=int)
    parser.add_argument("--max_per_class", default=3000, type=int)
    args = parser.parse_args()
    train_intent_classifier(**vars(args))
