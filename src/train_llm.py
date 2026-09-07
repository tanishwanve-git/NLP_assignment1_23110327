import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import os
import sys

# Add project root to sys.path so we can import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import LABELS, load_labeled_csv

# Custom PyTorch Dataset
class ToxicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Focal Loss Implementation
        gamma = 2.0
        alpha = self.class_weights.to(self.args.device) if self.class_weights is not None else 1.0
        
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.view(-1, self.model.config.num_labels), 
            labels.float().view(-1, self.model.config.num_labels), 
            reduction='none'
        )
        
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        loss = focal_loss.mean()
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # Sigmoid to get probabilities since this is multi-label classification
    probs = 1 / (1 + np.exp(-logits)) 
    predictions = (probs > 0.5).astype(int)
    
    metrics = {}
    metrics['macro_f1'] = f1_score(labels, predictions, average='macro', zero_division=0)
    try:
        metrics['roc_auc'] = roc_auc_score(labels, probs, average='macro')
    except ValueError:
        metrics['roc_auc'] = 0.0

    # Per-label metrics
    from src.preprocessing import LABELS
    for i, label in enumerate(LABELS):
        metrics[f'f1_{label}'] = f1_score(labels[:, i], predictions[:, i], zero_division=0)

    return metrics

def train_llm(model_name="distilbert-base-uncased", sample_size=None):
    """
    Fine-tunes a small LLM on the toxic comment dataset.
    sample_size: Int. If set, trains on a small subset for quick debugging.
    """
    print(f"Loading data for LLM fine-tuning ({model_name})...")
    train_df = load_labeled_csv("train.csv")
    dev_df   = load_labeled_csv("dev.csv")
    if train_df is None or dev_df is None:
        print("Data files not found. Please run src/data_pipeline.py first.")
        return

    # For fast debugging
    if sample_size:
        print(f"USING DEBUG MODE: Training on only {sample_size} samples.")
        train_df = train_df.sample(sample_size, random_state=42)
        dev_df = dev_df.sample(int(sample_size * 0.2), random_state=42)
    
    print("Tokenizing texts...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_encodings = tokenizer(train_df["comment_text"].tolist(), truncation=True, padding=True, max_length=128)
    dev_encodings   = tokenizer(dev_df["comment_text"].tolist(),   truncation=True, padding=True, max_length=128)

    train_labels = train_df[LABELS].values.tolist()
    dev_labels   = dev_df[LABELS].values.tolist()
    
    train_dataset = ToxicDataset(train_encodings, train_labels)
    dev_dataset = ToxicDataset(dev_encodings, dev_labels)

    print("Initializing Model...")
    # problem_type="multi_label_classification" ensures BCEWithLogitsLoss is used natively
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(LABELS),
        problem_type="multi_label_classification" 
    )

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1"
    )

    # Calculate class weights for BCEWithLogitsLoss
    train_labels_df = train_df[LABELS]
    pos_counts = train_labels_df.sum().values
    neg_counts = len(train_df) - pos_counts
    # neg / pos is standard pos_weight
    pos_weights = neg_counts / (pos_counts + 1e-5)
    pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float)
    print(f"Calculated positive class weights: {pos_weights_tensor}")

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        class_weights=pos_weights_tensor
    )

    print("Starting Training Loop...")
    trainer.train()

    print("Evaluating Best Model on Dev Set...")
    metrics = trainer.evaluate()
    print(metrics)

    print("Saving Fine-tuned LLM...")
    model.save_pretrained("./fine_tuned_llm")
    tokenizer.save_pretrained("./fine_tuned_llm")
    print("Saved to './fine_tuned_llm'")

if __name__ == "__main__":
    # Set sample_size=None for full training on Colab.
    train_llm(sample_size=None)
