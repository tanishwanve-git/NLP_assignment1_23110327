import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import joblib
import sys
import os

# Add project root to sys.path so we can import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import LABELS, load_labeled_csv

def train_baseline(train_path="train.csv", dev_path="dev.csv"):
    print("Loading data for baseline training...")
    train_df = load_labeled_csv(train_path)
    dev_df   = load_labeled_csv(dev_path)
    if train_df is None or dev_df is None:
        print("Data files not found. Please run src/data_pipeline.py first.")
        return

    X_train_raw = train_df["comment_text"]
    y_train     = train_df[LABELS]

    X_dev_raw = dev_df["comment_text"]
    y_dev     = dev_df[LABELS]

    print("Extracting TF-IDF features...")
    # Limited to 10,000 features to prevent out-of-memory errors on smaller machines
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_raw)
    X_dev = vectorizer.transform(X_dev_raw)

    print("Training Logistic Regression (One-Vs-Rest) Baseline...")
    # OneVsRestClassifier allows us to train a separate binary classifier for each label
    classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear', class_weight='balanced'))
    classifier.fit(X_train, y_train)

    print("Evaluating on Dev set...")
    y_pred = classifier.predict(X_dev)
    y_pred_proba = classifier.predict_proba(X_dev)

    # Metrics
    macro_f1 = f1_score(y_dev, y_pred, average='macro')
    roc_auc = roc_auc_score(y_dev, y_pred_proba, average='macro')

    print("\n--- Baseline Results ---")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print(f"ROC-AUC Score:  {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_dev, y_pred, target_names=y_dev.columns))

    print("Saving baseline model to disk...")
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    joblib.dump(classifier, 'baseline_lr_model.pkl')
    print("Saved as 'tfidf_vectorizer.pkl' and 'baseline_lr_model.pkl'")

if __name__ == "__main__":
    train_baseline()
