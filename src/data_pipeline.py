import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import report_class_distribution

def load_and_split_data(dataset_name="google/jigsaw_toxicity_pred", test_size=0.1, dev_size=0.1):
    """
    Loads the toxic comment dataset and splits it into Train, Dev, and Test sets.
    Because this is a multi-label dataset with extreme class imbalance, we must ensure
    our splits are robust.
    """
    print(f"Loading dataset: {dataset_name}...")
    try:
        # Load from HuggingFace
        dataset = load_dataset(dataset_name)
        df = pd.DataFrame(dataset['train'])
    except Exception as e:
        print(f"Failed to load from {dataset_name}: {e}")
        print("Fallback: Attempting to load 'thesofakillers/jigsaw-toxic-comment-classification-challenge'")
        dataset = load_dataset("thesofakillers/jigsaw-toxic-comment-classification-challenge")
        df = pd.DataFrame(dataset['train'])

    print(f"Total instances loaded: {len(df)}")
    
    print("Deduplicating dataset...")
    initial_len = len(df)
    df = df.drop_duplicates(subset=["comment_text"])
    print(f"Dropped {initial_len - len(df)} duplicate rows. New size: {len(df)}")
    
    # Define features and labels
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # We will stratify by a combination of labels if possible, but since it's multi-label,
    # true stratification is complex. A simple heuristic is to stratify based on whether
    # the comment is clean (0 across all labels) or toxic (at least 1 across all labels)
    df['is_toxic'] = df[labels].max(axis=1)

    print("Splitting into Train, Dev, Test...")
    # First split: Train vs Temp (Dev + Test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(test_size + dev_size), 
        stratify=df['is_toxic'],
        random_state=42
    )

    # Second split: Dev vs Test
    # Calculate proportion of test relative to temp
    test_prop = test_size / (test_size + dev_size)
    dev_df, test_df = train_test_split(
        temp_df, 
        test_size=test_prop, 
        stratify=temp_df['is_toxic'],
        random_state=42
    )

    # Drop the heuristic column used for splitting
    train_df = train_df.drop(columns=['is_toxic'])
    dev_df = dev_df.drop(columns=['is_toxic'])
    test_df = test_df.drop(columns=['is_toxic'])

    print(f"Train size: {len(train_df)}")
    print(f"Dev size: {len(dev_df)}")
    print(f"Test size: {len(test_df)}")

    # Leakage Audit
    train_texts = set(train_df["comment_text"])
    dev_texts = set(dev_df["comment_text"])
    test_texts = set(test_df["comment_text"])
    
    assert len(train_texts.intersection(dev_texts)) == 0, "Leakage detected between train and dev sets!"
    assert len(train_texts.intersection(test_texts)) == 0, "Leakage detected between train and test sets!"
    assert len(dev_texts.intersection(test_texts)) == 0, "Leakage detected between dev and test sets!"
    print("Leakage audit passed: No overlapping comments between splits.")

    report_class_distribution(train_df, "Train")
    report_class_distribution(dev_df, "Dev")
    report_class_distribution(test_df, "Test")

    return train_df, dev_df, test_df

if __name__ == "__main__":
    train_df, dev_df, test_df = load_and_split_data()
    
    # Save splits to disk for easy loading by the training scripts
    print("Saving splits to disk...")
    train_df.to_csv("train.csv", index=False)
    dev_df.to_csv("dev.csv", index=False)
    test_df.to_csv("test.csv", index=False)
    print("Done!")
