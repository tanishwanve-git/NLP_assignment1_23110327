import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import LABELS, load_labeled_csv, compute_binary_label

def run_eda(csv_path="train.csv"):
    df = load_labeled_csv(csv_path)
    if df is None:
        print(f"File {csv_path} not found or unreadable.")
        return

    print("--- Exploratory Data Analysis (EDA) ---")
    
    # 1. Label Distribution (Class Imbalance)
    label_counts = df[LABELS].sum()
    print("\nLabel Counts:")
    print(label_counts)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")
    plt.title("Distribution of Toxicity Labels (Class Imbalance)")
    plt.ylabel("Number of Comments")
    plt.xlabel("Label")
    plt.savefig("label_distribution.png", dpi=300)
    print("\nSaved 'label_distribution.png' for your report.")

    # 2. Clean vs Toxic Comments
    df['is_toxic'] = compute_binary_label(df)
    clean_count = (df['is_toxic'] == 0).sum()
    toxic_count = (df['is_toxic'] == 1).sum()
    
    print(f"\nClean Comments: {clean_count}")
    print(f"Toxic Comments: {toxic_count}")

    plt.figure(figsize=(6, 6))
    plt.pie([clean_count, toxic_count], labels=['Clean', 'Toxic'], autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
    plt.title("Clean vs Toxic Comments")
    plt.savefig("clean_vs_toxic.png", dpi=300)
    print("Saved 'clean_vs_toxic.png' for your report.")

    # 3. Comment Length Distribution
    # Note: Using str() in case there are missing values parsed as floats
    df['comment_length'] = df['comment_text'].astype(str).apply(len)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['comment_length'], bins=50, kde=True, color='purple')
    plt.title("Distribution of Comment Lengths (Characters)")
    plt.xlabel("Length")
    plt.xlim(0, 2000) # Truncate long tails for better visualization
    plt.savefig("comment_length.png", dpi=300)
    print("Saved 'comment_length.png' for your report.")

if __name__ == "__main__":
    run_eda()
