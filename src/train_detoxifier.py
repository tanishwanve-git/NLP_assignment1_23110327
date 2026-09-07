"""
train_detoxifier.py
===================
Script to fine-tune a T5-small model on the ParaDetox dataset for
intent-preserving text detoxification.

Usage (GPU highly recommended, e.g. Colab):
  python src/train_detoxifier.py
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

def train_detoxifier(model_name="t5-small"):
    print("Loading ParaDetox Dataset...")
    # 's-nlp/paradetox' contains toxic text and neutral paraphrases
    dataset = load_dataset("s-nlp/paradetox")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def preprocess_function(examples):
        # We want to translate toxic text to neutral text
        inputs = [ex for ex in examples["en_toxic_comment"]]
        targets = [ex for ex in examples["en_neutral_comment"]]
        
        # T5 requires a prefix, but we can just use the raw text if fine-tuning specifically for this
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)

        # Tokenize targets using modern text_target API (transformers >= 4.36)
        labels = tokenizer(
            text_target=targets, max_length=128, truncation=True
        )
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    # Split into train and eval
    split_dataset = tokenized_datasets["train"].train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./detox_results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(), # Use mixed precision if on GPU
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("Starting fine-tuning of Detoxifier...")
    trainer.train()
    
    print("Saving fine-tuned Detoxifier...")
    model.save_pretrained("./fine_tuned_detoxifier")
    tokenizer.save_pretrained("./fine_tuned_detoxifier")
    print("Saved to './fine_tuned_detoxifier'")

if __name__ == "__main__":
    train_detoxifier()
