#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference SMS Spam Classification using RoBERTa-base
This is a more sophisticated model that should achieve better performance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, classification_report, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
import torch
from torch.utils.data import Dataset
import os
import json

class SMSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """Compute accuracy, recall and F1 score"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    return {
        'accuracy': accuracy,
        'recall': recall,
        'f1': f1
    }

def load_data():
    """Load and prepare the SMS spam dataset"""
    # Load training data
    train_df = pd.read_csv('dataset/training_set/train.csv')
    val_df = pd.read_csv('dataset/validation_set/val.csv')

    # Convert labels to numeric (ham=0, spam=1)
    train_df['label_num'] = train_df['label'].map({'ham': 0, 'spam': 1})
    val_df['label_num'] = val_df['label'].map({'ham': 0, 'spam': 1})

    return train_df, val_df

def preprocess_text(text):
    """Basic text preprocessing"""
    # Convert to lowercase and strip whitespace
    text = str(text).lower().strip()
    return text

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    train_df, val_df = load_data()

    # Preprocess text
    print("Preprocessing text...")
    train_df['text'] = train_df['text'].apply(preprocess_text)
    val_df['text'] = val_df['text'].apply(preprocess_text)

    # Initialize tokenizer and model (RoBERTa-base)
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    # Move model to device
    model.to(device)

    # Create datasets
    print("Creating datasets...")
    train_dataset = SMSDataset(
        train_df['text'].tolist(),
        train_df['label_num'].tolist(),
        tokenizer,
        max_length=256
    )

    val_dataset = SMSDataset(
        val_df['text'].tolist(),
        val_df['label_num'].tolist(),
        tokenizer,
        max_length=256
    )

    # Advanced training arguments for better performance
    training_args = TrainingArguments(
        output_dir='./reference_model',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        report_to=None,
        dataloader_num_workers=2,
    )

    # Initialize trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train the model
    print("Training reference model...")
    trainer.train()

    # Make predictions on validation set
    print("Making predictions on validation set...")
    val_predictions = trainer.predict(val_dataset)
    val_pred_labels = np.argmax(val_predictions.predictions, axis=1)
    val_true_labels = val_df['label_num'].values

    # Calculate validation metrics
    val_accuracy = accuracy_score(val_true_labels, val_pred_labels)
    val_recall = recall_score(val_true_labels, val_pred_labels)
    val_f1 = f1_score(val_true_labels, val_pred_labels)

    print(f"Validation Results:")
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"Recall: {val_recall:.4f}")
    print(f"F1 Score: {val_f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(val_true_labels, val_pred_labels, target_names=['ham', 'spam']))

    # Load test dataset
    test_df = pd.read_csv('dataset/testing_set/test.csv')
    print(f"\nLoaded test dataset with {len(test_df)} samples")

    # Preprocess test data
    test_df['label_num'] = test_df['label'].map({'ham': 0, 'spam': 1})
    test_encodings = tokenizer(
        test_df['text'].tolist(),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )
    test_dataset = SMSDataset(test_encodings, test_df['label_num'].tolist())

    # Make predictions on test set
    print("Making predictions on test set...")
    test_predictions = trainer.predict(test_dataset)
    test_pred_labels = np.argmax(test_predictions.predictions, axis=1)

    # Save test predictions with confidence scores
    test_proba_scores = torch.softmax(torch.tensor(test_predictions.predictions), dim=1)
    test_results_df = test_df.copy()
    test_results_df['predicted'] = test_pred_labels
    test_results_df['predicted_label'] = test_results_df['predicted'].map({0: 'ham', 1: 'spam'})
    test_results_df['confidence_ham'] = test_proba_scores[:, 0].numpy()
    test_results_df['confidence_spam'] = test_proba_scores[:, 1].numpy()

    # Save test predictions
    test_results_df[['text', 'label', 'predicted_label', 'confidence_ham', 'confidence_spam']].to_csv(
        'reference_predictions.csv', index=False
    )

    # Save submission format
    submission_df = test_results_df[['text', 'predicted_label']].copy()
    submission_df.columns = ['text', 'prediction']
    submission_df.to_csv('submission.csv', index=False)

    # Save model
    model.save_pretrained('./reference_model')
    tokenizer.save_pretrained('./reference_model')

    # Save metrics (using validation results for model evaluation)
    metrics_dict = {
        'accuracy': float(val_accuracy),
        'recall': float(val_recall),
        'f1_score': float(val_f1),
        'model': model_name,
        'num_samples': len(val_df),
        'test_samples': len(test_df)
    }

    with open('reference_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    print("Reference model training completed!")
    print("Model saved to './reference_model'")
    print("Predictions saved to 'reference_predictions.csv'")
    print("Submission file saved to 'submission.csv'")
    print("Metrics saved to 'reference_metrics.json'")

if __name__ == "__main__":
    main()
