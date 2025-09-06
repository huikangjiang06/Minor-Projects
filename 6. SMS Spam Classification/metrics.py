#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics calculation for SMS Spam Classification
This script evaluates model predictions and generates score.json
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import json
import os
import sys

def load_ground_truth(dataset_type='validation'):
    """Load ground truth labels"""
    if dataset_type == 'validation':
        gt_file = 'dataset/validation_set/val.csv'
    elif dataset_type == 'testing':
        gt_file = 'dataset/testing_set/test.csv'
    else:
        raise ValueError("dataset_type must be 'validation' or 'testing'")

    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")

    gt_df = pd.read_csv(gt_file)
    return gt_df

def load_predictions(pred_file):
    """Load model predictions"""
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")

    pred_df = pd.read_csv(pred_file)
    return pred_df

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    # Convert labels to numeric if needed
    if isinstance(y_true[0], str):
        label_map = {'ham': 0, 'spam': 1}
        y_true_num = [label_map[label] for label in y_true]
        y_pred_num = [label_map[label] for label in y_pred]
    else:
        y_true_num = y_true
        y_pred_num = y_pred

    # Calculate metrics
    accuracy = accuracy_score(y_true_num, y_pred_num)
    precision = precision_score(y_true_num, y_pred_num)
    recall = recall_score(y_true_num, y_pred_num)
    f1 = f1_score(y_true_num, y_pred_num)

    # Confusion matrix
    cm = confusion_matrix(y_true_num, y_pred_num)
    tn, fp, fn, tp = cm.ravel()

    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'false_positive_rate': float(fpr),
        'false_negative_rate': float(fnr),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'total_samples': len(y_true)
    }

    return metrics

def evaluate_model(pred_file, dataset_type='validation', output_file='score.json'):
    """Main evaluation function"""
    print(f"Evaluating predictions from: {pred_file}")
    print(f"Dataset type: {dataset_type}")

    try:
        # Load ground truth and predictions
        gt_df = load_ground_truth(dataset_type)
        pred_df = load_predictions(pred_file)

        # Ensure same number of samples
        if len(gt_df) != len(pred_df):
            print(f"Warning: Ground truth has {len(gt_df)} samples, predictions have {len(pred_df)} samples")
            min_len = min(len(gt_df), len(pred_df))
            gt_df = gt_df.head(min_len)
            pred_df = pred_df.head(min_len)

        # Extract labels
        if 'label' in gt_df.columns:
            y_true = gt_df['label'].tolist()
        else:
            raise ValueError("Ground truth file must have 'label' column")

        if 'predicted_label' in pred_df.columns:
            y_pred = pred_df['predicted_label'].tolist()
        elif 'prediction' in pred_df.columns:
            y_pred = pred_df['prediction'].tolist()
        else:
            raise ValueError("Predictions file must have 'predicted_label' or 'prediction' column")

        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)

        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:     {metrics['accuracy']:.4f}")
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:       {metrics['recall']:.4f}")
        print(f"F1 Score:     {metrics['f1_score']:.4f}")
        print(f"Specificity:  {metrics['specificity']:.4f}")
        print("\nConfusion Matrix:")
        print(f"True Negatives:  {metrics['true_negatives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        print(f"True Positives:  {metrics['true_positives']}")

        # Create score structure for leaderboard
        if dataset_type == 'validation':
            score_a = metrics['accuracy']  # Public score
            score_b = 0.0  # Will be calculated on test set
        else:  # testing
            score_a = 0.0  # Public score already calculated
            score_b = metrics['accuracy']  # Final score

        # Create final score structure
        score_data = {
            'score_a': float(score_a),  # Public leaderboard score
            'score_b': float(score_b),  # Final leaderboard score
            'detailed_metrics': metrics,
            'dataset_type': dataset_type,
            'evaluation_timestamp': pd.Timestamp.now().isoformat()
        }

        # Save score.json
        with open(output_file, 'w') as f:
            json.dump(score_data, f, indent=2)

        print(f"\nResults saved to: {output_file}")
        print(f"Public Score (score_a): {score_a:.4f}")
        print(f"Final Score (score_b): {score_b:.4f}")

        return score_data

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return None

def main():
    """Main function with command line interface"""
    if len(sys.argv) < 2:
        print("Usage: python metrics.py <predictions_file> [dataset_type] [output_file]")
        print("Example: python metrics.py submission.csv validation score.json")
        sys.exit(1)

    pred_file = sys.argv[1]
    dataset_type = sys.argv[2] if len(sys.argv) > 2 else 'validation'
    output_file = sys.argv[3] if len(sys.argv) > 3 else 'score.json'

    # Run evaluation
    result = evaluate_model(pred_file, dataset_type, output_file)

    if result is None:
        sys.exit(1)

if __name__ == "__main__":
    main()
