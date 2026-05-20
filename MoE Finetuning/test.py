## -*- coding: utf-8 -*-
"""
Testing script for fine-tuned DaRE-MoE (ForensicMoE).

Loads a trained ForensicMoE checkpoint, runs inference on a test set, and reports
comprehensive metrics: Accuracy, AUC, EER, ACER, per-class accuracy, precision,
recall, F1, and a confusion matrix.

Usage:
    python test_moe_finetune.py -m checkpoint/DaRE_MoE/model_params_best_*.pkl -t datasets/test
    python test_moe_finetune.py -m checkpoint/DaRE_MoE/model_params_best_*.pkl -t datasets/test --save_results -o results/
"""
import os
import sys
import torch
import numpy as np
import random
import torch.nn.functional as F
from dataset import get_test_dataloader
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import time
from tqdm import tqdm
import argparse
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import transform as dataset_transform
from utils import cal_metrics
from models.moe_forensic import ForensicMoE


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def test_model(model, test_loader, device):
    """Run inference and collect labels, predictions, and probabilities."""
    model.eval()

    all_predictions = []
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, total=len(test_loader),
                                   ncols=70, desc='Testing', unit='batch'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # ForensicMoE forward returns (batch_size, 1) logits
            outputs = model(inputs)
            
            # Binary classification via BCEWithLogitsLoss
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            
            # Handle case where batch_size=1 makes squeeze remove batch dim
            if probs.ndim == 0:
                probs = np.expand_dims(probs, axis=0)

            preds = (probs > 0.5).astype(int)

            all_predictions.extend(preds)
            all_probabilities.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    return (np.array(all_labels),
            np.array(all_predictions),
            np.array(all_probabilities))


def calculate_metrics(labels, predictions, probabilities):
    """Calculate and print comprehensive evaluation metrics."""
    accuracy = accuracy_score(labels, predictions)

    try:
        auc_score = roc_auc_score(labels, probabilities)
    except ValueError:
        auc_score = 0.0
        print("Warning: Could not calculate AUC (single class present?)")

    cm = confusion_matrix(labels, predictions)

    # -----------------------------------------------------------
    # Detailed metrics via cal_metrics (uses class-0 = real as positive)
    # cal_metrics expects: y_trues (int list), y_preds (float list)
    # Convention in utils.py:
    #   confusion_matrix(y_trues, prediction, labels=[0,1])
    #   Row 0 → actual class 0 (real):  TP, FN
    #   Row 1 → actual class 1 (fake):  FP, TN
    # -----------------------------------------------------------
    try:
        detailed = cal_metrics(labels.tolist(), probabilities.tolist(), threshold=0.5)
        print(f"\nDetailed Metrics (cal_metrics):")
        print(f"  Accuracy:  {detailed.ACC:.4f} ({detailed.ACC * 100:.2f}%)")
        print(f"  AUC:       {detailed.AUC:.4f}")
        print(f"  EER:       {detailed.EER:.4f} ({detailed.EER * 100:.2f}%)")
        print(f"  APCER:     {detailed.APCER:.4f}")
        print(f"  BPCER:     {detailed.BPCER:.4f}")
        print(f"  ACER:      {detailed.ACER:.4f}")
    except Exception as e:
        detailed = None
        print(f"Warning: cal_metrics failed — {e}")

    # -----------------------------------------------------------
    # Classification report (sklearn)
    # -----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("CLASSIFICATION REPORT")
    print(f"{'=' * 60}")
    target_names = ['Real (Class 0)', 'Fake (Class 1)']
    print(classification_report(labels, predictions,
                                target_names=target_names, digits=4))

    # -----------------------------------------------------------
    # Per-class accuracy & standard metrics
    # -----------------------------------------------------------
    tn, fp, fn, tp = cm.ravel()
    real_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    fake_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"{'=' * 60}")
    print("SUMMARY METRICS")
    print(f"{'=' * 60}")
    print(f"Overall Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"AUC-ROC Score:     {auc_score:.4f}")
    print(f"\nPer-Class Accuracy:")
    print(f"  Real (Class 0): {real_acc:.4f} ({real_acc * 100:.2f}%)")
    print(f"  Fake (Class 1): {fake_acc:.4f} ({fake_acc * 100:.2f}%)")
    print(f"\nBinary Detection Metrics (Fake as positive):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\n{'=' * 60}")
    print("CONFUSION MATRIX")
    print(f"{'=' * 60}")
    print(f"                 Predicted Real  Predicted Fake")
    print(f"Actual Real      {tn:14d}  {fp:14d}")
    print(f"Actual Fake      {fn:14d}  {tp:14d}")
    print(f"{'=' * 60}\n")

    return {
        'accuracy': accuracy,
        'auc': auc_score,
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'real_accuracy': real_acc,
        'fake_accuracy': fake_acc,
        'detailed': detailed,
    }


def plot_confusion_matrix(cm, save_path=None):
    """Plot and optionally save confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix — DaRE-MoE')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Test fine-tuned DaRE-MoE (ForensicMoE) model')

    # Paths
    parser.add_argument('--model_path', '-m', type=str, required=True,
                        help='Path to the finetuned model checkpoint (.pkl or .tar)')
    parser.add_argument('--test_data_path', '-t', type=str, required=True,
                        help='Path to test dataset folder (ImageFolder layout)')
    parser.add_argument('--output_dir', '-o', type=str, default='test_results_dare_moe',
                        help='Directory to save results')

    # Data loading
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    # Device
    parser.add_argument('--device', '-dv', type=int, default=0, help='GPU device ID')

    # Model architecture kwargs (must match the fine-tuned checkpoint config)
    parser.add_argument('--num_experts', type=int, default=2)
    parser.add_argument('--feature_dim', type=int, default=768)

    # Testing options
    parser.add_argument('--strict_load', action='store_true', default=True,
                        help='Strict state_dict loading')
    parser.add_argument('--save_results', '-s', action='store_true',
                        help='Save metrics and confusion matrix plot')

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    setup_seed(2024)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Validate paths
    # ------------------------------------------------------------------
    if not os.path.isfile(args.model_path):
        print(f"Error: Model file not found — {args.model_path}")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = (args.test_data_path if os.path.isabs(args.test_data_path)
                 else os.path.join(base_dir, args.test_data_path))

    if not os.path.isdir(test_path):
        print(f"Error: Test data directory not found — {test_path}")
        print("Expected ImageFolder layout, e.g.:")
        print("  test_path/0_real/img1.jpg")
        print("  test_path/1_fake/img1.jpg")
        return

    # ------------------------------------------------------------------
    # Print configuration
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("DaRE-MoE TEST CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"  Model checkpoint : {args.model_path}")
    print(f"  Test data        : {test_path}")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  Num experts      : {args.num_experts}")
    print(f"  Strict loading   : {args.strict_load}")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # Dataset — use the same transform as training (dataset.py)
    # ------------------------------------------------------------------
    test_dataset = ImageFolder(os.path.join(test_path,"test"))
    print(f"Test dataset: {len(test_dataset)} images")
    print(f"  Classes: {test_dataset.classes}")
    print(f"  Class→idx: {test_dataset.class_to_idx}")

    # if len(test_dataset) == 0:
    #     print("Error: test dataset is empty!")
    #     return

    test_loader = get_test_dataloader(test_path)

    # ------------------------------------------------------------------
    # Build model (ForensicMoE from models/moe_forensic.py)
    # ------------------------------------------------------------------
    # Provide empty expert target state dicts to initialize the model architecture.
    # The actual parameters will be overwritten by the fine-tuned checkpoint.
    expert_states = [{}] * args.num_experts
    model = ForensicMoE(
        backbone=None,
        expert_models=expert_states,
        num_experts=args.num_experts,
        feature_dim=args.feature_dim,
        freeze_backbone=False
    )

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    print(f"Loading checkpoint from: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location='cpu', weights_only=False)

    # Support both raw state_dict and {'model_state_dict': ...} dicts
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    if args.strict_load:
        model.load_state_dict(state_dict)
        print("Checkpoint loaded (strict load passed).")
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: {len(missing)} missing key(s)")
        if unexpected:
            print(f"Warning: {len(unexpected)} unexpected key(s)")
        if not missing and not unexpected:
            print("Checkpoint loaded — all keys matched.")

    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters loaded: {total_params:,} total")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    print("\nStarting evaluation ...")
    start_time = time.time()
    labels, predictions, probabilities = test_model(model, test_loader, device)
    elapsed = time.time() - start_time

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("TEST RESULTS — DaRE-MoE")
    print(f"{'=' * 60}")
    metrics = calculate_metrics(labels, predictions, probabilities)

    print(f"Inference time : {elapsed:.2f}s ({elapsed / 60:.2f} min)")
    print(f"Images tested  : {len(labels)}")
    print(f"Throughput     : {len(labels) / elapsed:.1f} img/s")
    print(f"Latency/image  : {elapsed / len(labels) * 1000:.2f} ms")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)

        # Confusion matrix plot
        cm_path = os.path.join(args.output_dir, 'confusion_matrix_dare_moe.png')
        plot_confusion_matrix(metrics['confusion_matrix'], cm_path)

        # Text report
        report_path = os.path.join(args.output_dir, 'test_results_dare_moe.txt')
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DaRE-MoE TEST RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model:           {args.model_path}\n")
            f.write(f"Test data:       {test_path}\n")
            f.write(f"Total samples:   {len(labels)}\n")
            f.write(f"Classes:         {test_dataset.classes}\n\n")
            f.write(f"Architecture:\n")
            f.write(f"  Num experts:   {args.num_experts}\n")
            f.write(f"  Feature dim:   {args.feature_dim}\n\n")
            f.write(f"Metrics:\n")
            f.write(f"  Accuracy:      {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)\n")
            f.write(f"  AUC-ROC:       {metrics['auc']:.4f}\n")
            f.write(f"  Precision:     {metrics['precision']:.4f}\n")
            f.write(f"  Recall:        {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:      {metrics['f1']:.4f}\n")
            f.write(f"  Real Acc:      {metrics['real_accuracy']:.4f} ({metrics['real_accuracy'] * 100:.2f}%)\n")
            f.write(f"  Fake Acc:      {metrics['fake_accuracy']:.4f} ({metrics['fake_accuracy'] * 100:.2f}%)\n\n")
            if metrics['detailed'] is not None:
                d = metrics['detailed']
                f.write(f"  EER:           {d.EER:.4f} ({d.EER * 100:.2f}%)\n")
                f.write(f"  APCER:         {d.APCER:.4f}\n")
                f.write(f"  BPCER:         {d.BPCER:.4f}\n")
                f.write(f"  ACER:          {d.ACER:.4f}\n\n")
            f.write(f"Timing:\n")
            f.write(f"  Total:         {elapsed:.2f}s\n")
            f.write(f"  Throughput:    {len(labels) / elapsed:.1f} img/s\n")
            f.write(f"  Latency/img:   {elapsed / len(labels) * 1000:.2f} ms\n")

        print(f"\nResults saved to: {args.output_dir}")
        print(f"  Report:           {report_path}")
        print(f"  Confusion matrix: {cm_path}")

    print(f"\n{'=' * 60}")
    print("Testing completed.")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()