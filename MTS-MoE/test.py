## -*- coding: utf-8 -*-
"""
Testing script for Vision Transformer with Multi-Transform Spectral MoE (MTS-MoE).

Loads a trained MTS-MoE checkpoint, runs inference on a test set, and reports
comprehensive metrics: Accuracy, AUC, EER, ACER, per-class accuracy, precision,
recall, F1, and a confusion matrix.

Usage:
    python test_mts_moe.py -m checkpoints/best_model.pkl -t datasets/test
    python test_mts_moe.py -m checkpoints/best_model.pkl --datasets_root test_dm
    python test_mts_moe.py -m checkpoints/best_model.pkl -t datasets/test --save_results -o results/
"""
import os
import sys
import csv
import torch
import numpy as np
import random
import torch.nn.functional as F
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

from dataset import transform as dataset_transform
from utils import cal_metrics
from MTS_MoE import vit_base_patch16_224_in21k


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

            # MTS-MoE forward returns (logits, moe_loss)
            outputs, _ = model(inputs)
            probs = F.softmax(outputs, dim=-1)

            # Class 1 = fake probability
            fake_prob = probs[:, 1].cpu().numpy()
            preds = (fake_prob > 0.5).astype(int)

            all_predictions.extend(preds)
            all_probabilities.extend(fake_prob)
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
    plt.title('Confusion Matrix — MTS-MoE')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def resolve_test_data_path(input_path, base_dir=None):
    """Resolve either a direct test folder or a dataset root containing test/."""
    resolved_path = (input_path if os.path.isabs(input_path)
                     else os.path.join(base_dir or os.getcwd(), input_path))

    if not os.path.isdir(resolved_path):
        raise FileNotFoundError(f"Test data directory not found: {resolved_path}")

    test_split_path = os.path.join(resolved_path, 'test')
    if os.path.isdir(test_split_path):
        return test_split_path, resolved_path

    return resolved_path, resolved_path


def get_dataset_directories(datasets_root, dataset_names=None):
    """Discover dataset roots under a parent directory."""
    datasets_root = os.path.abspath(datasets_root)
    if not os.path.isdir(datasets_root):
        raise FileNotFoundError(f"Datasets root not found: {datasets_root}")

    requested = set(dataset_names) if dataset_names else None
    dataset_dirs = []

    for entry in sorted(os.listdir(datasets_root)):
        if entry.startswith('.'):
            continue
        if requested is not None and entry not in requested:
            continue

        candidate = os.path.join(datasets_root, entry)
        if not os.path.isdir(candidate):
            continue
        if os.path.isdir(os.path.join(candidate, 'test')):
            dataset_dirs.append(candidate)

    if requested is not None:
        found = {os.path.basename(path) for path in dataset_dirs}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(
                f"Requested datasets not found or missing a test folder: {', '.join(missing)}"
            )

    if not dataset_dirs:
        raise ValueError(f"No dataset folders with a test split were found in {datasets_root}")

    return dataset_dirs


def build_test_loader(dataset_dir, batch_size, num_workers):
    """Create an ImageFolder test loader from one dataset root."""
    test_path, dataset_root = resolve_test_data_path(dataset_dir)
    test_dataset = ImageFolder(test_path, transform=dataset_transform)

    print(f"Test dataset: {len(test_dataset)} images")
    print(f"  Dataset root: {dataset_root}")
    print(f"  Test split:   {test_path}")
    print(f"  Classes: {test_dataset.classes}")
    print(f"  Class→idx: {test_dataset.class_to_idx}")

    if len(test_dataset) == 0:
        raise ValueError("test dataset is empty")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return test_loader, test_dataset, test_path, dataset_root


def save_single_dataset_results(output_dir, args, metrics, labels, elapsed,
                                model_path, test_path, test_dataset,
                                report_name='test_results_mts.txt',
                                confusion_name='confusion_matrix_mts.png'):
    """Persist per-dataset reports and confusion matrix."""
    os.makedirs(output_dir, exist_ok=True)

    cm_path = os.path.join(output_dir, confusion_name)
    plot_confusion_matrix(metrics['confusion_matrix'], cm_path)

    report_path = os.path.join(output_dir, report_name)
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MTS-MoE TEST RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model:           {model_path}\n")
        f.write(f"Test data:       {test_path}\n")
        f.write(f"Total samples:   {len(labels)}\n")
        f.write(f"Classes:         {test_dataset.classes}\n\n")
        f.write("Architecture:\n")
        f.write("  LoRA dims:     [2, 4, 6, 8] (AuroRA enabled)\n")
        f.write(f"  LoRA top-k:    {args.lora_k}\n")
        f.write(f"  MTS-MoE top-k: {args.adapter_k}\n")
        f.write("  MTS-MoE experts: DFT, DCT, DWT-Haar, DWT-DB2, Spatial\n\n")
        f.write("Metrics:\n")
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
        f.write("Timing:\n")
        f.write(f"  Total:         {elapsed:.2f}s\n")
        f.write(f"  Throughput:    {len(labels) / elapsed:.1f} img/s\n")
        f.write(f"  Latency/img:   {elapsed / len(labels) * 1000:.2f} ms\n")

    print(f"\nResults saved to: {output_dir}")
    print(f"  Report:           {report_path}")
    print(f"  Confusion matrix: {cm_path}")


def evaluate_single_dataset(model, device, dataset_dir, output_dir, args, save_results):
    """Evaluate one dataset root and optionally write per-dataset outputs."""
    dataset_name = os.path.basename(os.path.normpath(dataset_dir))
    print(f"\n{'=' * 60}")
    print(f"Evaluating dataset: {dataset_name}")
    print(f"{'=' * 60}")

    test_loader, test_dataset, test_path, dataset_root = build_test_loader(
        dataset_dir=dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("\nStarting evaluation ...")
    start_time = time.time()
    labels, predictions, probabilities = test_model(model, test_loader, device)
    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"TEST RESULTS — MTS-MoE ({dataset_name})")
    print(f"{'=' * 60}")
    metrics = calculate_metrics(labels, predictions, probabilities)

    print(f"Inference time : {elapsed:.2f}s ({elapsed / 60:.2f} min)")
    print(f"Images tested  : {len(labels)}")
    print(f"Throughput     : {len(labels) / elapsed:.1f} img/s")
    print(f"Latency/image  : {elapsed / len(labels) * 1000:.2f} ms")

    if save_results:
        save_single_dataset_results(
            output_dir=output_dir,
            args=args,
            metrics=metrics,
            labels=labels,
            elapsed=elapsed,
            model_path=args.model_path,
            test_path=test_path,
            test_dataset=test_dataset,
        )

    result = {
        'dataset': dataset_name,
        'status': 'success',
        'accuracy': metrics['accuracy'],
        'auc': metrics['auc'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'real_accuracy': metrics['real_accuracy'],
        'fake_accuracy': metrics['fake_accuracy'],
        'num_samples': len(labels),
        'num_real': int((labels == 0).sum()),
        'num_fake': int((labels == 1).sum()),
        'data_dir': dataset_root,
        'test_path': test_path,
        'error': '',
    }

    if metrics['detailed'] is not None:
        result['eer'] = metrics['detailed'].EER
        result['acer'] = metrics['detailed'].ACER
    else:
        result['eer'] = None
        result['acer'] = None

    return result


def write_batch_summary(summary_rows, output_dir, args):
    """Write consolidated batch evaluation outputs."""
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'batch_test_summary.csv')
    txt_path = os.path.join(output_dir, 'batch_test_summary.txt')

    fieldnames = [
        'dataset', 'status', 'accuracy', 'auc', 'eer', 'acer',
        'precision', 'recall', 'f1', 'num_samples', 'num_real', 'num_fake',
        'data_dir', 'test_path', 'error'
    ]

    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})

    successful_rows = [row for row in summary_rows if row.get('status') == 'success']
    average_accuracy = (
        sum(row['accuracy'] for row in successful_rows) / len(successful_rows)
        if successful_rows else None
    )

    with open(txt_path, 'w') as txt_file:
        txt_file.write("=" * 80 + "\n")
        txt_file.write("MTS-MOE BATCH TEST SUMMARY\n")
        txt_file.write("=" * 80 + "\n")
        txt_file.write(f"Checkpoint: {os.path.abspath(args.model_path)}\n")
        txt_file.write(f"LoRA top-k: {args.lora_k}\n")
        txt_file.write(f"MTS-MoE top-k: {args.adapter_k}\n")
        txt_file.write(f"Datasets evaluated: {len(summary_rows)}\n")
        txt_file.write(f"Successful runs: {len(successful_rows)}\n")
        if average_accuracy is not None:
            txt_file.write(f"Average accuracy: {average_accuracy:.4f} ({average_accuracy * 100:.2f}%)\n")
        txt_file.write("\n")
        txt_file.write(
            f"{'Dataset':<18} {'Status':<10} {'Accuracy':<10} {'AUC':<10} {'EER':<10} {'Samples':<10}\n"
        )
        txt_file.write("-" * 80 + "\n")

        for row in summary_rows:
            accuracy = f"{row['accuracy']:.4f}" if row.get('accuracy') is not None else 'N/A'
            auc = f"{row['auc']:.4f}" if row.get('auc') is not None else 'N/A'
            eer = f"{row['eer']:.4f}" if row.get('eer') is not None else 'N/A'
            samples = str(row.get('num_samples', 'N/A'))
            txt_file.write(
                f"{row.get('dataset', 'unknown'):<18} {row.get('status', 'unknown'):<10} "
                f"{accuracy:<10} {auc:<10} {eer:<10} {samples:<10}\n"
            )
            if row.get('error'):
                txt_file.write(f"  Error: {row['error']}\n")

    print(f"\nUnified CSV summary saved to {csv_path}")
    print(f"Unified text summary saved to {txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Test MTS-MoE (Multi-Transform Spectral MoE) model')

    # Paths
    parser.add_argument('--model_path', '-m', type=str, required=True,
                        help='Path to the saved model checkpoint (.pkl or .tar)')
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--test_data_path', '-t', type=str,
                            help='Path to a test folder or a dataset root containing test/')
    data_group.add_argument('--datasets_root', type=str,
                            help='Path containing multiple dataset folders, each with a test folder')
    parser.add_argument('--output_dir', '-o', type=str, default='test_results_mts',
                        help='Directory to save results')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Optional list of dataset folder names to evaluate when using --datasets_root')

    # Data loading
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    # Device
    parser.add_argument('--device', '-dv', type=int, default=0,
                        help='GPU device ID')

    # Model architecture — must match the checkpoint
    # NOTE: In MTS_MoE.py, LoRA expert dimensions [2,4,6,8] and
    # AuroRA (lora_use_act=True) are hardcoded in LoRA_MoElayer.
    # Only top-k routing values are configurable from outside.
    parser.add_argument('--lora_k', type=int, default=1,
                        help='Top-k experts per sample for LoRA-MoE routing')
    parser.add_argument('--adapter_k', type=int, default=1,
                        help='Top-k experts per sample for MTS-MoE routing '
                             '(5 experts: DFT, DCT, DWT-Haar, DWT-DB2, Spatial)')

    # Loading behaviour
    parser.add_argument('--strict_load', action='store_true',
                        help='Strict state_dict loading (fail on mismatch)')
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
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
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
    datasets_root = None
    test_path = None
    if args.datasets_root:
        datasets_root = (args.datasets_root if os.path.isabs(args.datasets_root)
                         else os.path.join(base_dir, args.datasets_root))
        if not os.path.isdir(datasets_root):
            print(f"Error: Datasets root not found — {datasets_root}")
            return
    else:
        try:
            test_path, _ = resolve_test_data_path(args.test_data_path, base_dir)
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            print("Expected either:")
            print("  dataset_root/test/0_real/img1.jpg")
            print("  dataset_root/test/1_fake/img1.jpg")
            print("or a direct ImageFolder layout:")
            print("  test_path/0_real/img1.jpg")
            print("  test_path/1_fake/img1.jpg")
            return

    # ------------------------------------------------------------------
    # Print configuration
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("MTS-MoE TEST CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"  Model checkpoint : {args.model_path}")
    if datasets_root:
        print(f"  Datasets root    : {datasets_root}")
        if args.datasets:
            print(f"  Dataset filter   : {', '.join(args.datasets)}")
    else:
        print(f"  Test data        : {test_path}")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  LoRA dims (fixed): [2, 4, 6, 8] (AuroRA enabled)")
    print(f"  LoRA top-k       : {args.lora_k}")
    print(f"  MTS-MoE experts  : DFT, DCT, DWT-Haar, DWT-DB2, Spatial")
    print(f"  MTS-MoE top-k    : {args.adapter_k}")
    print(f"  Strict loading   : {args.strict_load}")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # Build model (MTS-MoE architecture from MTS_MoE.py)
    # ------------------------------------------------------------------
    model = vit_base_patch16_224_in21k(
        pretrained=False,
        num_classes=2,
        lora_topk=args.lora_k,
        adapter_topk=args.adapter_k,
    )

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    print(f"Loading checkpoint from: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device, weights_only=False)

    # Support both raw state_dict and {'model_state_dict': ...} wrappers
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    if args.strict_load:
        model.load_state_dict(state_dict)
        print("Checkpoint loaded (strict).")
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: {len(missing)} missing key(s)")
            for k in missing[:10]:
                print(f"  - {k}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
        if unexpected:
            print(f"Warning: {len(unexpected)} unexpected key(s)")
            for k in unexpected[:10]:
                print(f"  - {k}")
            if len(unexpected) > 10:
                print(f"  ... and {len(unexpected) - 10} more")
        if not missing and not unexpected:
            print("Checkpoint loaded — all keys matched.")

    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    batch_mode = datasets_root is not None
    save_results = args.save_results or batch_mode

    if batch_mode:
        try:
            dataset_dirs = get_dataset_directories(datasets_root, args.datasets)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}")
            return

        print(f"\nDiscovered {len(dataset_dirs)} datasets under {datasets_root}")
        summary_rows = []
        for dataset_dir in dataset_dirs:
            dataset_name = os.path.basename(os.path.normpath(dataset_dir))
            dataset_output_dir = os.path.join(args.output_dir, dataset_name)
            try:
                results = evaluate_single_dataset(
                    model=model,
                    device=device,
                    dataset_dir=dataset_dir,
                    output_dir=dataset_output_dir,
                    args=args,
                    save_results=save_results,
                )
                summary_rows.append(results)
            except Exception as exc:
                error_message = str(exc)
                print(f"Failed to evaluate dataset '{dataset_name}': {error_message}")
                summary_rows.append({
                    'dataset': dataset_name,
                    'status': 'failed',
                    'accuracy': None,
                    'auc': None,
                    'eer': None,
                    'acer': None,
                    'precision': None,
                    'recall': None,
                    'f1': None,
                    'num_samples': None,
                    'num_real': None,
                    'num_fake': None,
                    'data_dir': os.path.abspath(dataset_dir),
                    'test_path': os.path.join(os.path.abspath(dataset_dir), 'test'),
                    'error': error_message,
                })

        write_batch_summary(summary_rows, args.output_dir, args)
    else:
        evaluate_single_dataset(
            model=model,
            device=device,
            dataset_dir=args.test_data_path,
            output_dir=args.output_dir,
            args=args,
            save_results=save_results,
        )

    print(f"\n{'=' * 60}")
    print("Testing completed.")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
