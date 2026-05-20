"""
Visualization utilities for Forensic-MoE training and inference
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path
import seaborn as sns


def plot_training_curves(checkpoint_dir, output_path='training_curves.png'):
    """
    Plot training curves from checkpoint metadata
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        output_path: Output image path
    """
    checkpoints = sorted(Path(checkpoint_dir).glob('*.pth'))
    
    if len(checkpoints) == 0:
        print("No checkpoints found!")
        return
    
    epochs = []
    total_losses = []
    bce_losses = []
    ekd_losses = []
    expert0_losses = []
    expert1_losses = []
    
    for ckpt_path in checkpoints:
        try:
            import torch
            ckpt = torch.load(ckpt_path, map_location='cpu')
            
            if 'train_stats' in ckpt:
                stats = ckpt['train_stats']
                epoch = ckpt['epoch']
                
                epochs.append(epoch)
                total_losses.append(stats['loss'])
                bce_losses.append(stats['bce'])
                ekd_losses.append(stats['ekd'])
                
                # Per-expert stats
                if 'expert_stats' in stats:
                    if 0 in stats['expert_stats'] and 'avg_loss' in stats['expert_stats'][0]:
                        expert0_losses.append(stats['expert_stats'][0]['avg_loss'])
                    if 1 in stats['expert_stats'] and 'avg_loss' in stats['expert_stats'][1]:
                        expert1_losses.append(stats['expert_stats'][1]['avg_loss'])
        except Exception as e:
            print(f"Error loading {ckpt_path}: {e}")
            continue
    
    if len(epochs) == 0:
        print("No valid checkpoint data found!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Forensic-MoE Training Curves', fontsize=16)
    
    # Plot 1: Total Loss
    axes[0, 0].plot(epochs, total_losses, 'b-', linewidth=2, marker='o')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss (L_m = L_BCE + λ × L_EKD)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: BCE vs EKD
    axes[0, 1].plot(epochs, bce_losses, 'r-', linewidth=2, marker='s', label='BCE Loss')
    axes[0, 1].plot(epochs, ekd_losses, 'g-', linewidth=2, marker='^', label='EKD Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('BCE vs EKD Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Per-Expert Losses
    if len(expert0_losses) > 0 and len(expert1_losses) > 0:
        axes[1, 0].plot(epochs[:len(expert0_losses)], expert0_losses, 
                       'c-', linewidth=2, marker='d', label='Expert 0 (DM)')
        axes[1, 0].plot(epochs[:len(expert1_losses)], expert1_losses, 
                       'm-', linewidth=2, marker='*', label='Expert 1 (GAN)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Per-Expert Losses')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Loss Ratios
    if len(bce_losses) > 0 and len(ekd_losses) > 0:
        ratios = [e / (b + 1e-8) for b, e in zip(bce_losses, ekd_losses)]
        axes[1, 1].plot(epochs, ratios, 'orange', linewidth=2, marker='v')
        axes[1, 1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Equal contribution')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].set_title('EKD/BCE Loss Ratio')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {output_path}")
    plt.close()


def visualize_expert_weights(weights, expert_names=None, output_path='expert_weights.png'):
    """
    Visualize expert weight distribution
    
    Args:
        weights: Array of expert weights (num_samples, num_experts)
        expert_names: List of expert names
        output_path: Output image path
    """
    if expert_names is None:
        expert_names = [f'Expert {i}' for i in range(weights.shape[1])]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Average weights
    avg_weights = weights.mean(axis=0)
    axes[0].bar(expert_names, avg_weights, color=['#3498db', '#e74c3c'])
    axes[0].set_ylabel('Average Weight')
    axes[0].set_title('Average Expert Contribution')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(avg_weights):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # Plot 2: Weight distribution
    axes[1].boxplot([weights[:, i] for i in range(weights.shape[1])],
                    labels=expert_names)
    axes[1].set_ylabel('Weight')
    axes[1].set_title('Expert Weight Distribution')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Expert weights visualization saved to {output_path}")
    plt.close()


def visualize_feature_alignment(features, labels, expert_idx, output_path='feature_alignment.png'):
    """
    Visualize feature alignment using t-SNE
    
    Args:
        features: Feature array (num_samples, feature_dim)
        labels: Label array (num_samples,) - 0 for real, 1 for fake
        expert_idx: Index of expert
        output_path: Output image path
    """
    from sklearn.manifold import TSNE
    
    # Reduce dimensionality with t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot real samples
    real_mask = (labels == 0)
    ax.scatter(features_2d[real_mask, 0], features_2d[real_mask, 1],
              c='blue', alpha=0.6, s=50, label='Real', edgecolors='k', linewidth=0.5)
    
    # Plot fake samples
    fake_mask = (labels == 1)
    ax.scatter(features_2d[fake_mask, 0], features_2d[fake_mask, 1],
              c='red', alpha=0.6, s=50, label='Fake', edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title(f'Expert {expert_idx} Feature Alignment (t-SNE)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Feature alignment visualization saved to {output_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_path='confusion_matrix.png'):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Output image path
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Real', 'Fake'],
               yticklabels=['Real', 'Fake'],
               ax=ax, cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    # Add accuracy text
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    plt.text(0.5, -0.15, f'Accuracy: {accuracy:.2%}', 
            ha='center', transform=ax.transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {output_path}")
    plt.close()


def plot_roc_curve(y_true, y_scores, output_path='roc_curve.png'):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_scores: Prediction scores (probabilities)
        output_path: Output image path
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
           label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {output_path}")
    plt.close()


def create_training_report(checkpoint_dir, output_dir='visualizations'):
    """
    Create a complete training report with all visualizations
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        output_dir: Output directory for visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating training report...")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Output directory: {output_dir}")
    
    # Plot training curves
    plot_training_curves(
        checkpoint_dir,
        os.path.join(output_dir, 'training_curves.png')
    )
    
    print("\nTraining report created successfully!")
    print(f"Check {output_dir}/ for visualizations")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Forensic-MoE Training')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Directory containing training checkpoints')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    create_training_report(args.checkpoint_dir, args.output_dir)
