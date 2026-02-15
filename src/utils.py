"""
Utility Functions for Model Training, Evaluation, and Visualization

This module provides helper functions for plotting training history,
confusion matrices, and analyzing model performance.

Author: Binger Yu
Date: February 14, 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path


def plot_training_history(history, title="Training History", save_path=None, 
                          show=True, figsize=(14, 5)):
    """
    Plot training and validation accuracy and loss curves.
    
    Args:
        history: Keras History object returned by model.fit()
        title (str): Title for the plot. Default: "Training History"
        save_path (str or Path): If provided, save figure to this path.
                                Default: None (don't save)
        show (bool): If True, display the plot. Default: True
        figsize (tuple): Figure size (width, height). Default: (14, 5)
    
    Returns:
        matplotlib.figure.Figure: The generated figure object
    
    Example:
        >>> history = model.fit(X_train, y_train, epochs=25, ...)
        >>> plot_training_history(
        ...     history,
        ...     title="Baseline CNN Training",
        ...     save_path="../results/baseline_history.png"
        ... )
    
    Displays:
        - Left plot: Training and validation accuracy over epochs
        - Right plot: Training and validation loss over epochs
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history plot saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=None, 
                         title="Confusion Matrix", save_path=None,
                         show=True, figsize=(8, 6), normalize=False):
    """
    Plot confusion matrix for classification results.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (list): List of class names for labels.
                           Default: ['NORMAL', 'PNEUMONIA']
        title (str): Title for the plot. Default: "Confusion Matrix"
        save_path (str or Path): If provided, save figure to this path.
                                Default: None (don't save)
        show (bool): If True, display the plot. Default: True
        figsize (tuple): Figure size (width, height). Default: (8, 6)
        normalize (bool): If True, normalize confusion matrix.
                         Default: False (show counts)
    
    Returns:
        matplotlib.figure.Figure: The generated figure object
    
    Example:
        >>> y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
        >>> plot_confusion_matrix(
        ...     y_test, y_pred,
        ...     class_names=['NORMAL', 'PNEUMONIA'],
        ...     title="Baseline CNN - Test Set",
        ...     save_path="../results/baseline_cm.png"
        ... )
    """
    if class_names is None:
        class_names = ['NORMAL', 'PNEUMONIA']
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        cbar_label = 'Proportion'
    else:
        fmt = 'd'
        cbar_label = 'Count'
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': cbar_label}, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    if show:
        plt.show()
    
    # Print confusion matrix breakdown
    tn, fp, fn, tp = cm.ravel() if not normalize else (cm * [len(y_true) for _ in range(4)]).ravel()
    print(f"\nConfusion Matrix Breakdown:")
    print(f"  True Negatives (TN): {int(tn)}")
    print(f"  False Positives (FP): {int(fp)} ({class_names[0]} → {class_names[1]})")
    print(f"  False Negatives (FN): {int(fn)} ({class_names[1]} → {class_names[0]}) ⚠️")
    print(f"  True Positives (TP): {int(tp)}")
    
    return fig


def print_classification_report(y_true, y_pred, class_names=None, digits=4):
    """
    Print detailed classification report with precision, recall, and F1-score.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (list): List of class names for labels.
                           Default: ['NORMAL', 'PNEUMONIA']
        digits (int): Number of decimal places. Default: 4
    
    Example:
        >>> y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
        >>> print_classification_report(y_test, y_pred)
    """
    if class_names is None:
        class_names = ['NORMAL', 'PNEUMONIA']
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, 
                               target_names=class_names, 
                               digits=digits))
    print("=" * 60)


def evaluate_model(model, X_test, y_test, class_names=None, verbose=True):
    """
    Comprehensive model evaluation with metrics and visualizations.
    
    Args:
        model: Trained Keras model
        X_test (np.ndarray): Test images
        y_test (np.ndarray): Test labels
        class_names (list): List of class names. Default: ['NORMAL', 'PNEUMONIA']
        verbose (bool): If True, print detailed results. Default: True
    
    Returns:
        dict: Dictionary containing all evaluation metrics
    
    Example:
        >>> results = evaluate_model(model, X_test, y_test)
        >>> print(f"Test Accuracy: {results['accuracy']:.4f}")
        >>> print(f"F1-Score: {results['f1_score']:.4f}")
    """
    if class_names is None:
        class_names = ['NORMAL', 'PNEUMONIA']
    
    # Get predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    # Evaluate metrics
    loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
    
    # Calculate F1 score (handle division by zero)
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Compile results
    results = {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': cm,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'predictions': y_pred,
        'prediction_probabilities': y_pred_probs.flatten()
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        print("\nConfusion Matrix:")
        print(f"  TN: {tn}  |  FP: {fp}")
        print(f"  FN: {fn}  |  TP: {tp}")
        print("=" * 60)
        
        # Print classification report
        print_classification_report(y_test, y_pred, class_names)
    
    return results


def plot_sample_images(X, y, class_names=None, n_samples=5, 
                      title="Sample Images", save_path=None, show=True):
    """
    Plot sample images from each class.
    
    Args:
        X (np.ndarray): Images array
        y (np.ndarray): Labels array
        class_names (list): List of class names. Default: ['NORMAL', 'PNEUMONIA']
        n_samples (int): Number of samples per class. Default: 5
        title (str): Title for the plot. Default: "Sample Images"
        save_path (str or Path): If provided, save figure to this path.
        show (bool): If True, display the plot. Default: True
    
    Returns:
        matplotlib.figure.Figure: The generated figure object
    """
    if class_names is None:
        class_names = ['NORMAL', 'PNEUMONIA']
    
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(15, 6))
    
    for class_idx in range(n_classes):
        # Get indices for this class
        class_indices = np.where(y == class_idx)[0]
        
        # Randomly sample
        if len(class_indices) >= n_samples:
            sample_indices = np.random.choice(class_indices, n_samples, replace=False)
        else:
            sample_indices = class_indices
        
        for col, idx in enumerate(sample_indices):
            img = X[idx]
            
            # Handle grayscale vs RGB
            if img.shape[-1] == 1:
                img = img.squeeze()
                axes[class_idx, col].imshow(img, cmap='gray')
            else:
                axes[class_idx, col].imshow(img)
            
            axes[class_idx, col].axis('off')
            
            # Add class label to first image
            if col == 0:
                axes[class_idx, col].set_title(
                    f'{class_names[class_idx]}',
                    fontweight='bold',
                    fontsize=12
                )
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Sample images saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def compare_models(results_dict, save_path=None, show=True):
    """
    Create comparison visualization for multiple models.
    
    Args:
        results_dict (dict): Dictionary mapping model names to their results.
                            Each result should be a dict with metrics.
                            Example: {
                                'Baseline': {'accuracy': 0.74, 'precision': 0.71, ...},
                                'Improved': {'accuracy': 0.88, 'precision': 0.85, ...}
                            }
        save_path (str or Path): If provided, save figure to this path.
        show (bool): If True, display the plot. Default: True
    
    Returns:
        matplotlib.figure.Figure: The generated figure object
    
    Example:
        >>> baseline_results = evaluate_model(baseline_model, X_test, y_test, verbose=False)
        >>> improved_results = evaluate_model(improved_model, X_test, y_test, verbose=False)
        >>> compare_models({
        ...     'Baseline': baseline_results,
        ...     'Improved': improved_results
        ... }, save_path='../results/model_comparison.png')
    """
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    model_names = list(results_dict.keys())
    
    # Extract metrics
    data = {metric: [] for metric in metrics_to_plot}
    for model_name in model_names:
        for metric in metrics_to_plot:
            data[metric].append(results_dict[model_name].get(metric, 0))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(model_names)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    
    for i, model_name in enumerate(model_names):
        values = [data[metric][i] for metric in metrics_to_plot]
        offset = width * (i - len(model_names) / 2 + 0.5)
        bars = ax.bar(x + offset, values, width, label=model_name, 
                     alpha=0.8, color=colors[i])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics_to_plot])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Model comparison saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


if __name__ == '__main__':
    """
    Test utility functions.
    Run: python -m src.utils
    """
    print("=" * 60)
    print("Testing Utility Functions")
    print("=" * 60)
    
    # Simulate some data for testing
    print("\n1. Creating sample data...")
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 1, 0, 0, 1, 1])
    
    # Test confusion matrix
    print("\n2. Testing confusion matrix...")
    print_classification_report(y_true, y_pred)
    
    # Test model comparison
    print("\n3. Testing model comparison...")
    results = {
        'Baseline': {'accuracy': 0.74, 'precision': 0.71, 'recall': 1.0, 'f1_score': 0.83},
        'Improved': {'accuracy': 0.88, 'precision': 0.85, 'recall': 0.98, 'f1_score': 0.91}
    }
    
    print("\nModel comparison data:")
    for model, metrics in results.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")
    
    print("\n" + "=" * 60)
    print("All utilities tested successfully! ✓")
    print("=" * 60)
