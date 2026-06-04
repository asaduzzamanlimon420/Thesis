"""
Comprehensive evaluation:
  - Per-class accuracy, precision, recall, F1, AUC-ROC
  - Confusion matrix
  - Per-client fairness analysis
  - FL vs centralised comparison charts
  - Communication efficiency
"""

import os, json
import numpy as np
import torch
from torch.cuda.amp import autocast
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_fscore_support, accuracy_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config as C


@torch.no_grad()
def evaluate_model(model, loader, device, return_preds=False):
    """Run inference and compute metrics."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        with autocast(enabled=C.USE_AMP):
            logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.append(probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    # Per-class AUC (one-vs-rest)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "auc_roc_macro": float(auc),
    }

    # Per-class metrics
    report = classification_report(
        all_labels, all_preds,
        target_names=C.CLASS_NAMES, output_dict=True, zero_division=0,
    )
    for cls in C.CLASS_NAMES:
        if cls in report:
            metrics[f"{cls}_f1"] = report[cls]["f1-score"]
            metrics[f"{cls}_precision"] = report[cls]["precision"]
            metrics[f"{cls}_recall"] = report[cls]["recall"]

    if return_preds:
        return metrics, all_preds, all_labels, all_probs
    return metrics


def plot_confusion_matrix(preds, labels, save_path, title="Confusion Matrix"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")


def plot_training_curves(history: dict, save_dir: str, prefix=""):
    """
    Plot loss and accuracy curves.
    history: {"round": [...], "loss": [...], "accuracy": [...]}
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if "loss" in history:
        ax1.plot(history.get("round", range(len(history["loss"]))), history["loss"], "b-o", markersize=3)
        ax1.set_xlabel("Round / Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"{prefix} Training Loss")
        ax1.grid(True, alpha=0.3)

    if "accuracy" in history:
        ax2.plot(history.get("round", range(len(history["accuracy"]))), history["accuracy"], "g-o", markersize=3)
        ax2.set_xlabel("Round / Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title(f"{prefix} Accuracy")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison(results: dict, save_path: str):
    """
    Bar chart comparing different approaches.
    results: {"Centralised": {...metrics...}, "FedAvg": {...}, "FedBNProx": {...}}
    """
    approaches = list(results.keys())
    metrics_names = ["accuracy", "f1_macro", "auc_roc_macro"]
    labels = ["Accuracy", "F1 (Macro)", "AUC-ROC"]

    x = np.arange(len(labels))
    width = 0.8 / len(approaches)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, approach in enumerate(approaches):
        values = [results[approach].get(m, 0) for m in metrics_names]
        bars = ax.bar(x + i * width, values, width, label=approach)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x + width * (len(approaches) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison plot saved: {save_path}")


def plot_per_client_fairness(client_metrics: dict, save_path: str):
    """
    Radar / bar chart of per-client performance to show fairness.
    client_metrics: {0: {...}, 1: {...}, 2: {...}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy per client
    clients = sorted(client_metrics.keys())
    accs = [client_metrics[c].get("accuracy", 0) for c in clients]
    f1s = [client_metrics[c].get("f1_macro", 0) for c in clients]

    ax = axes[0]
    x = np.arange(len(clients))
    ax.bar(x - 0.15, accs, 0.3, label="Accuracy", color="steelblue")
    ax.bar(x + 0.15, f1s, 0.3, label="F1 Macro", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Client {c}" for c in clients])
    ax.set_ylabel("Score")
    ax.set_title("Per-Client Performance (Fairness)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Per-class F1 per client
    ax2 = axes[1]
    for c in clients:
        class_f1 = [client_metrics[c].get(f"{cls}_f1", 0) for cls in C.CLASS_NAMES]
        ax2.plot(C.CLASS_NAMES, class_f1, "-o", label=f"Client {c}")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Per-Class F1 by Client")
    ax2.legend()
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Fairness plot saved: {save_path}")


def plot_privacy_utility(pu_results: dict, save_path: str):
    """
    Plot accuracy vs privacy budget (epsilon).
    pu_results: {epsilon: accuracy}
    """
    epsilons = sorted(pu_results.keys())
    accs = [pu_results[e] for e in epsilons]

    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, accs, "ro-", linewidth=2, markersize=8)
    plt.xlabel("Privacy Budget (ε)")
    plt.ylabel("Accuracy")
    plt.title("Privacy-Utility Trade-off")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Privacy-utility plot saved: {save_path}")


def save_results_json(results: dict, path: str):
    """Persist results as JSON for reproducibility."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved: {path}")


def full_evaluation_report(model, test_loader, device, name, save_dir):
    """Complete evaluation pipeline for a single model."""
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n{'='*50}")
    print(f"  Evaluating: {name}")
    print(f"{'='*50}")

    metrics, preds, labels, probs = evaluate_model(model, test_loader, device, return_preds=True)

    # Print summary
    print(f"  Accuracy     : {metrics['accuracy']*100:.2f}%")
    print(f"  F1 (macro)   : {metrics['f1_macro']:.4f}")
    print(f"  AUC-ROC      : {metrics['auc_roc_macro']:.4f}")
    for cls in C.CLASS_NAMES:
        f1 = metrics.get(f"{cls}_f1", 0)
        print(f"    {cls:25s}  F1={f1:.4f}")

    # Confusion matrix
    plot_confusion_matrix(preds, labels, os.path.join(save_dir, f"{name}_cm.png"),
                          title=f"Confusion Matrix — {name}")

    # Save metrics
    save_results_json(metrics, os.path.join(save_dir, f"{name}_metrics.json"))

    return metrics
