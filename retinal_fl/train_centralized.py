"""
Centralised training baseline.
Trains on the combined dataset (all clients merged) for comparison.
"""

import os, time, json
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import config as C
from dataset import load_combined_data, compute_class_weights
from models import build_model
from evaluate import evaluate_model, full_evaluation_report, plot_training_curves


def train_centralised(model_name="retfound"):
    """Train a model on the combined (all-client) dataset."""

    print(f"\n{'#'*60}")
    print(f"  CENTRALISED BASELINE  —  {model_name}")
    print(f"{'#'*60}\n")

    device = torch.device(C.DEVICE)
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)

    # Data
    print("Loading combined dataset …")
    train_loader, val_loader, test_loader = load_combined_data()
    print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")

    # Model
    model = build_model(model_name).to(device)

    # Loss + optimiser
    class_weights = compute_class_weights(train_loader).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=C.CENTRAL_LR, weight_decay=C.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=C.CENTRAL_EPOCHS)
    scaler = GradScaler(enabled=C.USE_AMP)

    # Training loop
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_acc = 0.0
    save_dir = os.path.join(C.OUTPUT_DIR, f"centralised_{model_name}")
    os.makedirs(save_dir, exist_ok=True)

    t0 = time.time()
    for epoch in range(1, C.CENTRAL_EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{C.CENTRAL_EPOCHS}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(enabled=C.USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total*100:.1f}%")

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        val_acc = val_metrics["accuracy"]
        val_loss_proxy = 1.0 - val_acc  # proxy for plotting

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss_proxy)
        history["val_accuracy"].append(val_acc)

        print(f"  Epoch {epoch:3d}  train_loss={train_loss:.4f}  "
              f"train_acc={train_acc*100:.1f}%  val_acc={val_acc*100:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    elapsed = time.time() - t0
    print(f"\n  Training completed in {elapsed/60:.1f} min.  Best val acc: {best_val_acc*100:.2f}%")

    # Load best and evaluate on test set
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth"), weights_only=True))
    test_metrics = full_evaluation_report(model, test_loader, device, f"centralised_{model_name}", save_dir)

    # Save training curves
    plot_training_curves(
        {"round": history["epoch"], "loss": history["train_loss"], "accuracy": history["val_accuracy"]},
        save_dir, prefix=f"centralised_{model_name}",
    )

    # Save all results
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return test_metrics, model


if __name__ == "__main__":
    # Train RETFound centralised baseline
    metrics, _ = train_centralised("retfound")

    # Comparison models
    for m in C.COMPARISON_MODELS:
        print(f"\n--- Comparison: {m} ---")
        try:
            train_centralised(m)
        except Exception as e:
            print(f"  [ERROR] {m}: {e}")
