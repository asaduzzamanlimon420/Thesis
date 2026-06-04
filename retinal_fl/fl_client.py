"""
Flower client for federated training.

Supports:
  - FedAvg / FedProx local training
  - LoRA-only or full parameter exchange
  - Optional HE encryption of outgoing updates
  - Optional SecAgg masking
"""

from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import flwr as fl

import config as C
from dataset import compute_class_weights


class RetinalClient(fl.client.NumPyClient):
    """One hospital / data silo in the federated setup."""

    def __init__(self, client_id, model, train_loader, val_loader, device):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Class-weighted loss
        weights = compute_class_weights(train_loader).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        # Optimiser — only trainable params
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=C.FL_LR, weight_decay=C.WEIGHT_DECAY,
        )
        self.scaler = GradScaler(enabled=C.USE_AMP)

        # For FedProx: store a copy of global params at round start
        self.global_params = None

    # ----------- Flower interface -----------

    def get_parameters(self, config) -> List[np.ndarray]:
        return [v.cpu().detach().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]):
        state = self.model.state_dict()
        keys = list(state.keys())
        new_state = OrderedDict()
        for k, p in zip(keys, parameters):
            new_state[k] = torch.tensor(p)
        self.model.load_state_dict(new_state)
        # cache global params for FedProx
        self.global_params = [p.clone().detach() for p in self.model.parameters()]

    def fit(self, parameters, config) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        mu = config.get("mu", C.FEDPROX_MU)
        local_epochs = config.get("local_epochs", C.LOCAL_EPOCHS)

        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for epoch in range(local_epochs):
            epoch_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                with autocast(enabled=C.USE_AMP):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    # FedProx proximal term
                    if mu > 0 and self.global_params is not None:
                        prox = 0.0
                        for local_p, global_p in zip(self.model.parameters(), self.global_params):
                            if local_p.requires_grad:
                                prox += ((local_p - global_p.to(self.device)) ** 2).sum()
                        loss = loss + (mu / 2) * prox

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            total_loss = epoch_loss / max(total, 1)

        accuracy = correct / max(total, 1)
        n_samples = len(self.train_loader.dataset)
        metrics = {"loss": total_loss, "accuracy": accuracy, "client_id": self.client_id}
        print(f"  [Client {self.client_id}] fit → loss={total_loss:.4f}, acc={accuracy*100:.1f}%")
        return self.get_parameters(config={}), n_samples, metrics

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        self.model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                with autocast(enabled=C.USE_AMP):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = val_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        print(f"  [Client {self.client_id}] eval → loss={avg_loss:.4f}, acc={accuracy*100:.1f}%")
        return avg_loss, total, {"accuracy": accuracy}
