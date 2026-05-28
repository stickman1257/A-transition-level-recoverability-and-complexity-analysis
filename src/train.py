from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model import reconstruction_loss


@dataclass
class TrainerConfig:
    epochs: int = 50
    learning_rate: float = 1e-3
    beta: float = 1.0
    beta_start: float = 1.0
    beta_warmup_epochs: int = 0
    l1_weight: float = 0.0
    corr_weight: float = 0.0
    patience: int = 10
    device: str = "cpu"


class CVAETrainer:
    def __init__(self, model: nn.Module, config: TrainerConfig) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> tuple[nn.Module, list[dict]]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        history: list[dict] = []
        best_state = copy.deepcopy(self.model.state_dict())
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            running_loss = 0.0

            for target, condition in tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.epochs}"):
                target = target.to(self.device)
                condition = condition.to(self.device)

                optimizer.zero_grad()
                recon, _, _ = self.model(target, condition)
                loss, _ = reconstruction_loss(
                    recon,
                    target,
                    l1_weight=self.config.l1_weight,
                    corr_weight=self.config.corr_weight,
                )
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * target.size(0)

            train_loss = running_loss / len(train_loader.dataset)
            record = {"epoch": epoch, "train_loss": train_loss}

            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                record["val_loss"] = val_loss
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.config.patience:
                    history.append(record)
                    break
            else:
                if train_loss < best_loss:
                    best_loss = train_loss
                    best_state = copy.deepcopy(self.model.state_dict())

            history.append(record)

        self.model.load_state_dict(best_state)
        return self.model, history

    def evaluate(self, data_loader: DataLoader, epoch: int | None = None) -> float:
        del epoch
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for target, condition in data_loader:
                target = target.to(self.device)
                condition = condition.to(self.device)
                recon, _, _ = self.model(target, condition, deterministic=True)
                loss, _ = reconstruction_loss(
                    recon,
                    target,
                    l1_weight=self.config.l1_weight,
                    corr_weight=self.config.corr_weight,
                )
                total_loss += loss.item() * target.size(0)
        return total_loss / len(data_loader.dataset)

    def save_checkpoint(self, path: str | Path, payload: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
