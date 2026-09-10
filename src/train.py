"""CRNN training entry point."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from tqdm import tqdm

from .config import Config
from .dataset import create_data_loaders
from .model import create_model
from .utils import calculate_accuracy, decode_predictions


class Trainer:
    def __init__(self, train_dir: Path, val_dir: Path, output_dir: Path, epochs: int, batch_size: int) -> None:
        self.device = torch.device(Config.DEVICE)
        self.epochs = epochs
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = create_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=5)
        self.criterion = nn.CTCLoss(blank=Config.NUM_CLASSES - 1, zero_infinity=True)
        self.train_loader, self.val_loader = create_data_loaders(train_dir, val_dir, batch_size)

    def _run(self, loader, training: bool):
        self.model.train(training)
        losses, predictions, targets = [], [], []
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for images, labels, label_lengths, raw_labels in tqdm(loader, leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                input_lengths = torch.full((images.size(0),), output.size(0), dtype=torch.long, device=self.device)
                loss = self.criterion(output, labels, input_lengths, label_lengths)
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.optimizer.step()
                losses.append(loss.item())
                if not training:
                    predictions.extend(decode_predictions(output))
                    targets.extend(raw_labels)
        mean_loss = sum(losses) / max(1, len(losses))
        return mean_loss, calculate_accuracy(predictions, targets) if not training else 0.0

    def train(self) -> None:
        best = -1.0
        for epoch in range(1, self.epochs + 1):
            train_loss, _ = self._run(self.train_loader, True)
            val_loss, accuracy = self._run(self.val_loader, False)
            self.scheduler.step(val_loss)
            checkpoint = {"epoch": epoch, "model_state_dict": self.model.state_dict(), "val_accuracy": accuracy}
            torch.save(checkpoint, self.output_dir / "latest.pth")
            if accuracy > best:
                best = accuracy
                torch.save(checkpoint, self.output_dir / "best_model.pth")
            print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_accuracy={accuracy:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 CAPTCHA CRNN")
    parser.add_argument("--train-dir", type=Path, default=Config.TRAIN_DATA_DIR)
    parser.add_argument("--val-dir", type=Path, default=Config.VAL_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Config.MODEL_SAVE_PATH)
    parser.add_argument("--epochs", type=int, default=Config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    args = parser.parse_args()
    Trainer(args.train_dir, args.val_dir, args.output_dir, args.epochs, args.batch_size).train()


if __name__ == "__main__":
    main()
