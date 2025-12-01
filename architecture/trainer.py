"""Training utilities for the audio-visual highlight detector."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from av_model import AVHighlightDetector, masked_bce_loss


class AudioVisualDataset(Dataset):
    def __init__(
        self,
        audio_embeddings: Sequence[np.ndarray],
        visual_embeddings: Sequence[np.ndarray],
        label_sequences: Sequence[np.ndarray],
    ) -> None:
        if not (len(audio_embeddings) == len(visual_embeddings) == len(label_sequences)):
            raise ValueError("Audio, visual, and label lists must have the same length")
        self.audio = [np.asarray(a, dtype=np.float32) for a in audio_embeddings]
        self.visual = [np.asarray(v, dtype=np.float32) for v in visual_embeddings]
        self.labels = [np.asarray(l, dtype=np.float32) for l in label_sequences]

    def __len__(self) -> int:
        return len(self.audio)

    def __getitem__(self, idx: int):
        audio = torch.tensor(self.audio[idx], dtype=torch.float32)
        visual = torch.tensor(self.visual[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        return audio, visual, labels


def collate_av_batch(batch):
    audio, visual, labels = zip(*batch)
    batch_size = len(batch)
    audio_dim = audio[0].shape[-1]
    visual_dim = visual[0].shape[-1]
    max_len = max(item.shape[0] for item in audio)

    audio_tensor = torch.zeros((batch_size, max_len, audio_dim), dtype=torch.float32)
    visual_tensor = torch.zeros((batch_size, max_len, visual_dim), dtype=torch.float32)
    label_tensor = torch.zeros((batch_size, max_len), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, (a, v, l) in enumerate(zip(audio, visual, labels)):
        T = a.shape[0]
        audio_tensor[i, :T] = a
        visual_tensor[i, :T] = v
        label_tensor[i, :T] = l
        mask[i, :T] = True

    return audio_tensor, visual_tensor, label_tensor, mask


def split_datasets(
    audio_embeddings: Sequence[np.ndarray],
    visual_embeddings: Sequence[np.ndarray],
    labels_dict: Dict[int, np.ndarray],
    train_ratio: float = 0.8,
) -> Tuple[AudioVisualDataset, AudioVisualDataset]:
    if len(audio_embeddings) != len(visual_embeddings):
        raise ValueError("Audio and visual embeddings must have the same number of samples")

    labels_list: List[np.ndarray] = []
    for idx, emb in enumerate(audio_embeddings):
        if idx not in labels_dict:
            raise ValueError(f"Missing pseudo labels for sample {idx}")
        label = np.asarray(labels_dict[idx], dtype=np.float32)
        if label.shape[0] != emb.shape[0]:
            raise ValueError(f"Label length mismatch for sample {idx}")
        labels_list.append(label)

    num_samples = len(audio_embeddings)
    split_idx = max(1, int(num_samples * train_ratio))

    train_dataset = AudioVisualDataset(
        audio_embeddings[:split_idx],
        visual_embeddings[:split_idx],
        labels_list[:split_idx],
    )
    val_dataset = AudioVisualDataset(
        audio_embeddings[split_idx:],
        visual_embeddings[split_idx:],
        labels_list[split_idx:],
    )
    return train_dataset, val_dataset


class HighlightTrainer:
    def __init__(
        self,
        audio_dim: int,
        visual_dim: int,
        *,
        model_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.1,
        hidden_dim: int = 512,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.model = AVHighlightDetector(
            audio_dim=audio_dim,
            visual_dim=visual_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.best_val = float("inf")

    def _step(self, batch, train: bool = True) -> float:
        audio, visual, labels, mask = batch
        audio = audio.to(self.device)
        visual = visual.to(self.device)
        labels = labels.to(self.device)
        mask = mask.to(self.device)

        logits = self.model(audio, visual, mask)
        loss = masked_bce_loss(logits, labels, mask)

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
        return float(loss.item())

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total = 0.0
        for batch in tqdm(loader, desc="Training"):
            total += self._step(batch, train=True)
        return total / max(len(loader), 1)

    def validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating"):
                total += self._step(batch, train=False)
        return total / max(len(loader), 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        ckpt = Path(checkpoint_path) if checkpoint_path else None
        if ckpt:
            ckpt.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if ckpt and val_loss < self.best_val:
                self.best_val = val_loss
                torch.save({"model_state": self.model.state_dict(), "optimizer_state": self.optimizer.state_dict()}, ckpt)
                print(f"✓ Checkpoint saved to {ckpt}")

    def load(self, checkpoint_path: str) -> None:
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        if "optimizer_state" in state:
            self.optimizer.load_state_dict(state["optimizer_state"])