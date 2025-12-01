"""Dataset utilities for benchmark evaluation (YouTube, TVSum, QVHighlights)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_array(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing array file: {path}")
    if path.suffix in {".npy", ".npz"}:
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            # Expect single array stored under 'arr_0' or explicit key
            keys = list(data.keys())
            if len(keys) != 1:
                raise ValueError(f"Ambiguous npz contents in {path}; expected single array")
            return data[keys[0]]
        return data
    raise ValueError(f"Unsupported array format: {path}")


def _ensure_list(arr: np.ndarray) -> List[np.ndarray]:
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return [np.asarray(item, dtype=np.float32) for item in arr.tolist()]
    return [np.asarray(arr, dtype=np.float32)]


@dataclass
class SampleTriplet:
    audio_path: Path
    visual_path: Path
    label_path: Path
    clip_id: str


class EmbeddingDirectoryDataset(Dataset):
    """Loads per-video audio/visual embeddings and labels from directories.

    Directory layout (per split):
        split_root/
            audio/
                000.npy, 001.npy, ...
            visual/
                000.npy, 001.npy, ...
            labels/
                000.npy, 001.npy, ...
    Filenames without extension act as sample IDs and must match across dirs.
    """

    def __init__(self, root: str | Path, split: str) -> None:
        self.root = Path(root)
        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        self.audio_dir = split_dir / "audio"
        self.visual_dir = split_dir / "visual"
        self.labels_dir = split_dir / "labels"

        for d in (self.audio_dir, self.visual_dir, self.labels_dir):
            if not d.exists():
                raise FileNotFoundError(f"Directory missing: {d}")

        audio_ids = {p.stem for p in self.audio_dir.glob("*.npy")}
        visual_ids = {p.stem for p in self.visual_dir.glob("*.npy")}
        label_ids = {p.stem for p in self.labels_dir.glob("*.npy")}
        ids = sorted(audio_ids & visual_ids & label_ids)
        if not ids:
            raise RuntimeError(f"No overlapping samples in {split_dir}")
        self.samples: List[SampleTriplet] = [
            SampleTriplet(
                audio_path=self.audio_dir / f"{sid}.npy",
                visual_path=self.visual_dir / f"{sid}.npy",
                label_path=self.labels_dir / f"{sid}.npy",
                clip_id=sid,
            )
            for sid in ids
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        audio = _load_array(sample.audio_path)
        visual = _load_array(sample.visual_path)
        labels = _load_array(sample.label_path)
        return (
            torch.tensor(audio, dtype=torch.float32),
            torch.tensor(visual, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
            sample.clip_id,
        )


class ManifestDataset(Dataset):
    """Loads samples defined inside a manifest JSON file.

    Manifest schema:
        {
            "samples": [
                {
                    "clip_id": "video_0001",
                    "split": "train",
                    "audio": "embeddings/audio/video_0001.npy",
                    "visual": "embeddings/visual/video_0001.npy",
                    "labels": "labels/video_0001.npy"
                },
                ...
            ]
        }
    Paths are resolved relative to the manifest file location.
    """

    def __init__(self, manifest_path: str | Path, split: str) -> None:
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        base = manifest_path.parent
        samples = [
            SampleTriplet(
                audio_path=base / entry["audio"],
                visual_path=base / entry["visual"],
                label_path=base / entry["labels"],
                clip_id=entry.get("clip_id") or Path(entry["audio"]).stem,
            )
            for entry in payload.get("samples", [])
            if entry.get("split", "train") == split
        ]
        if not samples:
            raise RuntimeError(f"No samples for split '{split}' inside {manifest_path}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        audio = _load_array(sample.audio_path)
        visual = _load_array(sample.visual_path)
        labels = _load_array(sample.label_path)
        return (
            torch.tensor(audio, dtype=torch.float32),
            torch.tensor(visual, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
            sample.clip_id,
        )


@dataclass
class DatasetSplit:
    name: str
    root: str
    splits: Sequence[str]


@dataclass
class BenchmarkDataset:
    name: str
    root: str
    train_split: str
    val_split: str
    test_split: str
    epochs: int
    lr: float
    num_heads: int
    num_self_attn: int
    notes: str = ""


BENCHMARK_DATASETS: Dict[str, BenchmarkDataset] = {
    "youtube_highlights": BenchmarkDataset(
        name="YouTube Highlights",
        root="datasets/youtube_highlights",
        train_split="train",
        val_split="val",
        test_split="test",
        epochs=10,
        lr=2e-4,
        num_heads=4,
        num_self_attn=2,
        notes="Default hyperparameters based on paper Appendix B.",
    ),
    "tvsum": BenchmarkDataset(
        name="TVSum",
        root="datasets/tvsum",
        train_split="train",
        val_split="val",
        test_split="test",
        epochs=12,
        lr=1e-4,
        num_heads=4,
        num_self_attn=2,
        notes="Videos trimmed to 2s clips before embedding extraction.",
    ),
    "qvhighlights": BenchmarkDataset(
        name="QVHighlights",
        root="datasets/qvhighlights",
        train_split="train",
        val_split="val",
        test_split="test",
        epochs=15,
        lr=1e-4,
        num_heads=8,
        num_self_attn=3,
        notes="Long-form videos require higher capacity cross-attention heads.",
    ),
}


def get_benchmark(name: str) -> BenchmarkDataset:
    key = name.lower()
    if key not in BENCHMARK_DATASETS:
        raise KeyError(f"Unknown benchmark '{name}'. Available: {list(BENCHMARK_DATASETS)}")
    return BENCHMARK_DATASETS[key]


def dataset_from_config(
    config: BenchmarkDataset,
    split: str,
    use_manifest: bool = False,
    manifest_filename: str = "manifest.json",
):
    if use_manifest:
        manifest_path = Path(config.root) / manifest_filename
        return ManifestDataset(manifest_path, split)
    return EmbeddingDirectoryDataset(config.root, split)
