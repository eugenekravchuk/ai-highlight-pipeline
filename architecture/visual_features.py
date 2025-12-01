"""Visual feature extraction utilities for synchronized audio-visual processing."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torchvision.io import read_video
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

try:  # TorchVision optional imports (guarded for portability)
    from torchvision.models import resnet34, ResNet34_Weights
except Exception:  # pragma: no cover - handled at runtime
    resnet34 = None
    ResNet34_Weights = None

try:
    from torchvision.models.clip import clip_vit_b32, CLIP_ViT_B_32_Weights
except Exception:  # pragma: no cover
    clip_vit_b32 = None
    CLIP_ViT_B_32_Weights = None

try:
    from torchvision.models.video import slowfast_r50, SlowFast_R50_Weights
    HAS_SLOWFAST = True
except Exception:  # pragma: no cover
    slowfast_r50 = None
    SlowFast_R50_Weights = None
    HAS_SLOWFAST = False


def _resample_frames(frames: torch.Tensor, current_fps: float, target_fps: Optional[float]) -> tuple[torch.Tensor, float]:
    if not target_fps or current_fps <= 0:
        return frames, current_fps
    if abs(current_fps - target_fps) < 1e-2:
        return frames, current_fps

    total_frames = frames.shape[0]
    duration = total_frames / current_fps if current_fps > 0 else total_frames
    target_frame_count = max(1, int(round(duration * target_fps)))
    idx = torch.linspace(0, max(total_frames - 1, 0), target_frame_count)
    idx = idx.round().long().clamp(0, total_frames - 1)
    resampled = frames[idx]
    return resampled, target_fps


def load_video_frames(video_path: str | Path, target_fps: Optional[float] = None) -> tuple[torch.Tensor, float]:
    """Load RGB frames from a video file as a Tensor of shape (T, H, W, C)."""

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    frames, _, info = read_video(str(video_path), pts_unit="sec")
    fps = float(info.get("video_fps", 0.0) or 0.0)
    frames, fps = _resample_frames(frames, fps, target_fps)
    if fps <= 0:
        fps = target_fps or 25.0
    return frames, fps


def split_frames_into_clips(
    frames: torch.Tensor,
    *,
    num_clips: Optional[int] = None,
    clip_duration: Optional[float] = None,
    fps: Optional[float] = None,
) -> List[torch.Tensor]:
    """Split a tensor of frames into temporal clips."""

    total_frames = frames.shape[0]
    if total_frames == 0:
        raise ValueError("Video has no frames to split")

    if num_clips is None:
        if clip_duration is None or fps is None:
            raise ValueError("clip_duration and fps required when num_clips is not provided")
        frames_per_clip = max(1, int(round(clip_duration * fps)))
        num_clips = math.ceil(total_frames / frames_per_clip)
    else:
        frames_per_clip = math.ceil(total_frames / max(1, num_clips))

    clips: List[torch.Tensor] = []
    for idx in range(num_clips):
        start = idx * frames_per_clip
        end = min(total_frames, start + frames_per_clip)
        if start >= total_frames:
            chunk = frames[-1:].clone()
        else:
            chunk = frames[start:end]
        clips.append(chunk.contiguous())
    return clips


class VisualFeatureExtractor:
    """Wraps multiple visual backbones for clip-level feature extraction."""

    def __init__(
        self,
        *,
        backbone: str = "resnet34",
        device: str = "cuda",
        frame_batch_size: int = 32,
    ) -> None:
        self.backbone = backbone.lower()
        self.device = torch.device(device)
        self.frame_batch_size = max(1, frame_batch_size)

        if self.backbone == "resnet34":
            if resnet34 is None or ResNet34_Weights is None:
                raise ImportError("torchvision.models.resnet34 is not available in this environment")
            weights = ResNet34_Weights.IMAGENET1K_V1
            model = resnet34(weights=weights)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            self.model = model.to(self.device).eval()
            meta = weights.meta
            self.resize = 256
            self.crop = 224
            self.mean = meta.get("mean", [0.485, 0.456, 0.406])
            self.std = meta.get("std", [0.229, 0.224, 0.225])
            self.feature_dim = feature_dim
            self._mode = "2d"
        elif self.backbone == "clip_vit_b32":
            if clip_vit_b32 is None or CLIP_ViT_B_32_Weights is None:
                raise ImportError("torchvision.models.clip_vit_b32 is not available in this environment")
            weights = CLIP_ViT_B_32_Weights.OPENAI
            clip_model = clip_vit_b32(weights=weights)
            self.model = clip_model.visual.eval().to(self.device)
            self.resize = 224
            self.crop = 224
            self.mean = [0.48145466, 0.4578275, 0.40821073]
            self.std = [0.26862954, 0.26130258, 0.27577711]
            self.feature_dim = self.model.proj.shape[1] if hasattr(self.model, "proj") else 512
            self._mode = "2d"
        elif self.backbone == "slowfast_r50":
            if not HAS_SLOWFAST:
                raise ImportError(
                    "slowfast_r50 is unavailable. Install TorchVision >= 0.15 with video models or PyTorchVideo."
                )
            weights = SlowFast_R50_Weights.KINETICS400_V1
            model = slowfast_r50(weights=weights)
            model.classifier = nn.Identity()
            self.model = model.to(self.device).eval()
            self.video_transform = weights.transforms()
            self.feature_dim = 2304
            self._mode = "slowfast"
        else:
            raise ValueError(f"Unsupported visual backbone: {backbone}")

    def _prepare_2d_frames(self, clip: torch.Tensor) -> torch.Tensor:
        frames = clip.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
        processed = []
        for frame in frames:
            frame = TF.resize(frame, self.resize, interpolation=InterpolationMode.BILINEAR, antialias=True)
            frame = TF.center_crop(frame, self.crop)
            frame = TF.normalize(frame, self.mean, self.std)
            processed.append(frame)
        return torch.stack(processed, dim=0)

    def _encode_clip_resnet_like(self, clip: torch.Tensor) -> np.ndarray:
        frames = self._prepare_2d_frames(clip)
        outputs: List[torch.Tensor] = []
        for start in range(0, frames.shape[0], self.frame_batch_size):
            batch = frames[start : start + self.frame_batch_size].to(self.device)
            with torch.no_grad():
                feat = self.model(batch)
            outputs.append(feat.detach().cpu())
        stacked = torch.cat(outputs, dim=0)
        clip_feat = stacked.mean(dim=0).view(-1)
        return clip_feat.numpy()

    def _encode_clip_slowfast(self, clip: torch.Tensor) -> np.ndarray:
        clip = clip.to(torch.float32)
        transformed = self.video_transform(clip)
        if isinstance(transformed, (list, tuple)):
            inputs = [pathway.unsqueeze(0).to(self.device) for pathway in transformed]
        else:
            inputs = transformed.unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(inputs)
        return feat.squeeze(0).detach().cpu().numpy()

    def encode_clips(self, clips: Sequence[torch.Tensor]) -> np.ndarray:
        features: List[np.ndarray] = []
        for clip in clips:
            if clip.numel() == 0:
                continue
            if self._mode == "slowfast":
                feat = self._encode_clip_slowfast(clip)
            else:
                feat = self._encode_clip_resnet_like(clip)
            features.append(feat)
        if not features:
            raise RuntimeError("No visual features could be extracted from the provided clips")
        return np.stack(features, axis=0)


def get_visual_embeddings_list(
    video_paths: Sequence[str | Path],
    clip_counts: Sequence[int],
    *,
    clip_duration: float,
    target_fps: float = 16.0,
    device: str = "cuda",
    backbone: str = "resnet34",
    cache_dir: Optional[str | Path] = None,
) -> List[np.ndarray]:
    """Extract visual embeddings per video aligned with audio clip counts."""

    if len(video_paths) != len(clip_counts):
        raise ValueError("video_paths and clip_counts must have the same length")

    cache_dir_path = Path(cache_dir) if cache_dir else None
    if cache_dir_path:
        cache_dir_path.mkdir(parents=True, exist_ok=True)

    extractor = VisualFeatureExtractor(backbone=backbone, device=device)
    embeddings: List[np.ndarray] = []

    for path, expected_clips in zip(video_paths, clip_counts):
        frames, fps = load_video_frames(path, target_fps=target_fps)
        clips = split_frames_into_clips(frames, num_clips=expected_clips, clip_duration=clip_duration, fps=fps)
        clip_features = extractor.encode_clips(clips)
        if clip_features.shape[0] != expected_clips:
            # Simple resampling to match audio clip count
            idx = np.linspace(0, clip_features.shape[0] - 1, expected_clips).round().astype(int)
            clip_features = clip_features[idx]
        embeddings.append(clip_features)

        if cache_dir_path:
            cache_path = cache_dir_path / f"{Path(path).stem}_{extractor.backbone}.npy"
            np.save(cache_path, clip_features)

    return embeddings