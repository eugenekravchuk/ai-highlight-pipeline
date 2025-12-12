import torch
import torchvision
from torchvision import transforms
from torchvision.io import read_video
from decord import VideoReader, cpu
from torchvision.models.video import r3d_18, R3D_18_Weights
import time
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

weights = R3D_18_Weights.KINETICS400_V1   # або R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights)

model.fc = torch.nn.Identity()
model.eval().to(device)

DEFAULT_MEAN = [0.43216, 0.394666, 0.37645]
DEFAULT_STD  = [0.22803, 0.22145, 0.216989]

default_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 171)),
    transforms.CenterCrop((112, 112)),
    transforms.ToTensor(),                   # -> (C, H, W) float32 [0,1]
    transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
])


def load_clip_from_file(video_path: str,
                        start_frame: int,
                        clip_len: int = 16,
                        device: str = 'cpu',
                        preprocess = None) -> torch.Tensor:

    if preprocess is None:
        preprocess = default_preprocess

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    if start_frame >= total_frames:
        raise ValueError(f"start_frame ({start_frame}) >= total_frames ({total_frames})")

    end_frame = start_frame + clip_len

    if end_frame <= total_frames:
        frames_np = vr.get_batch(range(start_frame, end_frame)).asnumpy()
    else:
        available = vr.get_batch(range(start_frame, total_frames)).asnumpy()
        last = available[-1:]
        n_pad = end_frame - total_frames
        frames_np = np.concatenate([available, np.repeat(last, repeats=n_pad, axis=0)], axis=0)

    processed = []
    for f in frames_np:
        t = preprocess(f)
        processed.append(t)

    frames_t = torch.stack(processed, dim=0)            # (T, C, H, W)
    frames_t = frames_t.permute(1, 0, 2, 3).contiguous() # (C, T, H, W)

    return frames_t.to(device, dtype=torch.float32)

def get_video_embeddings():
    return model(batch)

with torch.no_grad():
    clip = load_clip_from_file("Emine.mp4", start_frame=0, clip_len=3).to(device)
    clip2 = load_clip_from_file("Emine.mp4", start_frame=0, clip_len=3).to(device)
    clip3 = load_clip_from_file("Emine.mp4", start_frame=0, clip_len=3).to(device)
    batch = torch.stack([clip, clip2, clip3], dim=0)

    time1 = time.time()
    features = model(batch)
    time2 = time.time()
    print(batch.shape)

