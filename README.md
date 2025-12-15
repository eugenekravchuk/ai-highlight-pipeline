# ai-highlight-pipeline

AI-powered pipeline for automatic football (soccer) match highlight generation using audio–visual embeddings, self-attention, pseudo-labeling, and automatic voiceover generation.

## Overview

The system takes a full match video and produces a final highlight video with:

- Automatically selected key moments
- Intro & outro posters
- Background music
- AI-generated analytical voiceover

## Data

Raw match videos and related assets are stored on Google Drive:

[Google Drive dataset](https://drive.google.com/drive/folders/1YjgOv60ueh2B9aV_yE0qIY37qndXXhPB?usp=sharing)

## Research Background

This project is inspired by the following research papers:

1. [Unsupervised Video Highlight Detection by Learning from Audio and Visual Recurrence](https://arxiv.org/pdf/2407.13933)
2. [Joint Visual and Audio Learning for Video Highlight Detection (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Badamdorj_Joint_Visual_and_Audio_Learning_for_Video_Highlight_Detection_ICCV_2021_paper.pdf)
3. [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

## Pipeline Overview

The pipeline consists of the following stages:

### 1. Match Segmentation

- Split the full match video into fixed-length segments
- Extract corresponding audio chunks

### 2. Feature Extraction

- **Video embeddings**: R3D-18 (Kinetics-400 pretrained)
- **Audio embeddings**: PANNs (Cnn14)
- Audio preprocessing via ffmpeg (no librosa dependency)

### 3. Pseudo-Label Generation

- Pseudo Highlight Scores (PHS) computed from audio–visual recurrence
- No manual annotations required

### 4. Model Training / Inference

- Self-attention for audio and video streams
- Bimodal attention for cross-modal interaction
- Highlight probability prediction per segment

### 5. Highlight Rendering

- Global highlight duration budget
- Temporal smoothing and merging
- Video-only highlight rendering

### 6. Voiceover Generation

- Speech-to-text via Whisper
- Script generation using OpenAI GPT
- Text-to-speech via ElevenLabs
- Background music mixing with ducking

### 7. Final Assembly

- Intro & outro poster videos
- Crossfade transitions
- Audio replacement
- Single final MP4 output

## Requirements

- Python 3.10+
- FFmpeg (must be available in PATH)
- GPU optional (CPU supported)

### External Services (optional but recommended)

- OpenAI API (script generation)
- ElevenLabs API (voiceover TTS)
- API-Football (match metadata)

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# or
.venv\Scripts\activate           # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key
ELEVEN_API_KEY=your_elevenlabs_key
FOOTBALL_API_KEY=your_api_football_key
ELEVEN_VOICE=Bella
```

## Usage

Run the complete highlight + voiceover pipeline:

```bash
python scripts/full_pipeline.py \
  --match_video data/video.mp4 \
  --music music/bg.mp3 \
  --intro_img img/intro.png \
  --outro_img img/outro.png \
  --checkpoint checkpoints/av_highlight_model.pt \
  --embeddings_cache cache/cached_embeddings_full_match.npz
```

Final output will be saved to:

```
output/final_assembly/FINAL.mp4
```

### Optional Arguments

| Argument | Description |
|----------|-------------|
| `--device` | Force cpu or cuda |
| `--epochs` | Training epochs (if no checkpoint) |
| `--batch_size` | Training batch size |
| `--global_budget_s` | Total highlight duration |
| `--seg_s` | Segment length in seconds |

## Project Structure

```
ai-highlight-pipeline/
│
├── scripts/
│   ├── full_pipeline.py
│   ├── generate_voiceover.py
│
├── core/
│   ├── attention.py
│   ├── classifier.py
│
├── highlight_utils/
│   ├── render_highlights.py
│   ├── media_split.py
│
├── checkpoints/
├── cache/
├── output/
├── data/
├── music/
├── img/
└── requirements.txt
```

## Notes & Limitations

- Highlight quality strongly depends on audio intensity and crowd reactions
- Voiceover duration depends on generated script length
- ElevenLabs free tier has request limits
- GPU is recommended for faster processing

## Future Improvements

- Multilingual voiceover support
- Player name entity correction
- Emotion-aware TTS control
- Live match processing
- Automatic subtitles

## License

MIT
