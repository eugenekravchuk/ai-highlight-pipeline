# ai-highlight-pipeline

# Our work mostly base on this papers:
  1. Unsupervised Video Highlight Detection by Learning from
      Audio and Visual Recurrence: https://arxiv.org/pdf/2407.13933
  2. Joint Visual and Audio Learning for Video Highlight Detection: 
      https://openaccess.thecvf.com/content/ICCV2021/papers/Badamdorj_Joint_Visual_and_Audio_Learning_for_Video_Highlight_Detection_ICCV_2021_paper.pdf
  3. Attention Is All You Need:
    https://arxiv.org/pdf/1706.03762

# How to use

First - create virtual environment:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run a script on a video of your choice:

```
python highlight-pipeline/run.py \
  --match_video "./data/matches/full_match_beic_barcelona.mp4" \
```

