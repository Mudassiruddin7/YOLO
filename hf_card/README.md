---
license: agpl-3.0
task_categories:
- object-detection
tags:
- visdrone
- small-object-detection
- yolov10
- uav
- drone-imagery
pretty_name: P2-Graft runs, results and weights
---

<div align="center">

# P2-Graft: runs, results and weights

### When does a stride-4 prediction level help a pretrained YOLOv10 on drone images?

[![Code](https://img.shields.io/badge/code-GitHub-181717?logo=github)](https://github.com/Mudassiruddin7/YOLO)
[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b)](https://huggingface.co/datasets/mudassiruddin/yolo/resolve/main/paper/main.pdf)
[![Data](https://img.shields.io/badge/data-VisDrone--DET2019-0072B2)](https://github.com/VisDrone/VisDrone-Dataset)
[![GPU](https://img.shields.io/badge/trained%20on-RTX%20PRO%206000%20Blackwell-76b900)](#reproduce)

<img src="https://huggingface.co/datasets/mudassiruddin/yolo/resolve/main/assets/fig_architecture.png" width="760" alt="P2-Graft: confident cells of every prediction level on a VisDrone crop">

*Quratulain Nayeem, Fahmina Taranum, Mohammed Mudassir Uddin, Sara Rizwan*

</div>

This dataset repository holds everything behind the paper: per-run results, training curves, saved test predictions and the final weights of the 51 runs reported there.
A stride-4 branch is grafted onto a pretrained YOLOv10 so that the original outputs stay unchanged at the start, and the new branch is initialised from the pretrained stride-8 weights.

**Jump to:** [Results](#results) · [Files](#files) · [Use the weights](#use-the-weights) · [Reproduce](#reproduce)

---

## Results

AP on **VisDrone test-dev** (primary), validation AP in the paper. One recipe, one GPU, last-epoch weights, gains paired by seed.

| Setting | Baseline | P2-Graft | Gain | Seeds |
|---|---|---|---|---|
| YOLOv10n, 640 px, 90 epochs | 14.09 | **15.77** | **+1.68** | 3 |
| YOLOv10n, 896 px, 90 epochs | 18.16 | **18.74** | **+0.58** | 3 |
| YOLOv10n, 1280 px, 90 epochs | 22.45 | 22.32 | **−0.13** | 1 |
| YOLOv10n, 640 px, 150 epochs | 14.46 | **15.63** | **+1.17** | 1 |
| YOLOv10s, 640 px, 90 epochs | 17.46 | **18.19** | **+0.73** | 3 |

<p align="center"><img src="https://huggingface.co/datasets/mudassiruddin/yolo/resolve/main/assets/fig_scaling.png" width="700" alt="gain by input size and accuracy against compute"></p>

The gain is largest at 640 px and shrinks as the input grows; at equal GFLOPs a plain baseline at higher resolution is more accurate. On YOLOv10s the per-seed gains are +0.06, +0.19 and +1.94.

<details>
<summary><b>Validation AP against published results</b> (ours in bold)</summary>

| Model | Input | Params (M) | AP | AP<sub>50</sub> |
|---|---|---|---|---|
| YOLOv8-N | 640 | 3.2 | 19.6 | 33.3 |
| FBRT-YOLO-N | 640 | 0.9 | 20.2 | 34.4 |
| YOLOv10-S | 640 | 7.2 | 23.8 | 39.3 |
| FBRT-YOLO-S | 640 | 2.9 | 25.9 | 42.4 |
| DroneScan-YOLO | 1280 | 10.23 | 35.6 | 55.3 |
| **P2-Graft, YOLOv10n** | **640** | **2.41** | **20.64** | **35.96** |
| **P2-Graft, YOLOv10s** | **640** | **7.58** | **23.29** | **39.86** |
| **YOLOv10n baseline** | **1536** | **2.27** | **30.29** | **49.42** |

Published numbers are quoted from [FBRT-YOLO](https://arxiv.org/abs/2504.20670) and [DroneScan-YOLO](https://arxiv.org/abs/2604.13278) with their own schedules; the rows in bold are 90-epoch runs.
</details>

<details>
<summary><b>What matters inside the branch</b> (35 epochs, 640 px)</summary>

<p align="center"><img src="https://huggingface.co/datasets/mudassiruddin/yolo/resolve/main/assets/fig_ablation.png" width="700" alt="variants of the branch and the YOLOv10s seeds"></p>

The warm start, copying the pretrained stride-8 block and head into the new level, is what makes the branch useful: +0.92 test-dev AP against +0.26 for a random start.
</details>

<details>
<summary><b>Qualitative examples</b> (validation crops)</summary>

<p align="center"><img src="https://huggingface.co/datasets/mudassiruddin/yolo/resolve/main/assets/fig_qualitative.png" width="700" alt="ground truth, baseline and P2-Graft"></p>

Green: match, red: false alarm, yellow: missed. The two crops show the largest gain in found objects.
</details>

## Files

```
README.md                         this card
assets/                           figures of the paper
paper/main.pdf                    the paper
results/                          runs_summary.csv (the 51 runs), final_runs.csv (all metrics), contrasts_session_{A,B,C}.csv
sessionA-20261104-0048/           896 px, 90 epochs, 3 seeds: model and baseline
sessionB-20261106-1639/           1280 px, 150-epoch schedule, batch-32 control
sessionC-20261107-0846/           YOLOv10s, 3 seeds
  runs/<run id>/                  result.json, results.csv (training curve), args.yaml,
                                  weights/last.pt (final weights), preds_*.npz (saved predictions)
  results/                        leaderboard.csv, contrasts.csv (image-bootstrap intervals), all_runs.csv
  logs/                           worker and resume logs
```

Run ids read `<family>-<variant>-<tag>-<input size>-e<epochs>-s<seed>`: variant `base` is the unmodified YOLOv10, `graftd` with transfer `role_ws` is P2-Graft (deeper block, warm start).
Runs of the 35-epoch ablation and of the 640 px comparisons come from earlier sessions and are listed with their source in `results/final_runs.csv`.

## Use the weights

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

path = hf_hub_download(
    "mudassiruddin/yolo", repo_type="dataset",
    filename="sessionB-20261106-1639/runs/v10-graftd-ws-640-e150-s0/weights/last.pt")
model = YOLO(path)                       # P2-Graft, YOLOv10n, 640 px, 150 epochs
model.predict("drone_image.jpg", imgsz=640, conf=0.25)
```

The checkpoints store the four-level architecture, so Ultralytics rebuilds it from the file.

## Reproduce

Code, notebook and the scripts behind every table and figure: **[github.com/Mudassiruddin7/YOLO](https://github.com/Mudassiruddin7/YOLO)**.
All runs used one NVIDIA RTX PRO 6000 Blackwell GPU, PyTorch 2.11 and Ultralytics 8.4.150.

## Licence and data

The weights derive from Ultralytics YOLOv10 checkpoints and keep the AGPL-3.0 licence. VisDrone images and annotations are not included; they belong to the VisDrone project.
