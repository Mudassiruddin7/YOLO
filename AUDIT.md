# Audit of the earlier LAF-YOLOv10 code and paper

Scope: `LAF-YOLOv10-codebase.zip` / `Small-Object-Detection-in-UAV-Imagery-main.zip` (identical notebooks, three released
checkpoints) and `LAF_YOLOv10_paper_arxiv (10).pdf` (the 24.0 % paper). Everything below was **re-run on this machine**
(scripts in `audit/`, raw outputs in `audit/*.json`), not inferred by reading alone. VisDrone-val = 548 images.

## Headline result: the "7.8-point deficit" is mostly an evaluation artefact

| quantity (seed 42 / 123 / 256) | value |
|---|---|
| paper protocol re-run (detections with confidence > 0.25 enter mAP) | AP50 = 23.80 / 23.76 / 24.42 → **23.99 ± 0.37** (paper: 24.0 ± 0.4) |
| same detections scored with **pycocotools** instead of torchmetrics | 23.80 / 23.76 / 24.42 (identical – the metric code is *not* the problem) |
| **same checkpoints, standard protocol** (confidence > 0.001, ≤ 300 dets/img, pycocotools) | AP50 = 29.65 / 29.73 / 30.05 → **29.81 ± 0.21** |
| AP (0.50:0.95), torchmetrics, paper protocol → standard protocol | 12.1–12.6 → 14.1–14.5 |

`evaluate()` (notebook cell 51) calls `postprocess_nms(..., conf_thres=0.25)` *before* `DetMetrics.update`, so mAP is computed on a
detection list truncated at the single operating point used for the P/R columns (R = 46.6–47.3 %, 48 detections per image for 71
ground-truth boxes per image). Average precision is an integral over the whole ranking; cutting it at 0.25 removes the
low-confidence tail that carries ~6 AP50 points on a dense, small-object dataset. Published detector numbers (31.8 for YOLOv10n etc.)
use confidence 0.001, so the −7.8 comparison mixes protocols; the like-for-like gap of this checkpoint to that literature number is
≈ −2 points, and even that is not a controlled comparison (see B2).

## A. Evaluation and reporting

| id | issue | evidence | resolution in the new study |
|---|---|---|---|
| A1 | mAP on confidence-truncated detections (above) | `audit/legacy_reeval.json` | all metrics at conf 0.001, 500 dets/img, from saved predictions |
| A2 | torchmetrics caps at 100 detections/img; 19.5 % of val images have > 100 GT (max 317; train max 902) | `audit/legacy_reeval.json`, `meta` | `maxDets = 500`; Ultralytics `max_det=500` |
| A3 | **TIDE was given `xyxy` boxes; tidecv expects `[x, y, w, h]`** (cell 79 `run_tide`) | `audit/legacy_tide.json`: legacy call AP50 31.18 (the "bbox AP @ 50: 31.17" in the notebook output) vs 23.80 with correct boxes at the same threshold (= the mAP evaluator) | TIDE computed from saved predictions with correct boxes; unit self-test |
| A3b | Table 6 consequently describes an artefact: legacy call → Cls 11.3, Loc 2.6, Both 0.08, Dupe 0.05, Bkg 0.03, **Miss 27.9**; correct boxes, same threshold → 16.2 / 2.2 / 0.34 / 0.09 / 0.70 / 22.4; correct boxes, conf 0.001 → 26.4 / 4.7 / 0.94 / 0.33 / 2.04 / **6.3** | `audit/legacy_tide.json` | the "fewer background errors, many more misses" story does not survive either fix |
| A4 | Baseline rows of Tables 1, 5, 6 are hard-coded constants (`PAPER_TABLE1/5/6`) that sit next to an older LAF row (35.1 % / 20.7 / 2.3 M, i.e. the notebook was written for an earlier draft); the YOLOv10n rows are the same constants as in the current paper, so no baseline was ever trained or evaluated here | notebook cells 59, 77, 79 | the baseline is trained in the same pipeline, same seeds, same evaluator |
| A5 | SSCW row differs between paper (37.8/22.3, 2.61 M) and notebook (34.3/19.4, 2.8 M) | cell 59 vs paper Table 1 | no quoted numbers in the new paper |
| A6 | "Zero-shot UAVDT" (Table 2) is not produced by the notebook: the UAVDT cell **trains a new model on UAVDT** and was never run (output: "UAVDT not found"); class order differs from the text (car,truck,bus vs car,bus,truck); no VisDrone-trained YOLOv10n checkpoint exists | cells 61–63 | not claimed; any UAVDT result must come from a saved-prediction run with an explicit class map |
| A7 | Tables 3–4, attention/loss/channel-selection comparisons: the notebook prints "SMOKE_TEST is False: skipping…"; no logs, checkpoints or outputs for them. `run_ablation_suite` trains with `pretrained=True` (default) while `run_ablation_suite_multiseed` defaults to `False` | cells 70–75 | every reported run is a finished, logged run with `result.json` |
| A8 | Held-out (23.5/11.9) and test-dev (22.5/11.4) numbers (Table 7) have no output in the notebook, and do not reproduce | cell 65 never executed; "Table 7 re-check" below | held-out/test-dev are scored from saved predictions by the same evaluator |
| A9 | Dataset statistics: paper says 54 % < 32², 22 % < 16², ~5 % < 8², < 0.8 % > 256² (343,205 objects). Re-counted on the official train annotations (area in original pixels): **60.5 %, 26.0 %, 5.3 %, 0.16 %** (343,204 objects with positive size, 6,471 images; one degenerate zero-size box explains the ±1) | `convert_legacy_layout.py` census and the new notebook's `dataset_census` | statistics are computed by the notebook and inserted into the text |
| A10 | Compute provenance: output shows an **NVIDIA A100 MIG 2g.10gb (9.75 GiB)**; the paper says a single 24 GB GPU, code comments say RTX 4090 and 6.2 GPU-h/run; the only timing in the logs is 20 epochs in 4,135 s (≈ 207 s/epoch ≈ 17 h per 300 epochs). "4.2 h for YOLOv10n" and "25 h for the ablation suite" have no source (neither was trained) | cell 3 output, cell 55 output | `env_info()` is stored in every `result.json`; wall-clock is measured |
| A11 | `README.md` says the charts are "regenerable from the numbers in the paper", i.e. they are not data-driven | README | figures are produced from `all_runs.csv` |

## B. Method / implementation

| id | issue | evidence | resolution |
|---|---|---|---|
| B1 | **Pretrained "transplant" is positional**: `load_pretrained_backbone` zips Conv/BN tensors of the two models in module-registration order and copies when shapes agree. The re-implementation's widths are (32,64,128,256) while YOLOv10n's stem is 16 channels, so the very first tensor mismatches ("ours=(32,3,3,3) vs yolov10n's=(16,3,3,3)") and the 73/150 "matches" after that are shape coincidences, not role-aligned layers | code: cell 49; log: cell 55 output | role-based key map + transfer accounting + exactness test (graft is bit-identical on the original levels) |
| B2 | The "baseline" in the ablations (1.98 M params, no PSA, no dual head) and the headline model (2.14 M) are re-implementations; neither is YOLOv10n (2.78 M training graph, 2.27 M fused) | cell 36/37 | the real Ultralytics YOLOv10n is used for every compared model |
| B3 | `WiseIoUv3Loss` computes `alpha* = (beta/delta)^gamma`, which **up**-weights high-loss boxes; docstrings and README say it "damps" outliers; the class name implies WIoU-v3's non-monotonic focusing, which it is not (the paper text does admit the difference) | cell 32 | no WIoU in the new study |
| B4 | The custom task-aligned assigner has no small-target handling (candidates strictly inside the box). Exact count on train labels @640 px: **16.6 % of boxes have no candidate anchor on P3–P5** (47.8 % of boxes with √(wh) < 8 px) | `analysis/assignability` of the new notebook | analysed and turned into a controlled factor (window floor) |

## C. Things that were checked and are fine

* The mAP implementation (torchmetrics) agrees with pycocotools on identical detections (23.80 vs 23.80).
* VisDrone → YOLO conversion matches Ultralytics' (`score == 0` rows dropped; no class index ≥ 10).
* The internal hold-out is reproducible (`random.Random(777).sample`) and the training loader is built only from non-hold-out indices.
* Reported mean ± std of the three seeds (23.99 ± 0.37 with ddof = 1) is consistent with the checkpoints.

## Table 7 re-check (`audit/legacy_splits.json`, seed 42)

| split | paper (Table 7) | re-run, paper protocol (conf > 0.25) | re-run, standard protocol (conf > 0.001) |
|---|---|---|---|
| internal hold-out (647 images, re-created with `random.Random(777)`) | 23.5 / 11.9 | 25.0 / 13.1 | 31.4 / 15.3 |
| VisDrone test-dev (1,610 images) | 22.5 / 11.4 | **17.9 / 8.9** | **22.1 / 10.4** |
| VisDrone val (548 images), for reference | 24.0 / 12.3 | 23.8 / 12.1 | 29.6 / 14.1 |

Neither number reproduces under the paper's own protocol, and the paper's sentence "held-out and test-dev scores remain close to the validation score" does **not**
hold for test-dev under either protocol (test-dev is 5.9 points below val with the paper's protocol, 7.5 below with the standard one — it is simply a harder split).
The hold-out lock file was created in an earlier session (the notebook output says "Loaded existing holdout lock") and is not part of the release, so the exact
hold-out of the paper cannot be verified; the code path that builds the training loader from the non-hold-out indices is correct.
