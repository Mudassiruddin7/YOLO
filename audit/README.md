# Audit scripts

These re-run the **earlier LAF-YOLOv10 notebook's own code** (extracted from `LAF_YOLOv10.ipynb`, cells that only *define* the model, dataset and evaluator)
on the released checkpoints. They were executed on CPU against the official VisDrone splits.

| script | what it does | output |
|---|---|---|
| `convert_legacy_layout.py <zip> <out_root> <split>` | unzips a VisDrone split into the layout the old notebook expects and prints an annotation census (categories, ignore flags, size bins) | `<out_root>/census_<split>.json` |
| `legacy_reeval.py <ipynb> <ckpt_dir> <root> <out.json>` | scores the three released checkpoints on VisDrone-val with (a) the old protocol (confidence > 0.25 enters mAP) and (b) a standard protocol (confidence > 0.001, ≤ 300 detections/img), with torchmetrics and pycocotools | `legacy_reeval.json` |
| `legacy_tide_check.py ...` | feeds the same detections to tidecv with the legacy (`xyxy`) call and with correct `[x, y, w, h]` boxes, at two confidence thresholds | `legacy_tide.json` |
| `legacy_splits_check.py ...` | re-creates the "locked" 647-image hold-out exactly as `make_holdout_split` does (`random.Random(777).sample`) and scores it and VisDrone test-dev (seed 42) | `legacy_splits.json` |

Run them with `python -I script.py <args>` from a directory that contains nothing else of yours (the notebook code is executed, so keep the working
directory clean). Requirements: torch, torchvision, torchmetrics, pycocotools, tidecv, opencv-python, pillow, numpy.
