# P2-Graft v2 — confirmatory runs (YOLOv10, VisDrone-DET2019)

One marimo notebook, `graft_p2_v2.py` (≈ 2,700 lines, 138 KB; v1 was 5,300 lines and also generated a paper — it is kept in `legacy/`).

## Run it on molab (RTX PRO 6000, 12 h)
1. molab → new notebook → paste `https://github.com/Mudassiruddin7/YOLO/blob/main/graft_p2_v2.py` → notebook-specs button → **toggle the GPU**.
2. Choose the session, **paste your Hugging Face token** (write access to the dataset in the repo field; a password field, kept in memory only) and press **START**. The notebook checks the token before training and then pushes the results (leaderboard, every finished run's result.json, curves, final weights, saved predictions) to the dataset **every 20 minutes and after every finished run**, so a closed laptop or a molab shutdown costs at most the last 20 minutes. Revoke the token afterwards.
3. Without a token the results live only on molab's disk (lost 90 min after the browser disconnects or at 12 h): download `diagnostics.zip` from the last cell before that.

Three sessions, each self-contained (≈ 8 h):

| Session | Runs | Answers |
|---|---|---|
| **A** seeds | model under test (M) vs baseline (B), 896 px, 90 epochs, **3 seeds each** | C1 |
| **B** best resolution | M vs B at **1280 px** (90 epochs); M vs B at 640 px with a **150-epoch** schedule | C4, C2 |
| **C** second scale | M vs B on **YOLOv10s**, 640 px, 90 epochs, **3 seeds each** | C3 |

M = *P2-Graft deep, warm-started*: a stride-4 prediction level with a 2× deeper neck block, grafted onto the pretrained detector (the original levels are reproduced bit-exactly at
initialisation) and initialised from the pretrained P3 level. B = the unmodified YOLOv10 baseline, trained with the identical recipe.

## What was found before v2 (VisDrone, YOLOv10n, rounds 1–5; see `../results/FINDINGS.md`)
* At the **same input size** M beats B: 640 px / 90 epochs: **+2.9 val, +1.7 test-dev AP** (3 seeds, every seed).
* At **equal GFLOPs** a plain baseline at a higher resolution beats M by ≈ 3 AP (B@896 vs M@640; B@1280 vs M@896). Best model overall: B at 1536 px (30.3 val / 23.9 test-dev AP).
* Warm starting the new level is what matters (a deep block without it = plain P2-Graft). Cheap "detail injection" helps at 35 epochs but not at 90.

## Pre-registered hypotheses of v2 (fixed before the runs; nothing is selected on the way)
* **C1** M > B in test-dev AP at 896 px, 90 epochs. *Supported* if every paired seed has a positive difference and the 95 % image-bootstrap interval of the seed-averaged difference excludes 0.
* **C2** M > B also with a 150-epoch schedule (640 px). Single seed → reported with the bootstrap interval only, never as "robust".
* **C3** M > B on a second model scale (YOLOv10s). Same rule as C1.
* **C4** M vs B at 1280 px: the sign and size of the gain at the best resolution (single seed, interval only).
No hypothesis about equal-compute superiority is made: it is already known that resolution wins there; the leaderboard reports the distance to that frontier (`d frontier`) for every model.

## What the notebook reports (and why)
* **Primary metric: AP on VisDrone test-dev**, never used for any decision (the model M was chosen on validation in an earlier search). Validation AP is reported next to it.
* `results/leaderboard.csv|md`: per condition mean ± std over seeds, the test-dev gain over the baseline *paired by seed* (mean, spread, number of pairs, whether every pair has the same sign),
  the distance to the baseline's resolution frontier, GFLOPs, parameters, **batch-16 throughput** (compute-sensitive) and batch-1 latency (median of 3 blocks; at batch 1 these models are
  launch-bound and measure 3.2–3.7 ms regardless of size — do not read speed claims into it).
* `results/contrasts.csv`: image-level paired bootstrap 95 % intervals (test-dev) of every model-minus-baseline difference with equal family, size and schedule.
* Evaluation: saved predictions, confidence 0.001, 500 detections/image, COCO AP on original-resolution boxes, one-to-one head; AP/AP50/APs; TIDE on validation; every run also tested at
  larger sizes (640 → 800/960, 896 → 1024) without retraining. Saved predictions can be included in the diagnostics zip.
* Batch policy: nominal batch 64 via accumulation; four-level models use batch 16 from 896 px and batch 8 from 1280 px because of memory. Compare runs only within one session.

## Removed from v1 to cut overhead
The LaTeX paper generator, figures, the old round plans/search, static analyses, zero-shot external evaluation, the one-to-many/NMS, ignore-region and Ultralytics-validator evaluations
(several minutes per run) — none is needed for the claims. Results of v1 sessions are comparable (same seeds give identical numbers) and are summarised in `../results/`.

## Limitations that no notebook change removes
One dataset (VisDrone); no cross-dataset evaluation (the earlier UAVDT claim is unsupported and not made); schedules are shorter than the 300-epoch recipes in the literature, so absolute numbers
must not be compared with published tables; the validation→test-dev gain shrinks (+2.9 → +1.7) and test-dev should be quoted.

## Other files
`AUDIT.md` + `audit/` — what was wrong in the earlier LAF-YOLOv10 code and paper (everything re-run with numbers). `legacy/` — the v1 notebook, README and layout previews.
