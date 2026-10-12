# P2-Graft — one marimo notebook for a controlled small-object study (YOLOv10n × VisDrone)

> **START HERE (molab, RTX PRO 6000, 12 h)** — the defaults now run **ROUND 2** (extensions H5–H7, ≈ 9.6 h in total)
> 1. molab → new notebook → paste `https://github.com/Mudassiruddin7/YOLO/blob/main/graft_p2_study.py` → notebook-specs button → **toggle the GPU**.
> 2. Press **Run all** and leave the tab open and the computer awake. Nothing else has to be clicked: environment, data, analyses, smoke test, calibration, training and the paper build run one after the other.
> 3. When ⑧ says the worker has finished and the paper is written, **download ⑩ (`paper_bundle.zip`) and ⑩b (`diagnostics.zip`) before the session ends** — molab deletes everything else.
>
> **Round 1** (main study, H1–H4: tiers 0+1+2, budget 8.5 h) is finished; its results are summarised below. To repeat it, set *Run tiers* to 0, 1, 2 and the budget to 8.5 in cell ① before pressing Run all.
>
> Details, molab facts and troubleshooting are below.

### Round 1 — results (molab, RTX PRO 6000, 23 runs, 35 epochs, VisDrone-val, mean over 3 seeds unless stated)

| model | AP | AP50 | AP_S | test-dev AP | GFLOPs | deploy params |
|---|---|---|---|---|---|---|
| YOLOv10n baseline | 16.22 ± 0.04 | 29.3 | 8.0 | 13.5 | 6.6 | 2.27 M |
| **P2-Graft** | **17.15 ± 0.19** | 30.3 | 9.5 | 13.7 | 12.7 | 2.39 M |
| naive P2 re-wiring | 15.89 | 28.4 | 8.8 | – | 10.7 | – |
| P2 replaces P5 | 16.92 | 30.2 | 9.3 | 13.6 | 12.2 | 1.88 M |
| baseline @ 896 px (≈ P2-Graft GFLOPs) | **21.65 ± 0.17** | 37.4 | 12.8 | 17.4 | 13.0 | 2.27 M |

H1 supported (+0.93 AP, 95 % bootstrap interval [0.72, 1.17], every seed), H2 supported (graft +1.26 over re-wiring), H3 inconclusive (interaction +0.09 [−0.19, 0.37]),
**H4 contradicted** (a higher-resolution baseline at equal GFLOPs is 4.5 AP *better* than P2-Graft). The earlier note that the paper's 23.99 AP came from confidence-truncated detections is
reproduced here too: truncating at conf ≥ 0.25 turns AP50 29.3 → 20.0. Latency of round 1 was measured while three trainings shared the GPU and is therefore not interpretable (the notebook now re-measures it
one model at a time); parameters and GFLOPs are exact. Absolute numbers are for a 35-epoch schedule and must not be compared with 300-epoch results from the literature.

### Round 2 — what it adds (hypotheses added after seeing round 1, marked as such in the generated paper)

| | question | runs |
|---|---|---|
| H5 | For the *same* architecture, is the role-based weight map better than key matching? (round 1 confounded architecture and transfer) | P2-Graft with key-matched transfer × 3 seeds |
| H6 | Does the P2 level still help at the higher, compute-matched resolution (896 px)? | base@896 and graft@896, 3 seeds (the third seed is the first to be dropped if time runs short) |
| H7 | Does the benefit of the level grow with a 2× longer schedule (70 epochs)? | base and graft, 70 epochs, 2 seeds |

Round 2 also re-runs the headline pair (tier 0) in the new session: it is the paired reference for H5–H7 and a replication of H1.

`graft_p2_study.py` is a single [marimo](https://marimo.io) notebook (plain `.py`, dependencies declared in its PEP 723 header).
It contains the **entire study**: data download/conversion, the model variants, a background training scheduler, evaluation from
saved predictions, statistics, figures, and a generator that writes the LaTeX paper from the results.

| file | what it is |
|---|---|
| `graft_p2_study.py` | the notebook — open this on molab |
| `run_headless.py` | terminal-only fallback that runs the same pipeline (data → static analyses → calibration → worker → paper) without the UI |
| `AUDIT.md` | what was wrong in the earlier LAF-YOLOv10 code/paper (everything re-run, with numbers) and how this study avoids it |
| `audit/` | the scripts and raw JSON outputs behind `AUDIT.md` |
| `preview/` | **layout preview only**: the paper generated from *synthetic* numbers, once per template (LNCS, 14 pages; two-column article, 9 pages), with a red banner and red header on every page: "NOT RESULTS". It shows what step ⑨ will produce; no figure or number in it is a measurement |

## What the study asks (fixed before any result)

| | question |
|---|---|
| H1 | Does a stride-4 (P2) level improve a COCO-pretrained YOLOv10n on VisDrone when the library-default label assignment is used? |
| H2 | Does the way pretrained weights are carried over matter — **graft** (original P3–P5 path kept bit-exact) vs. textbook **re-wiring** with key-matched transfer? |
| H3 | Is the benefit of the P2 level smaller when the assigner already enlarges the candidate window of tiny boxes (Ultralytics' "window floor")? |
| H4 | At matched GFLOPs, is a P2 level better than a higher input resolution? |
| H5–H7 | added for round 2, see above |

Design (priority order, so a partial run is still a coherent study): **tier 0** baseline vs. graft with the library-default assignment, 3 seeds each;
**tier 1** one seed of the compute-matched baseline (base@896 px), naive re-wiring, classic-TAL baseline and graft, P2→P5 replacement and the two random-init controls;
**tier 2** more seeds (first the classic-TAL cells, completing the 2×2 factorial, then naive/base@896/replace); **tier 3** a YOLO26 check; **tier 4** (round 2) the 70-epoch pair, key-matched graft and the 896 px pair.
The planner fits tiers 0–1 into the budget; whatever time remains goes to tier 2.
Evaluation: saved predictions, COCO AP on **original-resolution** boxes with 500 detections/img, TIDE with correct `xywh` boxes, paired image-level bootstrap,
Ultralytics-validator cross-check, VisDrone ignore-region variant, protocol-sensitivity table (confidence truncation, detection cap).
The paper text is generated with verdicts computed from the intervals — a negative or inconclusive result is reported as such.

## Running it on molab (RTX PRO 6000, 12 h) — step by step

What molab does ([docs](https://docs.marimo.io/guides/molab/), [announcement](https://marimo.io/blog/reintroducing-molab), [storage](https://marimo.io/blog/seamless-storage-in-molab)):
the GPU is an opt-in toggle behind the *notebook specs* button in the notebook header; a notebook gets 4 CPUs and 32 GB RAM by default; it runs for at most 12 hours and is shut down after
90 minutes without activity; and **only files uploaded through molab's file browser survive a shutdown** — everything this notebook writes (data, runs, results) is lost when the session ends.
So: keep the tab open and your computer awake while it trains, leave the ⑧ auto-refresh on, and download ⑩ / ⑩b before the end.

1. **Get the notebook into molab.** molab opens notebooks from GitHub. On github.com create a new repository (public is the safe choice), choose *Add file → Upload files*, drag in `graft_p2_study.py` (only this file, from this
   folder) and commit. Then on the molab home page use the new-notebook dropdown and paste the file's URL (`https://github.com/<you>/<repo>/blob/main/graft_p2_study.py`); replacing `github.com` by `molab.marimo.io/github` in
   that URL opens a preview from which you can fork it into your workspace (I could not verify the exact buttons). molab's docs describe no direct file upload for notebooks and do not say whether private repositories work.
   Open the notebook, click the notebook-specs button in the header and **toggle the GPU**.
2. **Run ① (settings; the defaults are meant to be kept) and ② (environment).** ② probes the packages and the GPU in a subprocess. If it offers *Fix automatically*, press it (it installs what is missing — Ultralytics pinned to
   8.4.150, pycocotools, faster-coco-eval, tidecv, the *headless* OpenCV — and never replaces the preinstalled PyTorch), then *Re-check*. Errors that remain (no GPU attached, a PyTorch build without `sm_120`) are explained in place.
3. **Press ⓪ Autopilot** (tick the box first). It runs: environment → data (≈ 1.9 GB download, 10–25 min) → exactness/transfer/assignability analyses → smoke test (a few minutes) → calibration (5–10 min) → plan → launch the detached
   worker. It stops at the first failure and says why; pressing it again resumes (finished parts are skipped). The numbered steps ③–⑦ do the same one at a time if you prefer.
4. **Watch ⑧** (auto-refresh on). Budget: the slider (default 8.5 h) is the time allowed for *training + evaluation* counted from the launch; the 12 h session also has to fit ≈ 1 h of set-up and the final paper build (≈ 0.7 h). The planner measures
   s/epoch on your machine and picks the largest epoch count (multiple of 5, 20–100) for which tiers 0–1 fit; runs that would overrun are skipped from the end of the queue.
5. **When the worker has finished** it builds the paper by itself (10–20 min: image-level bootstrap, TIDE tables, figures). Then press **⑩ Download** (paper bundle) and **⑩b** (diagnostics zip) and keep both.
   Send me `paper_bundle.zip` and `diagnostics.zip`; if the run fails or stalls, send `diagnostics.zip` (and the ② table) right away — it contains every status, log and per-run result.
   ⑧ also offers the results so far as a small CSV — download it every few hours as a backup.

If the session dies (idle shutdown, 12 h limit) nothing can be recovered from molab's disk: that is why the downloads matter. A re-launch in a *new* session starts from zero.

Memory/CPU: the data pipeline (mosaic, HSV, warp) is the bottleneck for a nano model, not the GPU. The notebook reads the *granted* CPU/RAM (cgroup limits),
so a container that shows 64 host cores but grants 4 is handled. With ≥ 12 CPUs the scheduler runs 2 trainings side by side (≥ 18: 3), never more than one per 24 GB of GPU memory:
the 4-level (P2) model needs roughly 20 GB at batch 32 on dense VisDrone batches (an estimate; the measured part is its task-aligned assigner: ≈ 12 GB when a batch contains an image with ~900 boxes, vs. ≈ 5 GB for the baseline).
Images are pre-resized once, and cached in RAM when enough memory is free. Seconds per epoch on the RTX PRO 6000 could not be measured by me — step ⑥ measures it
and the plan (epochs per run, which runs fit) is derived from that measurement, not from a guess.

Local run: `uvx marimo edit --sandbox graft_p2_study.py` (set `GRAFT_WORKDIR` to choose where data and runs go).

## Troubleshooting

| symptom | likely cause / fix |
|---|---|
| ② says "no GPU is visible (nvidia-smi finds none)" | the GPU toggle (notebook-specs button in the molab header) is off — switch it on and restart the kernel |
| ② says "the driver sees a GPU but PyTorch does not", or "compute capability 12.x … lists [sm_90 …]" | the installed PyTorch build does not fit this GPU/driver: press *Install PyTorch for CUDA 12.8* (uses `uv pip … --torch-backend cu128`, ≈ 3 GB), restart the kernel, *Re-check* |
| ② reports OpenCV / `libGL.so.1` problems or missing packages | press *Fix automatically* (headless OpenCV; Ultralytics pinned without letting it replace PyTorch), then *Re-check* |
| ② warns that data-loader workers fail (`/dev/shm`) | the container's shared memory is tiny; the study falls back to 0 workers (much slower — fewer epochs fit). Nothing to do |
| the autopilot stops with a failed step | read that step's message; it is re-runnable (finished parts are skipped). Send `diagnostics.zip` (⑩b) if you cannot tell |
| data step stalls or fails | the Ultralytics GitHub mirror rate-limits; the downloader resumes (`Range` requests) — just press the button again |
| a run dies with CUDA OOM | four layers of protection (each hit by a real CUDA OOM on a 4 GB GPU): Ultralytics halves the batch itself (96→48→24…), its assigner falls back to per-image assignment, `run_one` halves the batch and retries, and the worker re-queues a run that still died with OOM *once, in a fresh process* with half the batch (`nbs` stays 64; the paper lists such runs). Evaluation is chunked by this notebook because Ultralytics' predictor runs a whole list of images as one forward pass. If it persists: fewer parallel runs (`n_parallel` = 1) |
| the monitor shows "skipped (would exceed time budget)" | expected near the end of the budget: those runs were not started so that nothing is cut off mid-training |
| you closed molab and lost the session | the work directory persists only if molab keeps it; re-launch skips finished runs and resumes partial ones. Download ⑩ regularly |
| one run failed | see `<work>/logs/<run_id>.log`; delete `<work>/runs/<run_id>` and re-launch to retry it |

## Before you submit anything

* **Venue.** "ICPR 2027" is **ICPR (Virtual) 2027 (ICPRv 2027)**, the first virtual edition of the IAPR conference (the face-to-face ICPRs are 2026 in Lyon — already past — and 2028 in Sydney).
  Per the call for papers (checked on 2026-10-11; dates are marked *tentative* by the organisers): conference 4–6 and 14–15 October 2027; **submission deadline 1 March 2027**,
  reviews 9 April, optional rebuttal 16 April, notification 24 May, camera-ready 7 June; Springer **LNCS** format; **single-blind** review (author names and affiliations *must* be in the manuscript);
  *long* papers 15 pages (published by Springer in LNCS), *short* papers 8 pages for preliminary results (**not** in the proceedings); contact `icpr27pc@iapr.org`.
  Topics include object detection, efficient and scalable AI, datasets and evaluation, transfer learning and remote sensing, which fits this study.
  Sources: <https://www.icpr2027.com/> (IAPR's page redirects there) and the call-for-papers PDF linked from it. Re-read the final author instructions before submitting
  (page limits, whether references/appendices count, template).
* **Template.** Step ① lets you choose `lncs` (default: `\documentclass[runningheads]{llncs}` + `splncs04`) or `article` (self-contained two-column layout).
  Springer's `llncs.cls` and `splncs04.bst` are *not* included (they are Springer's files): copy them from Springer's LNCS author kit or from Overleaf's "Springer LNCS" template next to `main.tex`,
  then `pdflatex main && bibtex main && pdflatex main && pdflatex main`. The LNCS preview in `preview/` (synthetic numbers) compiles to 14 pages *including references and the all-runs appendix*;
  the real paper will differ by a page or so (more/fewer conditions, the optional zero-shot paragraph, longer verdict texts). If it overflows: move the all-runs table (appendix) into the code archive (`all_runs.csv`), shorten Related Work, and drop the per-class figure.
* **Authors, affiliations, code URL, acknowledgements** are placeholders in the generated paper (the LNCS template needs real names for single-blind review).
* **Verify every reference** in `references.bib` against the publisher record (several entries were reused from your earlier bibliography, others were added from memory).
* **Licences.** Ultralytics is AGPL-3.0 (releasing code that imports/modifies it has implications); VisDrone is for research use.
* Read the generated text once: claims are conditional on the data, but the introduction/related-work prose is static.
* The earlier arXiv paper's headline numbers should not be reused (see `AUDIT.md`).

## What has and has not been verified here

Verified here (Windows laptop: 16 CPU threads, 25 GB RAM, **RTX 3050 A Laptop GPU with 4 GB** and CUDA PyTorch 2.13/cu130, Ultralytics 8.4.150):
the data pipeline on the real VisDrone splits; the exactness of the graft against the real COCO checkpoint (YOLOv10n and YOLO26n: bit-identical outputs on the original levels);
transfer accounting; training and evaluation of all four architecture variants on CPU (tiny runs and a 1,000-image run); on the GPU, AMP training of the baseline and the graft, the complete
post-training evaluation (predictions, COCO AP, TIDE, Ultralytics validator, fp16 latency, FLOPs) and recovery from real CUDA out-of-memory errors (this found and fixed three bugs: memory left
allocated after training, a whole-list predictor pass that needs GBs per hundred images, and an unprotected secondary evaluation); the scheduler/worker including the OOM re-queue (unit test with a fake child);
COCO-eval vs. pycocotools vs. Ultralytics-validator agreement; TIDE; the bootstrap accumulation; the whole paper generator on synthetic numbers and on real predictions of toy models, in both templates
(compiled with Tectonic); the optional zero-shot evaluator on a stand-in dataset (VisDrone-val re-read as an "external" folder: identical AP50, per-class AP50 and class mapping).

For the molab hand-over specifically: the **autopilot in test mode** (48 images, 1 epoch) end to end on the laptop GPU *and* in two **bare virtual environments** that contain only marimo, numpy, pandas and CPU PyTorch (Python 3.14 and 3.12) —
there the environment repair installed everything else from scratch (pinned Ultralytics without letting it touch PyTorch, torchvision matched to the installed torch, tidecv, pycocotools, faster-coco-eval, the headless OpenCV);
the notebook's button cells executed with the buttons "pressed" (autopilot, repair report, diagnostics zip, downloads); a killed run **resuming** from its last checkpoint; the full calibration step through the notebook's subprocess path;
the notebook compiles under Python 3.10 – 3.12. Those tests found and fixed: a **race between training processes that start together** (they all prepare the same data files — now file-locked, with unique temp files and staggered starts), an
`ensure_environment` / `need_cuda` mix-up, and the GPU bugs listed above.

**Not** verified: the RTX PRO 6000 itself (Blackwell build of PyTorch, 96 GB); anything that is Linux-specific (cgroup limits, the detached worker via `start_new_session`: code read, not executed);
molab's own behaviour (how notebooks are opened, which packages are preinstalled, what "idle" means for its 90-minute shutdown); throughput and the 12-hour plan (the calibration measures them on your machine);
the zero-shot check on a real second dataset. The laptop's timings (e.g. 20–120 s per tiny epoch, memory spilling into shared RAM) say nothing about the target GPU. No experimental result of the study exists yet — it is produced by your run.
