# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.14",
#     "ultralytics==8.4.150",
#     "numpy",
#     "pandas",
#     "scipy",
#     "pyyaml",
#     "requests",
#     "matplotlib",
#     "pillow",
#     "pycocotools",
#     "faster-coco-eval",
#     "tidecv",
#     "psutil",
#     "tqdm",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="P2-Graft: VisDrone small-object study")

with app.setup:
    # Light-weight imports only: torch / ultralytics are imported lazily inside functions so that the
    # notebook UI starts instantly and a background worker can import this file as a plain module.
    import copy
    import csv
    import dataclasses
    import gzip
    import hashlib
    import io
    import json
    import math
    import os
    import platform
    import random
    import re
    import shutil
    import subprocess
    import sys
    import threading
    import time
    import traceback
    import zipfile
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import dataclass, field, asdict
    from pathlib import Path
    from typing import Any, Dict, List, Optional, Sequence, Tuple

    import numpy as np
    import pandas as pd

    STUDY_VERSION = "1.0.0"
    ULTRALYTICS_PIN = "8.4.150"
    try:  # path of this notebook file (needed to start the detached worker); marimo provides __file__
        NOTEBOOK_FILE = Path(__file__).resolve()
    except NameError:  # pragma: no cover
        NOTEBOOK_FILE = None

    CLASS_NAMES = [
        "pedestrian", "people", "bicycle", "car", "van",
        "truck", "tricycle", "awning-tricycle", "bus", "motor",
    ]
    VISDRONE_URLS = {
        "train": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
        "val": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
        "test-dev": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip",
    }
    # approximate sizes (bytes) used only as a sanity floor when verifying a finished download
    VISDRONE_MIN_BYTES = {"train": 1_300_000_000, "val": 80_000_000, "test-dev": 300_000_000}
    # COCO-style area bins (pixels^2, ORIGINAL image resolution) plus two finer VisDrone bins
    AREA_BINS = [
        ("all", 0.0, 1e10), ("small", 0.0, 32.0 ** 2), ("medium", 32.0 ** 2, 96.0 ** 2), ("large", 96.0 ** 2, 1e10),
        ("tiny", 0.0, 16.0 ** 2), ("xsmall", 16.0 ** 2, 32.0 ** 2),
    ]
    # LaTeX template of the generated paper: "lncs" (Springer LNCS, the format of ICPR/ICPRv) or "article" (two-column article).
    # A mutable dict so that the table/figure helpers can see the template chosen by render_paper_tex().
    PAPER_STYLE = {"template": "lncs"}


# =====================================================================================================
# SECTION A -- small utilities
# =====================================================================================================
@app.function
def jdump(obj, path):
    """Atomic JSON write (a temp name that is unique per process and thread, so concurrent writers never share a temp file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=json_default)
    os.replace(tmp, path)


@app.function
def acquire_lock(path, timeout=7200.0, poll=2.0):
    """Cross-process (and cross-platform) lock: exclusive creation of a lock file.  A lock whose holder is dead, or older than `timeout` seconds, is broken."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.time()}".encode())
            os.close(fd)
            return
        except FileExistsError:
            try:
                holder = path.read_text().split()
                if time.time() - path.stat().st_mtime > timeout or (holder and holder[0].isdigit() and not pid_alive(int(holder[0]))):
                    path.unlink()
                    continue
            except Exception:
                pass
            if time.time() - t0 > timeout:
                raise TimeoutError(f"could not acquire {path}")
            time.sleep(poll)


@app.function
def release_lock(path):
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


@app.function
def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if dataclasses.is_dataclass(o):
        return asdict(o)
    return str(o)


@app.function
def jload(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


@app.function
def log(msg, path=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if path is not None:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


@app.function
def fmt_hours(seconds):
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


@app.function
def default_work_dir():
    env = os.environ.get("GRAFT_WORKDIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / "graft_p2_work"


@app.function
def effective_cpus():
    """CPUs this process may really use: min(host CPUs, affinity mask, cgroup quota) - containers (molab, Docker) often
    report the host's core count although only a few cores are granted."""
    n = os.cpu_count() or 1
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except Exception:
        pass
    try:  # cgroup v2: "<quota> <period>" or "max <period>"
        q = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if q and q[0] != "max":
            n = min(n, max(1, int(math.ceil(int(q[0]) / int(q[1])))))
    except Exception:
        pass
    try:  # cgroup v1
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0:
            n = min(n, max(1, int(math.ceil(quota / period))))
    except Exception:
        pass
    return int(n)


@app.function
def effective_ram_gb():
    """(total, available) memory in GB honouring cgroup limits."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        total, avail = float(vm.total), float(vm.available)
    except Exception:
        return float("nan"), float("nan")
    for lim_f, use_f in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                         ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            v = Path(lim_f).read_text().strip()
            if v != "max" and int(v) < total:
                lim = float(int(v))
                used = float(int(Path(use_f).read_text().strip()))
                total, avail = lim, min(avail, max(0.0, lim - used))
        except Exception:
            pass
    return total / 1e9, avail / 1e9


@app.function
def env_info():
    """Hardware / software fingerprint that is stored next to every result (provenance)."""
    ram_total, ram_avail = effective_ram_gb()
    info = {
        "study_version": STUDY_VERSION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": effective_cpus(),
        "cpu_count_host": os.cpu_count(),
        "ram_gb": round(ram_total, 1),
        "ram_available_gb": round(ram_avail, 1),
    }
    for mod in ("torch", "torchvision", "ultralytics", "numpy", "pycocotools", "cv2", "tidecv", "faster_coco_eval"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception as e:
            info[mod] = None
            info[mod + "_error"] = repr(e)[:200]
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["gpu_mem_gb"] = round(p.total_memory / 1e9, 1)
            info["gpu_capability"] = f"{p.major}.{p.minor}"
            info["torch_cuda"] = torch.version.cuda
            info["torch_arch_list"] = torch.cuda.get_arch_list()
    except Exception as e:  # pragma: no cover
        info["torch_error"] = repr(e)
    return info


@app.function
def download_file(url, dest, min_bytes=0, retries=12, chunk=1 << 20, log_path=None):
    """Resumable HTTP download (Range requests) with retries. Returns the destination Path."""
    import requests
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            have = dest.stat().st_size if dest.exists() else 0
            try:  # some CDNs reject HEAD on signed URLs: fall back to a plain GET without a known size
                head = requests.head(url, allow_redirects=True, timeout=60)
                total = int(head.headers.get("Content-Length", 0)) or None
            except Exception:
                total = None
            if total and have == total and have >= min_bytes:
                return dest
            if total and have > total:
                dest.unlink()
                have = 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, stream=True, headers=headers, timeout=120, allow_redirects=True) as r:
                if r.status_code == 416:  # already complete
                    return dest
                r.raise_for_status()
                mode = "ab" if (have and r.status_code == 206) else "wb"
                with open(dest, mode) as f:
                    for part in r.iter_content(chunk_size=chunk):
                        if part:
                            f.write(part)
            have = dest.stat().st_size
            if (total is None or have == total) and have >= min_bytes:
                return dest
        except Exception as e:  # network hiccup -> retry with resume
            log(f"download attempt {attempt}/{retries} for {Path(dest).name} failed: {e!r}", log_path)
            time.sleep(min(30, 2 * attempt))
    raise RuntimeError(f"could not download {url} -> {dest}")


# =====================================================================================================
# SECTION B -- data: VisDrone-DET2019 download, conversion, pre-resizing, ground-truth JSON, statistics
# =====================================================================================================
@app.function
def split_dir_name(split, size=None):
    s = split.replace("-", "")
    return f"{s}_s{size}" if size else s


@app.function
def parse_visdrone_annotation(path):
    """Return (objects[N,5] = x,y,w,h,cls0..9 , ignore[M,4] = x,y,w,h) from one VisDrone-DET annotation file.

    Rows: x,y,w,h,score,category,truncation,occlusion. score==0 marks entries that must be ignored
    (category 0 = ignored regions, category 11 = 'others'); everything else with category in 1..10 is a GT object.
    """
    objs, ign = [], []
    txt = Path(path).read_text(encoding="utf-8").strip()
    if txt:
        for row in txt.splitlines():
            p = row.strip().split(",")
            if len(p) < 6:
                continue
            x, y, w, h, score, cat = (int(float(v)) for v in p[:6])
            if score == 0 or cat == 0 or cat == 11:
                if w > 0 and h > 0:
                    ign.append([x, y, w, h])
                continue
            if 1 <= cat <= 10 and w > 0 and h > 0:
                objs.append([x, y, w, h, cat - 1])
    return np.asarray(objs, dtype=np.float64).reshape(-1, 5), np.asarray(ign, dtype=np.float64).reshape(-1, 4)


@app.function
def prepare_split(work, split, sizes=(640,), n_threads=None, limit=None, log_path=None):
    """Download+extract one VisDrone split and build, for every requested long-side size S:
         <work>/data/<split>_s<S>/images/*.jpg   (pre-resized, INTER_AREA; copied if already <= S)
         <work>/data/<split>_s<S>/labels/*.txt   (normalised YOLO labels)
       plus ground-truth COCO json files in ORIGINAL pixel coordinates (std + ignore-region variants)
       and a meta json (image ids, original sizes).  Idempotent, and safe to call from several processes at once
       (training processes that start together all call it): the work is done under a file lock, after which the others find it finished."""
    data = Path(work) / "data"
    meta_path = data / f"meta_{split_dir_name(split)}.json"

    def finished():
        if limit is not None or not meta_path.exists() or not all((data / split_dir_name(split, s) / ".done").exists() for s in sizes):
            return False
        bx = data / f"boxes_{split_dir_name(split)}.npy"
        return bx.exists() and np.load(bx, mmap_mode="r").shape[1] == 8

    if finished():
        return jload(meta_path)
    lock = data / f".prepare_{split_dir_name(split)}.lock"
    acquire_lock(lock)
    try:
        if finished():  # another process finished it while we waited for the lock
            return jload(meta_path)
        return prepare_split_locked(work, split, sizes=sizes, n_threads=n_threads, limit=limit, log_path=log_path)
    finally:
        release_lock(lock)


@app.function
def prepare_split_locked(work, split, sizes=(640,), n_threads=None, limit=None, log_path=None):
    import cv2
    from PIL import Image

    work = Path(work)
    data = work / "data"
    raw_zip = data / "raw" / f"VisDrone2019-DET-{split}.zip"
    meta_path = data / f"meta_{split_dir_name(split)}.json"
    n_threads = n_threads or max(2, effective_cpus())

    ext = data / "extracted" / split_dir_name(split)
    if not (ext / "annotations").exists():
        download_file(VISDRONE_URLS[split], raw_zip, min_bytes=VISDRONE_MIN_BYTES[split], log_path=log_path)
        log(f"extracting {raw_zip.name} ...", log_path)
        tmp = data / "extracted" / (split_dir_name(split) + "_tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(raw_zip) as z:
            z.extractall(tmp)
        inner = next(p for p in tmp.iterdir() if p.is_dir())
        shutil.rmtree(ext, ignore_errors=True)
        ext.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(inner), str(ext))
        shutil.rmtree(tmp, ignore_errors=True)

    ann_files = sorted((ext / "annotations").glob("*.txt"))
    if limit:
        ann_files = ann_files[:limit]
    images, gt_std, gt_ign, ignore_regions = [], [], [], {}
    ann_id = 1
    per_image_boxes = {}
    for i, af in enumerate(ann_files):
        stem = af.stem
        ip = ext / "images" / f"{stem}.jpg"
        with Image.open(ip) as im:
            W, H = im.size
        objs, ign = parse_visdrone_annotation(af)
        images.append({"id": i, "file_name": f"{stem}.jpg", "stem": stem, "width": W, "height": H, "n_gt": int(len(objs))})
        per_image_boxes[i] = objs
        ignore_regions[i] = ign.tolist()
        for x, y, w, h, c in objs:
            gt_std.append({"id": ann_id, "image_id": i, "category_id": int(c) + 1, "bbox": [float(x), float(y), float(w), float(h)],
                           "area": float(w * h), "iscrowd": 0})
            ann_id += 1
    cats = [{"id": k + 1, "name": n} for k, n in enumerate(CLASS_NAMES)]
    # variant that turns ignored regions into per-class crowd annotations (VisDrone-style "ignore detections inside")
    gt_ign = list(gt_std)
    aid = ann_id
    for i, regs in ignore_regions.items():
        for (x, y, w, h) in regs:
            for c in range(1, 11):
                gt_ign.append({"id": aid, "image_id": i, "category_id": c, "bbox": [x, y, w, h], "area": w * h, "iscrowd": 1})
                aid += 1
    tag = "" if limit is None else f"_lim{limit}"
    jdump({"images": images, "annotations": gt_std, "categories": cats}, data / f"gt_{split_dir_name(split)}{tag}.json")
    jdump({"images": images, "annotations": gt_ign, "categories": cats}, data / f"gt_{split_dir_name(split)}_ign{tag}.json")
    meta = {"split": split, "images": images, "n_objects": len(gt_std), "n_ignore_regions": int(sum(len(v) for v in ignore_regions.values()))}
    # compact box table for statistics (original pixels): img_id, cls, x, y, w, h, W, H
    rows = []
    for i, objs in per_image_boxes.items():
        W, H = images[i]["width"], images[i]["height"]
        for x, y, w, h, c in objs:
            rows.append((i, c, x, y, w, h, W, H))
    np.save(data / f"boxes_{split_dir_name(split)}{tag}.npy", np.asarray(rows, dtype=np.float32).reshape(-1, 8))
    if limit is None:
        jdump(meta, meta_path)

    for S in sizes:
        out = data / split_dir_name(split, S) if limit is None else data / (split_dir_name(split, S) + f"_lim{limit}")
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "labels").mkdir(parents=True, exist_ok=True)

        def one(i):
            info = images[i]
            stem, W, H = info["stem"], info["width"], info["height"]
            dst = out / "images" / f"{stem}.jpg"
            if not dst.exists():
                src = ext / "images" / f"{stem}.jpg"
                if max(W, H) > S:
                    im = cv2.imread(str(src), cv2.IMREAD_COLOR)
                    r = S / max(W, H)
                    nw, nh = max(1, int(round(W * r))), max(1, int(round(H * r)))
                    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(dst), im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                else:
                    shutil.copy2(src, dst)
            lines = []
            for x, y, w, h, c in per_image_boxes[i]:
                lines.append(f"{int(c)} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
            (out / "labels" / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            return i

        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            list(ex.map(one, range(len(images))))
        (out / ".done").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        log(f"prepared {split} @ {S}px: {len(images)} images", log_path)
    return meta


@app.function
def write_data_yaml(work, size=640, train_split="train", val_split="val", name=None, limit_suffix=""):
    work = Path(work)
    data = work / "data"
    name = name or f"visdrone_s{size}{limit_suffix}"
    y = {
        "path": str(data),
        "train": f"{split_dir_name(train_split, size)}{limit_suffix}/images",
        "val": f"{split_dir_name(val_split, size)}{limit_suffix}/images",
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    import yaml
    p = work / "cfg" / f"{name}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
    return p


@app.function
def dataset_census(work, split="train"):
    """Per-split statistics from the raw annotations (original resolution)."""
    b = np.load(Path(work) / "data" / f"boxes_{split_dir_name(split)}.npy")
    meta = jload(Path(work) / "data" / f"meta_{split_dir_name(split)}.json")
    w, h = b[:, 4].astype(np.float64), b[:, 5].astype(np.float64)
    area = w * h
    side = np.sqrt(area)
    per_img = np.bincount(b[:, 0].astype(int), minlength=len(meta["images"]))
    cls = np.bincount(b[:, 1].astype(int), minlength=10)
    return {
        "split": split, "images": len(meta["images"]), "objects": int(len(b)), "ignore_regions": meta["n_ignore_regions"],
        "frac_area_lt_8sq": float((area < 64).mean()), "frac_area_lt_16sq": float((area < 256).mean()),
        "frac_area_lt_32sq": float((area < 1024).mean()), "frac_area_ge_96sq": float((area >= 96 ** 2).mean()),
        "frac_area_gt_256sq": float((area > 256 ** 2).mean()),
        "objects_per_image_mean": float(per_img.mean()), "objects_per_image_max": int(per_img.max()),
        "frac_images_gt100": float((per_img > 100).mean()), "frac_images_gt300": float((per_img > 300).mean()),
        "median_side_px": float(np.median(side)), "class_counts": {CLASS_NAMES[i]: int(cls[i]) for i in range(10)},
    }


# =====================================================================================================
# SECTION C -- architecture surgery: variants, role-based weight transfer, exact-preservation test
# =====================================================================================================
@app.function
def load_base_cfg(family="v10"):
    """Nano-scale YAML of the base detector as a python dict (taken from the installed Ultralytics package)."""
    import yaml
    import ultralytics
    pkg = Path(ultralytics.__file__).parent / "cfg" / "models"
    f = {"v10": pkg / "v10" / "yolov10n.yaml", "yolo26": pkg / "26" / "yolo26.yaml"}[family]
    cfg = yaml.safe_load(open(f, encoding="utf-8"))
    cfg["scales"] = {"n": cfg["scales"]["n"]}  # keep only the nano scale: no filename-based scale guessing
    cfg["scale"] = "n"
    return cfg


@app.function
def make_variant_cfg(base, variant, p2_backbone_idx=2):
    """Return a new YAML dict for one architecture variant of the base detector.

    base    : unchanged P3/P4/P5 detector.
    graft   : P3/P4/P5 path left bit-exact; a stride-4 side branch (upsample P3 neck feature, concat backbone
              stride-4 feature, one neck block) is appended and its output becomes the LAST detection level.
    naive   : textbook P2 re-wiring (YOLOv8-P2 style). The P2 output feeds a new bottom-up path, so every layer after
              the P3 top-down block changes role.
    replace : graft-style P2 branch, but the P5 level and its neck layers are removed (P2/P3/P4 heads).
    """
    cfg = copy.deepcopy(base)
    bb, head = cfg["backbone"], cfg["head"]
    n_bb = len(bb)
    det = head[-1]
    body = head[:-1]
    src = list(det[0])
    assert len(src) == 3, "expected a 3-level detector"
    p3_idx, p4_idx, p5_idx = src
    full = bb + body
    p3_spec = full[p3_idx]
    n_before = len(full)
    if variant == "base":
        return cfg
    p2_block = [-1] + copy.deepcopy(p3_spec[1:])  # same block type / repeats / args as the P3 neck block
    up_spec = [p3_idx, 1, "nn.Upsample", [None, 2, "nearest"]]
    cat_spec = [[-1, p2_backbone_idx], 1, "Concat", [1]]
    if variant == "graft":
        cfg["head"] = body + [up_spec, cat_spec, p2_block] + [[[p3_idx, p4_idx, p5_idx, n_before + 2], 1, det[2], det[3]]]
        return cfg
    if variant == "replace":
        keep = body[: p4_idx + 1 - n_bb]
        k = n_bb + len(keep)
        cfg["head"] = keep + [up_spec, cat_spec, p2_block] + [[[p3_idx, p4_idx, k + 2], 1, det[2], det[3]]]
        return cfg
    if variant == "naive":
        kinds = [x[2] for x in body]
        assert kinds[0] == "nn.Upsample" and kinds[1] == "Concat" and kinds[3] == "nn.Upsample" and kinds[4] == "Concat", kinds
        assert kinds[7] == "Concat" and kinds[10] == "Concat" and len(body) == 12, kinds
        td_p4 = n_bb + 2
        half = copy.deepcopy(p3_spec)
        half[3] = [half[3][0] // 2] + half[3][1:]
        conv_half = copy.deepcopy(body[6])
        conv_half[3] = [conv_half[3][0] // 2] + conv_half[3][1:]
        new = [copy.deepcopy(x) for x in body[:6]]
        new += [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, p2_backbone_idx], 1, "Concat", [1]],
            [-1] + half[1:],
            conv_half,
            [[-1, p3_idx], 1, "Concat", [1]],
            [-1] + copy.deepcopy(p3_spec[1:]),
            copy.deepcopy(body[6]),
            [[-1, td_p4], 1, "Concat", [1]],
            copy.deepcopy(body[8]),
            copy.deepcopy(body[9]),
            copy.deepcopy(body[10]),
            copy.deepcopy(body[11]),
        ]
        cfg["head"] = new + [[[n_bb + 8, n_bb + 11, n_bb + 14, n_bb + 17], 1, det[2], det[3]]]
        return cfg
    raise ValueError(f"unknown variant {variant}")


@app.function
def write_variant_yaml(work, family, variant):
    import yaml
    cfg = make_variant_cfg(load_base_cfg(family), variant)
    p = Path(work) / "cfg" / f"{'yolov10n' if family == 'v10' else 'yolo26n'}-{variant}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return p


@app.function
def layer_signatures(cfg):
    """Structural signature of every layer = hash(type, repeats, args, signatures of its inputs)."""
    layers = cfg["backbone"] + cfg["head"]
    sigs = []
    for i, (f, n, m, a) in enumerate(layers):
        ins = [f] if isinstance(f, int) else list(f)
        ins = [(i + x if x < 0 else x) for x in ins]
        s = json.dumps([m, n, a, [sigs[j] if j >= 0 else "input" for j in ins]], default=str)
        sigs.append(hashlib.md5(s.encode()).hexdigest()[:10])
    return sigs


@app.function
def role_key_rename(variant, n_base_layers=23):
    """(rename, drop) callables that map base-model state-dict keys to variant keys BY ROLE."""
    det_old = n_base_layers
    pref = lambda i: f"model.{i}."

    def keep(k):
        return k

    if variant == "base":
        return keep, (lambda k: False)
    if variant == "graft":
        det_new = n_base_layers + 3

        def rn(k):
            return k.replace(pref(det_old), pref(det_new), 1) if k.startswith(pref(det_old)) else k
        return rn, (lambda k: False)
    if variant == "replace":
        def drop(k):
            m = re.match(r"model\.(\d+)\.(.*)", k)
            if not m:
                return False
            i = int(m.group(1))
            if 20 <= i <= 22:
                return True
            return i == det_old and re.search(r"(cv2|cv3|one2one_cv2|one2one_cv3)\.2\.", m.group(2)) is not None
        return keep, drop
    raise ValueError(f"no role map for variant {variant!r}")


@app.function
def transfer_weights(model, pre_model, variant, mode, base_cfg=None, var_cfg=None):
    """Copy pretrained weights into `model`.
       mode='role': explicit role-based key map (graft / replace / base)    -> every copied tensor is role-aligned
       mode='key' : Ultralytics' default (same key AND same shape)           -> may copy wrong-role tensors
       mode='none': random initialisation
    Returns an accounting dict (tensors / parameters carried over, and how many of them are role-aligned)."""
    from ultralytics.utils.torch_utils import intersect_dicts
    pre_sd = pre_model.float().state_dict()
    tot_t, tot_p = len(pre_sd), int(sum(v.numel() for v in pre_sd.values()))
    acc = {"mode": mode, "variant": variant, "tensors_pretrained": tot_t, "params_pretrained": tot_p}
    if mode == "none":
        acc.update(tensors_transferred=0, params_transferred=0, tensors_aligned=0, params_aligned=0, cls_rows_remapped=0)
        return acc
    if mode == "role":
        rename, drop = role_key_rename(variant)
        csd = {rename(k): v for k, v in pre_sd.items() if not drop(k)}
        aligned_keys = set(csd)
    elif mode == "key":
        csd = dict(pre_sd)
        aligned_keys = None
    else:
        raise ValueError(mode)
    cls_remapped = model._remap_cls_by_names(csd, pre_model, verbose=False)
    upd = intersect_dicts(csd, model.state_dict())
    model.load_state_dict(upd, strict=False)
    acc["cls_rows_remapped"] = int(cls_remapped)
    acc["tensors_transferred"] = len(upd) + int(cls_remapped)
    acc["params_transferred"] = int(sum(v.numel() for v in upd.values()))
    if mode == "role":
        acc["tensors_aligned"], acc["params_aligned"] = acc["tensors_transferred"], acc["params_transferred"]
    else:
        # key-matching copies by index; a copied tensor is role-aligned only if the whole sub-graph feeding that layer
        # (types, args, wiring) is identical in the base and the variant, and the layer index is the same
        base_sig = layer_signatures(base_cfg)
        var_sig = layer_signatures(var_cfg)
        ok_t = ok_p = 0
        for k, v in upd.items():
            m = re.match(r"model\.(\d+)\.", k)
            if m and int(m.group(1)) < len(base_sig) and int(m.group(1)) < len(var_sig) and base_sig[int(m.group(1))] == var_sig[int(m.group(1))]:
                ok_t += 1
                ok_p += int(v.numel())
        acc["tensors_aligned"], acc["params_aligned"] = ok_t, ok_p
    acc["frac_params_transferred"] = acc["params_transferred"] / max(tot_p, 1)
    acc["frac_params_aligned"] = acc["params_aligned"] / max(tot_p, 1)
    return acc


@app.function
def build_detection_model(cfg, nc=10, verbose=False):
    from ultralytics.nn.tasks import DetectionModel
    c = copy.deepcopy(cfg)
    c["nc"] = nc
    return DetectionModel(c, nc=nc, ch=3, verbose=verbose)


@app.function
def load_pretrained(work, family="v10"):
    """Download (if needed) and load the COCO-pretrained nano checkpoint as a torch module."""
    from ultralytics import YOLO
    try:  # no anonymous usage analytics from a research pipeline
        from ultralytics.utils import SETTINGS
        if SETTINGS.get("sync", False):
            SETTINGS.update({"sync": False})
    except Exception:
        pass
    wdir = Path(work) / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    name = {"v10": "yolov10n.pt", "yolo26": "yolo26n.pt"}[family]
    cwd = os.getcwd()
    try:
        os.chdir(wdir)
        y = YOLO(name)
    finally:
        os.chdir(cwd)
    return y.model, str(wdir / name)


@app.function
def model_stats(model, imgsz=640):
    """Training-graph and deploy-graph parameters (one-to-many branch dropped) and deploy GFLOPs at imgsz."""
    import torch
    from ultralytics.utils.torch_utils import get_flops
    n_train = int(sum(p.numel() for p in model.parameters()))
    m = copy.deepcopy(model).eval().float()
    det = m.model[-1]
    n_o2m = int(sum(p.numel() for n, p in det.named_parameters() if n.startswith(("cv2.", "cv3."))))
    out = {"params_train_graph_M": n_train / 1e6, "params_deploy_M": (n_train - n_o2m) / 1e6}
    try:
        m.end2end = True
        m.fuse(verbose=False) if "verbose" in m.fuse.__code__.co_varnames else m.fuse()
        out["params_deploy_fused_M"] = int(sum(p.numel() for p in m.parameters())) / 1e6
        out["gflops_deploy"] = float(get_flops(m, imgsz))
    except Exception as e:  # pragma: no cover
        out["flops_error"] = repr(e)
    return out


@app.function
def exactness_report(work, family="v10", variants=("graft", "replace"), seed=0):
    """Numerically verify that the grafted model reproduces the pretrained detector on its original levels.
    Compares neck features and the raw outputs of the four prediction branches (one2many/one2one x box/cls)."""
    import torch
    torch.manual_seed(seed)
    pre, _ = load_pretrained(work, family)
    pre.eval()
    base = load_base_cfg(family)
    nc = pre.model[-1].nc
    rep = {}
    x = torch.rand(2, 3, 640, 640)
    store = {}
    h = pre.model[-1].register_forward_pre_hook(lambda mod, a: store.__setitem__("pre", [t.clone() for t in a[0]]))
    with torch.no_grad():
        pre(x)
    h.remove()
    for v in variants:
        m = build_detection_model(make_variant_cfg(base, v), nc=nc)
        acc = transfer_weights(m, pre, v, "role")
        m.eval()
        h = m.model[-1].register_forward_pre_hook(lambda mod, a: store.__setitem__("var", [t.clone() for t in a[0]]))
        with torch.no_grad():
            m(x)
        h.remove()
        det_p, det_v = pre.model[-1], m.model[-1]
        # level correspondence: graft keeps P3,P4,P5 as levels 0,1,2 ; replace keeps P3,P4 as levels 0,1
        n_cmp = 3 if v == "graft" else 2
        feat_diff = [float((store["pre"][i] - store["var"][i]).abs().max()) for i in range(n_cmp)]
        raw = []
        with torch.no_grad():
            for i in range(n_cmp):
                for nm in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
                    raw.append(float((getattr(det_p, nm)[i](store["pre"][i]) - getattr(det_v, nm)[i](store["var"][i])).abs().max()))
        rep[v] = {"levels_compared": n_cmp, "max_abs_feature_diff": max(feat_diff), "max_abs_head_output_diff": max(raw),
                  "transfer": acc, "var_feature_shapes": [tuple(t.shape[1:]) for t in store["var"]]}
    return rep


# =====================================================================================================
# SECTION C2 -- assignability analysis (which ground-truth boxes can own an anchor at all?)
# =====================================================================================================
@app.function
def count_centers_inside(x1, y1, x2, y2, stride):
    """Exact number of grid points ((i+0.5)*stride, (j+0.5)*stride) strictly inside each box (vectorised)."""
    nx = np.ceil(x2 / stride - 0.5) - np.floor(x1 / stride - 0.5) - 1.0
    ny = np.ceil(y2 / stride - 0.5) - np.floor(y1 / stride - 0.5) - 1.0
    return np.maximum(nx, 0.0) * np.maximum(ny, 0.0)


@app.function
def assignability_table(work, split="train", imgsz=640, n_shifts=4, seed=0):
    """Fraction of ground-truth boxes that own NO candidate anchor centre (a necessary condition for any positive
    assignment in TAL-style assigners), for several stride sets and small-target window floors.

    Boxes are mapped to the network input (long side = imgsz, no augmentation).  The statistic is evaluated at the true
    box positions and, for robustness to letterbox/augmentation phase, under `n_shifts` random sub-grid translations."""
    b = np.load(Path(work) / "data" / f"boxes_{split_dir_name(split)}.npy").astype(np.float64)
    # columns: img, cls, x, y, w, h, W, H
    r = imgsz / np.maximum(b[:, 6], b[:, 7])
    x1, y1 = b[:, 2] * r, b[:, 3] * r
    w, h = b[:, 4] * r, b[:, 5] * r
    cx, cy = x1 + w / 2, y1 + h / 2
    rng = np.random.default_rng(seed)
    shifts = [(0.0, 0.0)] + [tuple(rng.uniform(0, 32, size=2)) for _ in range(n_shifts)]
    out = {"n_boxes": int(len(b)), "imgsz": imgsz, "median_w_px": float(np.median(w)), "median_h_px": float(np.median(h)),
           "frac_min_side_lt_4": float((np.minimum(w, h) < 4).mean()), "frac_min_side_lt_8": float((np.minimum(w, h) < 8).mean()),
           "frac_min_side_lt_16": float((np.minimum(w, h) < 16).mean())}
    configs = {"P3-P5": [8, 16, 32], "P2-P5": [4, 8, 16, 32], "P2-P4": [4, 8, 16]}
    for name, strides in configs.items():
        for floor in (0, 8, 16):
            un, cand = [], []
            for dx, dy in shifts:
                ww, hh = np.maximum(w, floor), np.maximum(h, floor)
                bx1, bx2 = cx + dx - ww / 2, cx + dx + ww / 2
                by1, by2 = cy + dy - hh / 2, cy + dy + hh / 2
                tot = np.zeros_like(w)
                for s in strides:
                    tot += count_centers_inside(bx1, by1, bx2, by2, s)
                un.append(float((tot == 0).mean()))
                cand.append(float(tot.mean()))
            out[f"{name}|floor={floor}"] = {"frac_unassigned": float(np.mean(un)), "frac_unassigned_exact_positions": un[0],
                                            "mean_candidates": float(np.mean(cand))}
    # per-size breakdown for the default configuration vs. P2 (floor 16)
    bins = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 1e9)]
    br = []
    for lo, hi in bins:
        sel = (np.sqrt(w * h) >= lo) & (np.sqrt(w * h) < hi)
        row = {"bin": f"[{lo},{hi if hi < 1e8 else 'inf'})", "n": int(sel.sum())}
        for name, strides in (("P3-P5", [8, 16, 32]), ("P2-P5", [4, 8, 16, 32])):
            for floor in (0, 16):
                ww, hh = np.maximum(w[sel], floor), np.maximum(h[sel], floor)
                tot = np.zeros(int(sel.sum()))
                for s in strides:
                    tot += count_centers_inside(cx[sel] - ww / 2, cy[sel] - hh / 2, cx[sel] + ww / 2, cy[sel] + hh / 2, s)
                row[f"{name}|floor={floor}"] = float((tot == 0).mean()) if sel.any() else float("nan")
        br.append(row)
    out["by_size"] = br
    return out


# =====================================================================================================
# SECTION D -- training
# =====================================================================================================
@app.class_definition
@dataclass
class RunSpec:
    """One training run.  `group` is the label used in tables; `tier` orders the queue (0 = must-have)."""
    run_id: str
    group: str
    family: str = "v10"
    variant: str = "base"          # base | graft | naive | replace
    transfer: str = "role"         # role | key | none
    seed: int = 0
    epochs: int = 40
    imgsz: int = 640
    floor: Optional[int] = 16      # small-target window floor of the TAL assigner (0 = classic TAL, None = library default)
    batch: int = 32
    tier: int = 0
    eval_sizes: Tuple[int, ...] = ()
    notes: str = ""


@app.function
def apply_tal_floor(floor):
    """Patch TaskAlignedAssigner so that its small-target window floor equals `floor`; returns a restore() callable."""
    from ultralytics.utils import tal
    cls = tal.TaskAlignedAssigner
    orig = cls.__init__
    if floor is None:
        return lambda: None

    def patched(self, *a, **k):
        orig(self, *a, **k)
        self.stride_val = floor

    cls.__init__ = patched

    def restore():
        cls.__init__ = orig
    return restore


@app.function
def make_trainer_class():
    """Detection trainer whose model construction applies our explicit weight-transfer policy."""
    import yaml
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import RANK

    class GraftTrainer(DetectionTrainer):
        opts: dict = {}

        def get_model(self, cfg=None, weights=None, verbose=True):
            o = GraftTrainer.opts
            model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
            model = self.set_model_names_for_load(model)
            if weights is not None and o.get("transfer", "role") != "none":
                var_cfg = cfg if isinstance(cfg, dict) else yaml.safe_load(open(cfg, encoding="utf-8"))
                o["transfer_report"] = transfer_weights(model, weights, o["variant"], o["transfer"],
                                                        base_cfg=o["base_cfg"], var_cfg=var_cfg)
            elif o.get("transfer") == "none":
                o["transfer_report"] = {"mode": "none", "variant": o.get("variant"), "frac_params_transferred": 0.0, "frac_params_aligned": 0.0}
            return model

    return GraftTrainer


@app.function
def gpu_total_gb():
    """Total memory (GB) of GPU 0, or None without CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass
    return None


@app.function
def remeasure_latency(work, log_path=None):
    """Re-measure the latency of every finished run ONE AT A TIME.  The measurement taken at the end of a run happens while other training processes share
    the GPU and the CPU (a first real run showed 3.6-6.2 ms for models of the same architecture), so it is replaced by this sequential pass."""
    import torch
    n = 0
    for rf in sorted((Path(work) / "runs").glob("*/result.json")):
        r = jload(rf)
        if not r or not r.get("finished_at") or not r.get("spec"):
            continue
        w = rf.parent / "weights" / "last.pt"
        if not w.exists() or not torch.cuda.is_available():
            continue
        try:
            r["latency_concurrent"] = r.get("latency")
            r["latency"] = {**latency_benchmark(w, int(r["spec"]["imgsz"]), iters=300), "sequential": True}
            jdump(r, rf)
            n += 1
        except Exception as e:
            log(f"latency re-measurement failed for {rf.parent.name}: {e!r}", log_path)
        release_gpu_memory()
    return n


@app.function
def parallel_plan(cpu_count=None, gpu_gb=None):
    """How many training processes to run side by side and how many data-loader workers each gets.

    The CPU data pipeline is the bottleneck for nano models, so the CPU count decides; the GPU only caps it: a P2 model at batch 32 needs about
    20 GB on dense VisDrone batches (the task-aligned assigner alone ~12 GB when a batch contains an image with ~900 boxes: measured with
    proto/t_tal_mem.py), hence at most one process per 24 GB."""
    cpus = cpu_count or effective_cpus()
    n_par = int(max(1, min(3, cpus // 6)))
    gpu_gb = gpu_gb if gpu_gb is not None else gpu_total_gb()
    if gpu_gb:
        n_par = int(max(1, min(n_par, gpu_gb // 24)))
    workers = int(max(2, min(12, (cpus - 1) // n_par)))
    return {"n_parallel": n_par, "workers": workers, "cpus": cpus, "gpu_gb": gpu_gb}


@app.function
def plan_parallel(probe):
    """parallel_plan() from a probe_environment() result, falling back to 0 data-loader workers when worker processes cannot run (tiny /dev/shm)."""
    par = parallel_plan(probe.get("cpus"), probe.get("gpu_gb") or 0.0)
    if probe.get("dataloader_workers_ok") is False:
        par = {**par, "workers": 0}
    return par


@app.function
def choose_cache(n_parallel=1, need_gb=9.0):
    _, avail = effective_ram_gb()
    if not (avail == avail):  # NaN
        return False
    return "ram" if avail >= need_gb * n_parallel + 4 else False


@app.function
def stray_run_pids(work):
    """Training processes of an earlier worker that are still alive (state.json says running and the pid exists)."""
    out = []
    for sj in (Path(work) / "runs").glob("*/state.json"):
        st = jload(sj, {})
        if st.get("state") == "running" and st.get("pid") and pid_alive(st["pid"]):
            out.append((sj.parent.name, st["pid"]))
    return out


@app.function
def run_training(work, spec, workers=8, cache=False, device=None, fraction=1.0, data_suffix="", log_path=None):
    """Train one RunSpec with Ultralytics.  Returns a dict with paths, transfer accounting and timing."""
    import torch
    from ultralytics import YOLO
    work = Path(work)
    device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
    rdir = work / "runs" / spec.run_id
    base_cfg = load_base_cfg(spec.family)
    ycfg = write_variant_yaml(work, spec.family, spec.variant)
    data_yaml = write_data_yaml(work, spec.imgsz, limit_suffix=data_suffix)
    pre_model, pre_path = load_pretrained(work, spec.family)
    del pre_model
    Trainer = make_trainer_class()
    Trainer.opts = {"variant": spec.variant, "transfer": spec.transfer, "base_cfg": base_cfg}
    epoch_stamps: List[float] = []

    def on_epoch_end(trainer):
        epoch_stamps.append(time.time())

    y = YOLO(str(ycfg))
    y.add_callback("on_train_epoch_end", on_epoch_end)
    args = dict(
        data=str(data_yaml), epochs=spec.epochs, imgsz=spec.imgsz, batch=spec.batch, nbs=64, device=device, workers=workers,
        seed=spec.seed, deterministic=False, optimizer="SGD", lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=3.0, cos_lr=False, close_mosaic=int(min(10, max(1, spec.epochs // 4))), amp=True, cache=cache, val=False,
        plots=False, patience=spec.epochs + 10, project=str(work / "runs"), name=spec.run_id, exist_ok=True,
        pretrained=(pre_path if spec.transfer != "none" else False), fraction=fraction, verbose=False, save_period=-1,
    )
    restore = apply_tal_floor(spec.floor)
    t0 = time.time()
    try:
        y.train(trainer=Trainer, **args)
    finally:
        restore()
    wall = time.time() - t0
    ep_times = list(np.diff([t0] + epoch_stamps)) if epoch_stamps else []
    info = {"run_dir": str(rdir), "last": str(rdir / "weights" / "last.pt"), "train_wall_s": wall, "epoch_times_s": ep_times,
            "transfer_report": Trainer.opts.get("transfer_report"), "args": {k: v for k, v in args.items() if k != "device"}}
    jdump(info, rdir / "train_info.json")
    log(f"[{spec.run_id}] trained {spec.epochs} epochs in {fmt_hours(wall)}", log_path)
    del y
    release_gpu_memory()  # evaluation (and the next run of calibrate()) happens in this same process
    return info


@app.function
def release_gpu_memory():
    """Drop what a finished Ultralytics training leaves on the GPU (trainer, optimizer, EMA and cached batches are kept alive by reference cycles)."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# =====================================================================================================
# SECTION E -- evaluation: saved predictions, COCO-style metrics on ORIGINAL-resolution boxes, TIDE, latency
# =====================================================================================================
@app.function
def predict_split(work, weights, split, size, head="o2o", conf=0.001, max_det=500, batch=32, device=None,
                  limit=None, verbose=False):
    """Run a trained detector on a split and return detections in ORIGINAL image pixel coordinates.

    head='o2o' : NMS-free one-to-one branch (YOLOv10 deployment mode)
    head='o2m' : one-to-many branch followed by NMS (IoU 0.7)
    Images are pre-resized to long side `size` (INTER_AREA); boxes are rescaled back with the stored original sizes."""
    work = Path(work)
    tag = "" if limit is None else f"_lim{limit}"
    meta = jload(work / "data" / f"gt_{split_dir_name(split)}{tag}.json")
    imgs = meta["images"]
    img_dir = work / "data" / (split_dir_name(split, size) + tag) / "images"
    paths = [str(img_dir / f"{im['stem']}.jpg") for im in imgs]
    return predict_files(weights, paths, [(im["width"], im["height"]) for im in imgs], size, head=head, conf=conf, max_det=max_det,
                         batch=batch, device=device)


@app.function
def predict_files(weights, paths, orig_sizes, size, head="o2o", conf=0.001, max_det=500, batch=32, device=None):
    """Detections for a list of (pre-resized) image files, mapped back to the ORIGINAL pixel grid.

    paths[k] is the image actually fed to the network, orig_sizes[k] = (W, H) of the original image it was resized from;
    the result's `img` column is k.  On a CUDA out-of-memory error the whole pass is repeated with half the batch (a 34k-anchor P2 model at
    batch 32 needs several GB), which cannot change the detections."""
    while True:
        try:
            return predict_files_once(weights, paths, orig_sizes, size, head=head, conf=conf, max_det=max_det, batch=batch, device=device)
        except Exception as e:
            if "out of memory" in str(e).lower() and batch > 1:
                release_gpu_memory()
                batch = max(1, batch // 2)
                print(f"[predict_files] CUDA out of memory -> repeating with batch {batch}", flush=True)
                continue
            raise


@app.function
def predict_files_once(weights, paths, orig_sizes, size, head="o2o", conf=0.001, max_det=500, batch=32, device=None):
    """One prediction pass.  Ultralytics' predictor ignores `batch` for a list of paths and runs the WHOLE list as one forward pass (measured: GPU memory
    grows linearly with the number of images - 17 GB for the 548 val images with a P2 model), so the images are chunked here, and each chunk holds images
    of one shape only: the predictor then letterboxes with minimal padding, independent of the neighbours."""
    import torch
    from PIL import Image
    from ultralytics import YOLO
    device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
    model = YOLO(str(weights))
    name2idx = {Path(p).name: k for k, p in enumerate(paths)}  # the predictor may normalise paths; file names identify the image
    if len(name2idx) != len(paths):
        raise ValueError("predict_files: image file names must be unique")
    by_shape: Dict[Tuple[int, int], List[int]] = {}
    for k, p in enumerate(paths):
        with Image.open(p) as im:
            by_shape.setdefault(im.size, []).append(k)
    chunks = [ks[c0:c0 + batch] for _, ks in sorted(by_shape.items()) for c0 in range(0, len(ks), batch)]
    out_img, out_box, out_sc, out_cls = [], [], [], []
    kw = dict(imgsz=size, conf=conf, iou=0.7, max_det=max_det, device=device, batch=batch, verbose=False,
              stream=True, save=False, augment=False, agnostic_nms=False)
    if head == "o2o":
        kw["nms"] = False
    for chunk in chunks:
        for r in model.predict(source=[paths[k] for k in chunk], **kw):
            k = name2idx[Path(r.path).name]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            b = r.boxes.xyxy.cpu().numpy().astype(np.float64)
            h0, w0 = r.orig_shape
            W, H = orig_sizes[k]
            b[:, [0, 2]] *= W / w0
            b[:, [1, 3]] *= H / h0
            out_img.append(np.full(len(b), k, dtype=np.int32))
            out_box.append(b.astype(np.float32))
            out_sc.append(r.boxes.conf.cpu().numpy().astype(np.float32))
            out_cls.append(r.boxes.cls.cpu().numpy().astype(np.int8))
    if not out_img:
        return {"img": np.zeros(0, np.int32), "xyxy": np.zeros((0, 4), np.float32), "score": np.zeros(0, np.float32), "cls": np.zeros(0, np.int8)}
    P = {"img": np.concatenate(out_img), "xyxy": np.concatenate(out_box), "score": np.concatenate(out_sc), "cls": np.concatenate(out_cls)}
    order = np.argsort(P["img"], kind="stable")  # canonical order: by image index, detections in the predictor's (score) order
    return {k_: v[order] for k_, v in P.items()}


@app.function
def preds_to_coco(preds, max_per_image=None):
    xyxy = preds["xyxy"].astype(np.float64)
    wh = xyxy[:, 2:] - xyxy[:, :2]
    out = []
    order = np.lexsort((-preds["score"], preds["img"]))  # by image, then descending score
    cnt = {}
    for i in order:
        k = int(preds["img"][i])
        if max_per_image is not None:
            c = cnt.get(k, 0)
            if c >= max_per_image:
                continue
            cnt[k] = c + 1
        out.append({"image_id": k, "category_id": int(preds["cls"][i]) + 1,
                    "bbox": [float(xyxy[i, 0]), float(xyxy[i, 1]), float(wh[i, 0]), float(wh[i, 1])], "score": float(preds["score"][i])})
    return out


@app.function
def filter_conf(preds, thr):
    keep = preds["score"] >= thr
    return {k: v[keep] for k, v in preds.items()}


@app.function
def cap_per_image(preds, k):
    """Keep only the k highest-scoring detections of every image."""
    order = np.lexsort((-preds["score"], preds["img"]))
    img_sorted = preds["img"][order]
    first = np.r_[0, np.flatnonzero(np.diff(img_sorted)) + 1]
    rank = np.arange(len(order)) - np.repeat(first, np.diff(np.r_[first, len(order)]))
    keep = order[rank < k]
    return {kk: v[keep] for kk, v in preds.items()}


@app.function
def save_preds(path, preds):
    np.savez_compressed(path, **preds)


@app.function
def load_preds(path):
    z = np.load(path)
    return {k: z[k] for k in ("img", "xyxy", "score", "cls")}


@app.function
def get_coco_backends():
    """Return (COCO, COCOeval) classes; prefer faster-coco-eval (same numbers, much faster), fall back to pycocotools."""
    try:
        from faster_coco_eval import COCO, COCOeval_faster
        return COCO, COCOeval_faster, "faster_coco_eval"
    except Exception:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        return COCO, COCOeval, "pycocotools"


@app.function
def coco_evaluate(gt_json, preds, max_dets=500, backend=None, return_eval=False):
    """COCO-protocol evaluation of saved detections against ORIGINAL-resolution ground truth.

    Returns overall AP / AP50 / AP75, AP in the COCO area bins (small/medium/large) and two finer VisDrone bins
    (tiny < 16^2, xsmall 16^2..32^2), AR@{1,10,max_dets}, and per-class AP / AP50."""
    return coco_eval_dets(gt_json, preds_to_coco(preds), max_dets=max_dets, backend=backend, return_eval=return_eval)


@app.function
def coco_eval_dets(gt_json, dets, max_dets=500, backend=None, return_eval=False, n_classes=10):
    """Core of coco_evaluate: `dets` is a COCO-results list [{image_id, category_id, bbox[x,y,w,h], score}]."""
    COCO, COCOeval, name = get_coco_backends()
    if backend == "pycocotools":
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        name = "pycocotools"
    gt = COCO(str(gt_json))
    if len(dets) == 0:
        zero_area = {lab: {"AP": 0.0, "AP50": 0.0, "AR": 0.0, "AR50": 0.0} for (lab, _, _) in AREA_BINS}
        return {"backend": name, "n_dets": 0, "max_dets": max_dets, "AP": 0.0, "AP50": 0.0, "AP75": 0.0, "by_area": zero_area,
                "AR1": 0.0, "AR10": 0.0, "ARmax": 0.0, "per_class_AP": [0.0] * n_classes, "per_class_AP50": [0.0] * n_classes}
    dt = gt.loadRes(dets)
    E = COCOeval(gt, dt, "bbox")
    E.params.maxDets = [1, 10, max_dets]
    E.params.areaRng = [[lo, hi] for (_, lo, hi) in AREA_BINS]
    E.params.areaRngLbl = [n for (n, _, _) in AREA_BINS]
    E.params.imgIds = sorted(gt.getImgIds())
    E.evaluate()
    E.accumulate()
    P = np.asarray(E.eval["precision"])  # T,R,K,A,M
    R = np.asarray(E.eval["recall"])     # T,K,A,M

    def mean_valid(x):
        x = np.asarray(x)
        x = x[x > -1]
        return float(x.mean()) * 100.0 if x.size else float("nan")

    labels = [n for (n, _, _) in AREA_BINS]
    m = len(E.params.maxDets) - 1
    res = {"backend": name, "n_dets": int(len(dets)), "max_dets": max_dets}
    res["AP"] = mean_valid(P[:, :, :, 0, m])
    res["AP50"] = mean_valid(P[0, :, :, 0, m])
    res["AP75"] = mean_valid(P[5, :, :, 0, m])
    res["by_area"] = {lab: {"AP": mean_valid(P[:, :, :, a, m]), "AP50": mean_valid(P[0, :, :, a, m]),
                            "AR": mean_valid(R[:, :, a, m]), "AR50": mean_valid(R[0, :, a, m])} for a, lab in enumerate(labels)}
    res["AR1"], res["AR10"], res["ARmax"] = (mean_valid(R[:, :, 0, 0]), mean_valid(R[:, :, 0, 1]), mean_valid(R[:, :, 0, 2]))
    res["per_class_AP"] = [mean_valid(P[:, :, k, 0, m]) for k in range(P.shape[2])]
    res["per_class_AP50"] = [mean_valid(P[0, :, k, 0, m]) for k in range(P.shape[2])]
    if return_eval:
        return res, E
    return res


@app.function
def yolo_dataset_to_coco(images_dir, labels_dir, class_names, out_json):
    """Ground-truth COCO json (ORIGINAL pixel coordinates) from a YOLO-format folder pair (images + normalised `cls xc yc w h` labels)."""
    from PIL import Image
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    images, anns = [], []
    aid = 1
    for i, p in enumerate(files):
        with Image.open(p) as im:
            W, H = im.size
        images.append({"id": i, "file_name": p.name, "width": W, "height": H})
        lf = labels_dir / (p.stem + ".txt")
        if lf.exists() and lf.stat().st_size > 0:
            for row in lf.read_text().strip().splitlines():
                c, xc, yc, w, h = row.split()[:5]
                w, h = float(w) * W, float(h) * H
                x, y = float(xc) * W - w / 2, float(yc) * H - h / 2
                anns.append({"id": aid, "image_id": i, "category_id": int(float(c)) + 1, "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
                aid += 1
    jdump({"images": images, "annotations": anns, "categories": [{"id": k + 1, "name": n} for k, n in enumerate(class_names)]}, out_json)
    return {"images": len(images), "annotations": len(anns)}


@app.function
def evaluate_external(work, run_id, name, images_dir, gt_json, class_map, size=640, max_dets=500, head="o2o", device=None, batch=32):
    """Zero-shot evaluation of a finished run on ANOTHER dataset with an explicit class map.

    gt_json   COCO json in ORIGINAL pixel coordinates (see yolo_dataset_to_coco); image file names are relative to `images_dir`.
    class_map {source class index (VisDrone 0..9): category id in gt_json}; detections of unmapped classes are dropped after the per-image
              top-`max_dets` selection (the target's ground truth only contains the mapped categories).
    The images go through the same pre-processing as the VisDrone splits (INTER_AREA down-scaling to long side `size`), then the same
    prediction code and the same COCO evaluator.  Writes runs/<run_id>/external_<name>.json."""
    import cv2
    work, images_dir = Path(work), Path(images_dir)
    g = jload(gt_json)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", name)
    cache = work / "data" / f"ext_{tag}_s{size}"
    cache.mkdir(parents=True, exist_ok=True)
    cmap = {int(k): int(v) for k, v in class_map.items()}
    if not cmap:
        raise ValueError("class_map is empty")
    paths, sizes = [], []
    for im in g["images"]:
        dst = cache / f"{im['id']}.jpg"
        if not dst.exists():
            src = images_dir / im["file_name"]
            if max(im["width"], im["height"]) <= size and src.suffix.lower() in (".jpg", ".jpeg"):
                shutil.copy2(src, dst)                                 # same as prepare_split: nothing to down-scale, keep the bytes
            else:
                arr = cv2.imread(str(src), cv2.IMREAD_COLOR)
                if arr is None:
                    raise FileNotFoundError(str(src))
                h, w = arr.shape[:2]
                if max(h, w) > size:
                    r = size / max(h, w)
                    arr = cv2.resize(arr, (max(1, int(round(w * r))), max(1, int(round(h * r)))), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(dst), arr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        paths.append(str(dst))
        sizes.append((im["width"], im["height"]))
    weights = work / "runs" / run_id / "weights" / "last.pt"
    P = predict_files(weights, paths, sizes, size, head=head, conf=0.001, max_det=max_dets, batch=batch, device=device)
    keep = np.isin(P["cls"], list(cmap))
    P = {k: v[keep] for k, v in P.items()}
    lut = np.full(max(cmap) + 1, -1, dtype=np.int32)
    for s, t in cmap.items():
        lut[s] = t - 1                                                 # preds_to_coco adds 1 back
    P["cls"] = lut[P["cls"].astype(np.int64)]
    dets = preds_to_coco({**P, "img": np.asarray([g["images"][k]["id"] for k in P["img"]], dtype=np.int64)})
    res = coco_eval_dets(gt_json, dets, max_dets=max_dets, n_classes=len(g["categories"]))
    out = {"name": name, "run_id": run_id, "class_map": {str(k): v for k, v in cmap.items()}, "target_names": {str(c["id"]): c["name"] for c in g["categories"]},
           "n_images": len(g["images"]), "size": size, "head": head, "metrics": res}
    jdump(out, work / "runs" / run_id / f"external_{tag}.json")
    return out


@app.function
def tide_evaluate(gt_json, preds, max_dets=500, use_ignore_regions=False, gt_ign_json=None):
    """TIDE error decomposition (Bolya et al., ECCV'20) computed from SAVED predictions.
    NOTE: tidecv expects boxes as [x, y, width, height]."""
    from tidecv import TIDE, Data
    g = jload(gt_json)
    gt, dt = Data("gt", max_dets=max_dets), Data("pred", max_dets=max_dets)
    for c in g["categories"]:
        gt.add_class(c["id"], c["name"])
        dt.add_class(c["id"], c["name"])
    for im in g["images"]:
        gt.add_image(im["id"], im["file_name"])
        dt.add_image(im["id"], im["file_name"])
    for a in g["annotations"]:
        if a.get("iscrowd", 0):
            gt.add_ignore_region(a["image_id"], a["category_id"], a["bbox"])
        else:
            gt.add_ground_truth(a["image_id"], a["category_id"], a["bbox"])
    for d in preds_to_coco(preds, max_per_image=max_dets):
        dt.add_detection(d["image_id"], d["category_id"], d["score"], d["bbox"])
    tide = TIDE()
    tide.evaluate(gt, dt, mode=TIDE.BOX, name="pred")
    main = {k: float(v) for k, v in tide.get_main_errors()["pred"].items()}
    spec = {k: float(v) for k, v in tide.get_special_errors()["pred"].items()}
    ap50 = float(tide.runs["pred"].ap)
    return {"main": main, "special": spec, "AP50_tide": ap50}


@app.function
def latency_benchmark(weights, imgsz=640, iters=200, warmup=50, device=None):
    """Median single-image latency of the deploy graph (fused, one-to-one head, FP16 on GPU, FP32 on CPU)."""
    import torch
    from ultralytics import YOLO
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
    m = YOLO(str(weights)).model.float().eval()
    m.end2end = True
    m.fuse()
    half = dev.type == "cuda"
    m = m.to(dev)
    x = torch.rand(1, 3, imgsz, imgsz, device=dev)
    if half:
        m = m.half()
        x = x.half()
    ts = []
    with torch.no_grad():
        for i in range(warmup + iters):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            m(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            if i >= warmup:
                ts.append((time.perf_counter() - t) * 1000.0)
    return {"latency_ms_median": float(np.median(ts)), "latency_ms_p95": float(np.percentile(ts, 95)),
            "device": str(dev), "precision": "fp16" if half else "fp32", "imgsz": imgsz}


@app.function
def tag_suffix(limit):
    return "" if limit is None else f"_lim{limit}"


@app.function
def ultralytics_val(work, weights, size, data_suffix="", nms=False, device=None):
    """Independent cross-check: Ultralytics' own validator (rectangular batches, its matching rule) on the same weights."""
    import torch
    from ultralytics import YOLO
    device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
    yaml_path = write_data_yaml(work, size, limit_suffix=data_suffix)
    res = YOLO(str(weights)).val(data=str(yaml_path), imgsz=size, batch=32, conf=0.001, iou=0.7, max_det=500, device=device, plots=False,
                                 verbose=False, nms=nms, split="val")
    return {"mAP50": float(res.box.map50) * 100.0, "mAP": float(res.box.map) * 100.0, "head": "one2one" if nms is False else "one2many+NMS"}


@app.function
def evaluate_run(work, spec, device=None, splits=("val", "test-dev"), tide=True, data_suffix="", limit=None,
                 log_path=None, skip_existing=True):
    """Post-training evaluation of one run: saved predictions + metrics (both ground-truth protocols), TIDE, model stats."""
    work = Path(work)
    run = work / "runs" / spec.run_id
    wts = run / "weights" / "last.pt"
    res_path = run / "result.json"
    out = jload(res_path, {}) if skip_existing else {}
    out.update({"run_id": spec.run_id, "spec": asdict(spec), "env": env_info()})
    out.setdefault("metrics", {})
    tag = "" if limit is None else f"_lim{limit}"
    sizes = [spec.imgsz] + [s for s in spec.eval_sizes if s != spec.imgsz]
    for split in splits:
        for size in sizes:
            prepare_split(work, split, sizes=(size,), limit=limit, log_path=log_path)
            key = f"{split}|{size}|o2o"
            if key + "|std" in out["metrics"] and skip_existing:
                continue
            t0 = time.time()
            p = predict_split(work, wts, split, size, head="o2o", device=device, limit=limit)
            save_preds(run / f"preds_{split_dir_name(split)}_{size}_o2o.npz", p)
            t_pred = time.time() - t0
            for proto, gname in (("std", f"gt_{split_dir_name(split)}{tag}.json"), ("ign", f"gt_{split_dir_name(split)}_ign{tag}.json")):
                t1 = time.time()
                out["metrics"][f"{key}|{proto}"] = coco_evaluate(work / "data" / gname, p)
                out["metrics"][f"{key}|{proto}"]["eval_s"] = time.time() - t1
            out["metrics"][key + "|std"]["predict_s"] = t_pred
            log(f"[{spec.run_id}] {split}@{size}: AP50={out['metrics'][key + '|std']['AP50']:.2f} AP={out['metrics'][key + '|std']['AP']:.2f}", log_path)
            jdump(out, res_path)
        if split == "val":  # protocol-sensitivity: what common shortcuts do to the same saved predictions
            p0 = load_preds(run / f"preds_val_{spec.imgsz}_o2o.npz")
            for ptag, pp in (("conf0.05", filter_conf(p0, 0.05)), ("conf0.25", filter_conf(p0, 0.25)), ("maxdet100", cap_per_image(p0, 100))):
                k2 = f"val|{spec.imgsz}|o2o|std@{ptag}"
                if k2 not in out["metrics"] or not skip_existing:
                    out["metrics"][k2] = coco_evaluate(work / "data" / f"gt_val{tag_suffix(limit)}.json", pp)
            jdump(out, res_path)
        # secondary head: one-to-many + NMS at the training size
        key = f"{split}|{spec.imgsz}|o2m|std"
        if key not in out["metrics"] or not skip_existing:
            try:  # NMS over ~30k candidate boxes per image needs a multi-GB IoU matrix: a SECONDARY result must never cost a finished training run
                p = predict_split(work, wts, split, spec.imgsz, head="o2m", device=device, limit=limit)
                out["metrics"][key] = coco_evaluate(work / "data" / f"gt_{split_dir_name(split)}{tag}.json", p)
            except Exception as e:
                out.setdefault("secondary_errors", {})[key] = repr(e)[:300]
                log(f"[{spec.run_id}] secondary evaluation {key} failed: {repr(e)[:200]}", log_path)
            jdump(out, res_path)
    if "ultra_val" not in out:
        try:
            out["ultra_val"] = ultralytics_val(work, wts, spec.imgsz, data_suffix=data_suffix)
        except Exception as e:
            out["ultra_val"] = {"error": repr(e)}
    if tide and "tide|val" not in out:
        try:
            t0 = time.time()
            p = load_preds(run / f"preds_val_{spec.imgsz}_o2o.npz")
            out["tide|val"] = tide_evaluate(work / "data" / f"gt_val{tag}.json", p)
            out["tide|val"]["seconds"] = time.time() - t0
        except Exception as e:
            out["tide|val"] = {"error": repr(e)}
    try:
        import torch
        from ultralytics import YOLO
        mdl = YOLO(str(wts)).model
        out["model_stats"] = model_stats(mdl, spec.imgsz)
        out["latency"] = latency_benchmark(wts, spec.imgsz) if torch.cuda.is_available() else {"note": "no GPU available"}
    except Exception as e:
        out["model_stats_error"] = repr(e)
    ti = jload(run / "train_info.json", {})
    out["train"] = {"wall_s": ti.get("train_wall_s"), "epoch_times_s": ti.get("epoch_times_s"), "transfer": ti.get("transfer_report")}
    out["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    jdump(out, res_path)
    return out


# =====================================================================================================
# SECTION F -- experiment plan, time-budget planner, detached worker
# =====================================================================================================
@app.function
def spec_from_dict(d):
    d = dict(d)
    d["eval_sizes"] = tuple(d.get("eval_sizes", ()))
    return RunSpec(**d)


@app.function
def run_id_for(family, variant, tag, seed):
    return f"{family}-{variant}-{tag}-s{seed}"


@app.function
def make_plan(epochs, seeds=(0, 1, 2), batch=32, families=("v10",), tiers=(0, 1, 2, 3), iso_size=896):
    """Priority-ordered list of RunSpecs.  Tier 0 = headline contrast (3 seeds), tier 1 = one seed of everything else the hypotheses need
    (compute-matched resolution baseline, naive re-wiring, classic-TAL factorial cells, P5 removal, random init), tier 2 = more seeds of
    those cells (completes the factorial first), tier 3 = second detector family."""
    P: List[RunSpec] = []

    def add(tier, family, variant, tag, group, seed, transfer="role", floor=16, imgsz=640, eval_sizes=(), notes=""):
        P.append(RunSpec(run_id=run_id_for(family, variant, tag, seed), group=group, family=family, variant=variant, transfer=transfer,
                         seed=seed, epochs=epochs, imgsz=imgsz, floor=floor, batch=batch, tier=tier, eval_sizes=tuple(eval_sizes), notes=notes))

    fam = families[0]
    if 0 in tiers:
        for s in seeds:  # interleaved so that partial completion still yields paired seeds
            add(0, fam, "base", "f16", "base", s, eval_sizes=(800, 960) if s == seeds[0] else ())
            add(0, fam, "graft", "f16", "graft", s, eval_sizes=(800, 960) if s == seeds[0] else ())
    if 1 in tiers:
        s0 = seeds[0]
        add(1, fam, "base", f"res{iso_size}", f"base @ {iso_size}px (compute-matched)", s0, imgsz=iso_size, notes="iso-GFLOPs reference (H4)")
        add(1, fam, "naive", "key", "naive P2 (key-matched transfer)", s0, transfer="key", notes="H2")
        add(1, fam, "base", "f0", "base (classic TAL)", s0, floor=0, notes="H3")
        add(1, fam, "graft", "f0", "graft (classic TAL)", s0, floor=0, notes="H3")
        add(1, fam, "replace", "role", "P2 replaces P5 (role transfer)", s0, transfer="role")
        add(1, fam, "graft", "scratch", "graft, random init", s0, transfer="none")
        add(1, fam, "base", "scratch", "base, random init", s0, transfer="none")
    if 2 in tiers:
        for s in seeds[1:]:  # complete the 2x2 factorial first
            add(2, fam, "base", "f0", "base (classic TAL)", s, floor=0)
            add(2, fam, "graft", "f0", "graft (classic TAL)", s, floor=0)
        for s in seeds[1:]:
            add(2, fam, "naive", "key", "naive P2 (key-matched transfer)", s, transfer="key")
            add(2, fam, "base", f"res{iso_size}", f"base @ {iso_size}px (compute-matched)", s, imgsz=iso_size)
        for s in seeds[1:]:
            add(2, fam, "replace", "role", "P2 replaces P5 (role transfer)", s, transfer="role")
    if 3 in tiers and len(families) > 1:
        f2 = families[1]
        for s in seeds[:2]:
            add(3, f2, "base", "f16", f"{f2} base", s)
            add(3, f2, "graft", "f16", f"{f2} graft", s)
    return P


@app.function
def cost_factor(spec):
    """Relative training cost of a run (pixels dominate: data loading + activations)."""
    return (spec.imgsz / 640.0) ** 2 * (1.12 if spec.variant in ("graft", "naive", "replace") else 1.0)


@app.function
def choose_epochs(budget_s, specs_template, s_per_epoch, eval_s_per_run=240.0, fixed_s=900.0, lo=20, hi=100, tiers=(0, 1)):
    """Largest epoch count (multiple of 5) for which all runs of the given tiers fit into `budget_s`."""
    must = [s for s in specs_template if s.tier in tiers]
    units = sum(cost_factor(s) for s in must)
    avail = budget_s - fixed_s - eval_s_per_run * len(must)
    e = int(avail // max(units * s_per_epoch, 1e-6))
    e = max(lo, min(hi, e - e % 5))
    return e, {"runs_in_budget": len(must), "cost_units": units, "est_total_s": units * s_per_epoch * e + eval_s_per_run * len(must) + fixed_s}


@app.function
def estimate_run_seconds(spec, s_per_epoch, eval_s=240.0, ratio=1.0):
    return spec.epochs * s_per_epoch * cost_factor(spec) * ratio + eval_s * (1 + 0.3 * len(spec.eval_sizes))


@app.function
def prepare_all_data(work, imgszs=(640,), splits=("train", "val", "test-dev"), limit=None, log_path=None):
    for sz in sorted(set(imgszs)):
        for sp in splits:
            prepare_split(work, sp, sizes=(sz,), limit=limit, log_path=log_path)


@app.function
def run_one(work, spec_dict, workers=8, cache=False, device=None, data_suffix="", limit=None, tide=True, test_dev=True):
    """Child-process entry point: train (resuming if possible), evaluate, write result.json.  Never raises."""
    work = Path(work)
    spec = spec_from_dict(spec_dict)
    lp = work / "logs" / f"{spec.run_id}.log"
    rdir = work / "runs" / spec.run_id
    status_path = rdir / "state.json"
    try:
        if (rdir / "result.json").exists() and jload(rdir / "result.json", {}).get("finished_at"):
            log(f"[{spec.run_id}] already finished -> skip", lp)
            return True
        jdump({"state": "running", "started": time.time(), "pid": os.getpid()}, status_path)
        prepare_all_data(work, imgszs=[spec.imgsz], splits=("train", "val") + (("test-dev",) if test_dev else ()), limit=limit, log_path=lp)
        if spec.eval_sizes:
            prepare_all_data(work, imgszs=list(spec.eval_sizes), splits=("val",) + (("test-dev",) if test_dev else ()), limit=limit, log_path=lp)
        info = jload(rdir / "train_info.json")
        if info is None:
            # resume a partially trained run if possible, otherwise start from scratch
            last = rdir / "weights" / "last.pt"
            if last.exists():
                try:
                    info = resume_training(work, spec, workers=workers, cache=cache, device=device, data_suffix=data_suffix, log_path=lp)
                except Exception as e:
                    log(f"[{spec.run_id}] resume failed ({e!r}); restarting from scratch", lp)
                    shutil.rmtree(rdir, ignore_errors=True)
                    info = None
            if info is None:
                batch = spec.batch
                for attempt in range(3):
                    oom = False
                    try:
                        info = run_training(work, spec, workers=workers, cache=cache, device=device, data_suffix=data_suffix, log_path=lp)
                        break
                    except Exception as e:
                        if "out of memory" in str(e).lower() and attempt < 2:
                            oom = True
                        else:
                            raise
                    if oom:  # clean up outside the except block so that the failed attempt's tensors can be freed
                        release_gpu_memory()
                        batch = max(4, batch // 2)
                        spec = dataclasses.replace(spec, batch=batch)
                        log(f"[{spec.run_id}] CUDA OOM -> retry with batch {batch} (nbs fixed at 64)", lp)
                        shutil.rmtree(rdir, ignore_errors=True)
        evaluate_run(work, spec, device=device, splits=("val",) + (("test-dev",) if test_dev else ()), tide=tide,
                     data_suffix=data_suffix, limit=limit, log_path=lp)
        jdump({"state": "done", "finished": time.time()}, status_path)
        return True
    except Exception as e:
        log(f"[{spec.run_id}] FAILED: {e!r}\n{traceback.format_exc()}", lp)
        jdump({"state": "failed", "error": repr(e), "finished": time.time()}, status_path)
        return False


@app.function
def resume_training(work, spec, workers=8, cache=False, device=None, data_suffix="", log_path=None):
    import torch
    from ultralytics import YOLO
    work = Path(work)
    rdir = work / "runs" / spec.run_id
    device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
    Trainer = make_trainer_class()
    Trainer.opts = {"variant": spec.variant, "transfer": spec.transfer, "base_cfg": load_base_cfg(spec.family)}
    restore = apply_tal_floor(spec.floor)
    t0 = time.time()
    try:
        YOLO(str(rdir / "weights" / "last.pt")).train(trainer=Trainer, resume=True)
    finally:
        restore()
    info = {"run_dir": str(rdir), "last": str(rdir / "weights" / "last.pt"), "train_wall_s": time.time() - t0, "epoch_times_s": [],
            "transfer_report": None, "resumed": True}
    jdump(info, rdir / "train_info.json")
    return info


@app.function
def calibrate(work, workers=8, cache=False, device=None, imgsz=640, log_path=None):
    """Measure seconds/epoch on 10% of the training set for the base and the grafted model (steady-state = last epoch)."""
    out = {}
    for variant in ("base", "graft"):
        spec = RunSpec(run_id=f"calib-{variant}", group="calib", variant=variant, epochs=3, imgsz=imgsz, batch=32, seed=0)
        shutil.rmtree(Path(work) / "runs" / spec.run_id, ignore_errors=True)
        info = run_training(work, spec, workers=workers, cache=cache, device=device, fraction=0.1, log_path=log_path)
        et = info["epoch_times_s"]
        out[variant] = float(et[1] * 10.0) if len(et) > 1 else (float(et[-1] * 10.0) if et else float("nan"))
        out[variant + "_startup_s"] = float(info["train_wall_s"] - sum(et)) if et else float("nan")
    out["s_per_epoch"] = float(np.nanmax([out["base"], out["graft"]]))
    jdump(out, Path(work) / "calibration.json")
    return out


@app.function
def launch_worker(work, plan, notebook_file=None, extra_env=None):
    """Spawn the detached scheduler process (survives notebook-kernel restarts)."""
    work = Path(work)
    (work / "logs").mkdir(parents=True, exist_ok=True)
    nb = Path(notebook_file) if notebook_file else NOTEBOOK_FILE
    if nb is None or not Path(nb).exists():
        raise RuntimeError("cannot locate the notebook file; pass notebook_file=...")
    plan = dict(plan)
    plan["notebook_file"] = str(Path(nb).resolve())
    plan_path = work / "plan.json"
    jdump(plan, plan_path)
    code = ("import sys, importlib.util; "
            f"spec = importlib.util.spec_from_file_location('graft_p2_study_mod', r'{plan['notebook_file']}'); "
            "m = importlib.util.module_from_spec(spec); sys.modules['graft_p2_study_mod'] = m; spec.loader.exec_module(m); "
            f"m.worker_main(r'{plan_path}')")
    env = dict(os.environ)
    env.update({"GRAFT_WORKDIR": str(work), "YOLO_CONFIG_DIR": str(work / "ultralytics_cfg"), "PYTHONUNBUFFERED": "1"})
    env.update(extra_env or {})
    logf = open(work / "logs" / "worker.log", "a")
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = 0x00000200 | 0x00000008  # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
    else:
        kw["start_new_session"] = True
    p = subprocess.Popen([sys.executable, "-c", code], cwd=str(work), env=env, stdout=logf, stderr=subprocess.STDOUT, **kw)
    jdump({"pid": p.pid, "started": time.time()}, work / "worker.json")
    return p.pid


@app.function
def pid_alive(pid):
    try:
        import psutil
        return psutil.pid_exists(int(pid)) and psutil.Process(int(pid)).status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


@app.function
def worker_main(plan_path):
    """Scheduler: keeps `n_parallel` run_one() children alive, honours the deadline, writes status.json."""
    plan = jload(plan_path)
    work = Path(plan["work"])
    (work / "logs").mkdir(parents=True, exist_ok=True)
    strays = stray_run_pids(work)
    if strays:
        log(f"refusing to start: training processes of an earlier worker are still alive: {strays}", work / "logs" / "worker.log")
        return
    specs = [spec_from_dict(d) for d in plan["specs"]]
    n_par = int(plan.get("n_parallel", 1))
    deadline = float(plan["deadline"])
    hard_deadline = float(plan.get("hard_deadline", deadline + 1800))
    s_per_epoch = float(plan["s_per_epoch"])
    ratio = 1.0
    queue = list(specs)
    running: Dict[str, Any] = {}
    status: Dict[str, Any] = {s.run_id: {"state": "queued", "tier": s.tier, "group": s.group} for s in specs}
    lp = work / "logs" / "worker.log"
    (work / "logs").mkdir(parents=True, exist_ok=True)
    mod_path = Path(plan["notebook_file"]).resolve()
    child_code = ("import sys, json, importlib.util; "
                  f"spec = importlib.util.spec_from_file_location('graft_p2_study_mod', r'{mod_path}'); "
                  "m = importlib.util.module_from_spec(spec); sys.modules['graft_p2_study_mod'] = m; spec.loader.exec_module(m); "
                  "a = json.loads(sys.argv[1]); m.run_one(**a)")

    def write_status():
        jdump({"updated": time.time(), "deadline": deadline, "ratio": ratio, "runs": status, "queue_left": [s.run_id for s in queue]}, work / "status.json")

    # runs that already have a finished result are skipped immediately
    for s in list(queue):
        r = jload(work / "runs" / s.run_id / "result.json", {})
        if r.get("finished_at"):
            status[s.run_id]["state"] = "done"
            queue.remove(s)
    write_status()
    log(f"worker started: {len(queue)} runs queued, {n_par} parallel, deadline in {fmt_hours(deadline - time.time())}", lp)
    oom_retried: set = set()
    stagger_s = float(plan.get("stagger_s", 60.0))
    last_start = 0.0
    while queue or running:
        # reap finished children
        for rid, (proc, spec, t0) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                st = jload(work / "runs" / rid / "result.json", {})
                ok = bool(st.get("finished_at"))
                status[rid].update(state="done" if ok else "failed", seconds=time.time() - t0, returncode=rc)
                if not ok and rid not in oom_retried and spec.batch > 8 and "out of memory" in str(jload(work / "runs" / rid / "state.json", {}).get("error", "")).lower():
                    # the CUDA context of a process that hit an OOM can be unusable: start the run again in a FRESH process with half the batch (nbs stays 64)
                    oom_retried.add(rid)
                    nspec = dataclasses.replace(spec, batch=max(8, spec.batch // 2))
                    shutil.rmtree(work / "runs" / rid, ignore_errors=True)
                    queue.insert(0, nspec)
                    status[rid].update(state="queued", note=f"CUDA out of memory: retried in a fresh process with batch {nspec.batch}")
                    log(f"run {rid} failed with CUDA out of memory -> re-queued with batch {nspec.batch}", lp)
                    del running[rid]
                    continue
                if ok and rc == 0:
                    est = estimate_run_seconds(spec, s_per_epoch, plan.get("eval_s", 240.0), 1.0)
                    ratio = 0.7 * ratio + 0.3 * ((time.time() - t0) / max(est, 1.0))
                log(f"run {rid} finished (ok={ok}, {fmt_hours(time.time() - t0)}), runtime ratio now {ratio:.2f}", lp)
                del running[rid]
            elif time.time() > hard_deadline:
                proc.terminate()
                status[rid].update(state="killed (hard deadline)")
        # start new children if time allows (staggered: simultaneous starters would race on shared downloads and caches)
        while queue and len(running) < n_par:
            if time.time() - last_start < stagger_s:
                break
            spec = queue[0]
            est = estimate_run_seconds(spec, s_per_epoch, plan.get("eval_s", 240.0), ratio) * (1.0 + 0.15 * max(0, len(running)))
            if time.time() + est > deadline:
                status[spec.run_id]["state"] = "skipped (would exceed time budget)"
                queue.pop(0)
                log(f"skip {spec.run_id}: est {fmt_hours(est)} exceeds remaining {fmt_hours(deadline - time.time())}", lp)
                continue
            queue.pop(0)
            args = {"work": str(work), "spec_dict": asdict(spec), "workers": plan["workers"], "cache": plan["cache"],
                    "device": plan.get("device"), "data_suffix": plan.get("data_suffix", ""), "limit": plan.get("limit"),
                    "tide": plan.get("tide", True), "test_dev": plan.get("test_dev", True)}
            env = dict(os.environ)
            env.update({"GRAFT_WORKDIR": str(work), "YOLO_CONFIG_DIR": str(work / "ultralytics_cfg"), "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2"})
            outf = open(work / "logs" / f"{spec.run_id}.out", "a")
            proc = subprocess.Popen([sys.executable, "-c", child_code, json.dumps(args)], cwd=str(work), env=env, stdout=outf, stderr=subprocess.STDOUT)
            running[spec.run_id] = (proc, spec, time.time())
            last_start = time.time()
            status[spec.run_id].update(state="running", started=time.time(), est_s=est)
            log(f"started {spec.run_id} (est {fmt_hours(est)})", lp)
        write_status()
        time.sleep(5)
    write_status()
    log("worker finished", lp)
    if not plan.get("limit"):
        try:
            log(f"sequential latency re-measurement: {remeasure_latency(work, lp)} runs", lp)
        except Exception as e:
            log(f"latency re-measurement failed: {e!r}", lp)
    if plan.get("auto_paper", True) and not plan.get("limit"):
        # build the paper now: the user may be away, and nothing but uploaded files survives a molab shutdown, so the bundle should be ready to download
        try:
            po = plan.get("paper") or {}
            res = build_paper(work, B=int(po.get("B", 300)), iso_size=int(po.get("iso_size", 896)), template=po.get("template", "lncs"), log_path=work / "logs" / "analysis.log")
            log(f"paper written: {res['out_dir']} (bundle: {res['bundle']})", lp)
        except Exception as e:
            log(f"automatic paper build failed (press step 9 to retry): {e!r}\n{traceback.format_exc()}", lp)


# =====================================================================================================
# SECTION G -- results table, statistics (seeds + image-level bootstrap), figures
# =====================================================================================================
@app.function
def flatten_result(r):
    sp = r["spec"]
    # evaluation at other test sizes stores AP50 only through the dedicated keys below
    row = {k: sp.get(k) for k in ("run_id", "group", "family", "variant", "transfer", "floor", "seed", "epochs", "imgsz", "tier", "batch")}
    for split, short in (("val", "val"), ("test-dev", "test")):
        m = r.get("metrics", {}).get(f"{split}|{sp['imgsz']}|o2o|std")
        if not m:
            continue
        ba = m["by_area"]
        row.update({f"{short}_AP": m["AP"], f"{short}_AP50": m["AP50"], f"{short}_AP75": m["AP75"], f"{short}_APs": ba["small"]["AP"],
                    f"{short}_APm": ba["medium"]["AP"], f"{short}_APl": ba["large"]["AP"], f"{short}_AP_tiny": ba["tiny"]["AP"],
                    f"{short}_AP_xsmall": ba["xsmall"]["AP"], f"{short}_AP50_tiny": ba["tiny"]["AP50"], f"{short}_AP50_xsmall": ba["xsmall"]["AP50"],
                    f"{short}_AP50_small": ba["small"]["AP50"], f"{short}_AP50_medium": ba["medium"]["AP50"], f"{short}_AP50_large": ba["large"]["AP50"],
                    f"{short}_ARmax": m["ARmax"], f"{short}_ARtiny": ba["tiny"]["AR"]})
        if split == "val":
            row["val_perclass_AP50"] = m["per_class_AP50"]
            row["val_perclass_AP"] = m["per_class_AP"]
        mi = r["metrics"].get(f"{split}|{sp['imgsz']}|o2o|ign")
        if mi:
            row[f"{short}_AP50_ign"], row[f"{short}_AP_ign"] = mi["AP50"], mi["AP"]
        m2 = r["metrics"].get(f"{split}|{sp['imgsz']}|o2m|std")
        if m2:
            row[f"{short}_AP50_o2m"], row[f"{short}_AP_o2m"] = m2["AP50"], m2["AP"]
    for sz in (800, 896, 960):
        m = r.get("metrics", {}).get(f"val|{sz}|o2o|std")
        if m and sz != sp["imgsz"]:
            row[f"val_AP50_at{sz}"], row[f"val_AP_at{sz}"], row[f"val_APs_at{sz}"] = m["AP50"], m["AP"], m["by_area"]["small"]["AP"]
    for ptag, short in (("conf0.05", "conf05"), ("conf0.25", "conf25"), ("maxdet100", "maxdet100")):
        mm = r.get("metrics", {}).get(f"val|{sp['imgsz']}|o2o|std@{ptag}")
        if mm:
            row[f"val_AP50_{short}"], row[f"val_AP_{short}"] = mm["AP50"], mm["AP"]
    uv = r.get("ultra_val") or {}
    if "mAP50" in uv:
        row["val_ultra_AP50"], row["val_ultra_AP"] = uv["mAP50"], uv["mAP"]
    ms = r.get("model_stats") or {}
    row["params_deploy_M"] = ms.get("params_deploy_fused_M")
    row["params_train_M"] = ms.get("params_train_graph_M")
    row["gflops"] = ms.get("gflops_deploy")
    lat = r.get("latency") or {}
    row["latency_ms"] = lat.get("latency_ms_median")
    row["latency_seq"] = bool(lat.get("sequential", False))
    tr = r.get("train") or {}
    row["train_h"] = (tr.get("wall_s") or float("nan")) / 3600.0
    tf = tr.get("transfer") or {}
    row["frac_transferred"] = tf.get("frac_params_transferred")
    row["frac_aligned"] = tf.get("frac_params_aligned")
    td = r.get("tide|val") or {}
    for k, v in (td.get("main") or {}).items():
        row[f"tide_{k}"] = v
    for k, v in (td.get("special") or {}).items():
        row[f"tide_{k}"] = v
    return row


@app.function
def collect_results(work):
    import pandas as pd
    rows = []
    for rf in sorted((Path(work) / "runs").glob("*/result.json")):
        r = jload(rf)
        if r and r.get("finished_at") and r.get("spec") and not r["spec"]["run_id"].startswith(("calib", "smoke", "mini", "wk-")):
            row = flatten_result(r)
            for jf in sorted(rf.parent.glob("external_*.json")):  # optional zero-shot evaluations on other datasets
                j = jload(jf)
                if j and j.get("metrics"):
                    nm = re.sub(r"[^A-Za-z0-9]+", "_", j["name"])
                    row[f"ext_{nm}_AP"], row[f"ext_{nm}_AP50"] = j["metrics"]["AP"], j["metrics"]["AP50"]
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        for c in ("floor", "imgsz", "seed", "epochs", "tier", "batch"):
            if c in df:
                df[c] = df[c].fillna(-1).astype(int)
    return df


@app.function
def parse_class_map(text, n_target):
    """'3:0, 5:1, 8:2' -> {3: 1, 5: 2, 8: 3}: VisDrone class index -> 1-based category id of the target dataset (given as 0-based class index)."""
    cmap = {}
    for part in [p for p in text.replace(" ", "").split(",") if p]:
        a, b = part.split(":")
        a, b = int(a), int(b)
        if not (0 <= a < len(CLASS_NAMES)):
            raise ValueError(f"source class {a} is not a VisDrone class index (0..{len(CLASS_NAMES) - 1})")
        if not (0 <= b < n_target):
            raise ValueError(f"target class index {b} is outside 0..{n_target - 1}")
        cmap[a] = b + 1
    if not cmap:
        raise ValueError("empty class map")
    return cmap


@app.function
def run_external_check(work, folder, name, class_names, class_map_text, size=640, family="v10", log_path=None):
    """Zero-shot evaluation of the finished baseline / P2-Graft runs (the main pair) on a YOLO-format folder of ANOTHER dataset.
    `folder` must contain images/ and labels/.  Returns the list of run ids evaluated."""
    work, folder = Path(work), Path(folder).expanduser()
    if not ((folder / "images").is_dir() and (folder / "labels").is_dir()):
        raise FileNotFoundError(f"{folder} must contain images/ and labels/ (YOLO format)")
    cmap = parse_class_map(class_map_text, len(class_names))
    tag = re.sub(r"[^A-Za-z0-9]+", "_", name)
    gt = work / "data" / f"gt_ext_{tag}.json"
    info = yolo_dataset_to_coco(folder / "images", folder / "labels", class_names, gt)
    log(f"external dataset {name}: {info['images']} images, {info['annotations']} boxes; class map {cmap}", log_path)
    df = collect_results(work)
    K = canonical_keys(family)
    done = []
    for key in ("base", "graft"):
        for _seed, rid in runs_for_key(df, K[key]):
            evaluate_external(work, rid, name, folder / "images", gt, cmap, size=size)
            log(f"[{rid}] external {name}: done", log_path)
            done.append(rid)
    return done


@app.function
def collect_external(work):
    """{dataset name: {class_map, n_images, size}} for every optional zero-shot evaluation found under runs/."""
    info = {}
    for jf in sorted((Path(work) / "runs").glob("*/external_*.json")):
        j = jload(jf)
        if j and j.get("name"):
            e = info.setdefault(j["name"], {"class_map": j.get("class_map", {}), "target_names": j.get("target_names", {}), "n_images": j.get("n_images"),
                                            "size": j.get("size"), "column": re.sub(r"[^A-Za-z0-9]+", "_", j["name"]), "runs": []})
            e["runs"].append(j.get("run_id"))
    return info


@app.function
def group_key(row_or_df):
    """Seed-free identity of an experimental condition."""
    return (row_or_df["family"], row_or_df["variant"], row_or_df["transfer"], row_or_df["floor"], row_or_df["imgsz"])


@app.function
def aggregate_groups(df, metrics):
    """Mean / std (ddof=1) / n over seeds for every condition."""
    import pandas as pd
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["_key"] = d.apply(lambda r: "|".join(str(x) for x in group_key(r)), axis=1)
    out = []
    for key, g in d.groupby("_key", sort=False):
        row = {"key": key, "group": g["group"].iloc[0], "family": g["family"].iloc[0], "variant": g["variant"].iloc[0],
               "transfer": g["transfer"].iloc[0], "floor": g["floor"].iloc[0], "imgsz": int(g["imgsz"].iloc[0]),
               "n": int(len(g)), "seeds": sorted(int(s) for s in g["seed"])}
        for m in metrics:
            if m in g:
                vals = g[m].astype(float).to_numpy()
                vals = vals[~np.isnan(vals)]
                row[m] = float(vals.mean()) if len(vals) else float("nan")
                row[m + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        out.append(row)
    return pd.DataFrame(out)


@app.function
def paired_comparison(df, key_a, key_b, metric):
    """Paired-by-seed comparison B - A on `metric` (key_* are group_key tuples)."""
    from scipy import stats as st
    def pick(k):
        sel = df[(df.family == k[0]) & (df.variant == k[1]) & (df.transfer == k[2]) & (df.floor == k[3]) & (df.imgsz == k[4])]
        return {int(s): float(v) for s, v in zip(sel.seed, sel[metric])}
    a, b = pick(key_a), pick(key_b)
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    d = np.array([b[s] - a[s] for s in seeds])
    out = {"metric": metric, "seeds": seeds, "per_seed_delta": d.tolist(), "mean_delta": float(d.mean()),
           "std_delta": float(d.std(ddof=1)) if len(d) > 1 else float("nan"), "n": len(d)}
    if len(d) > 1 and d.std(ddof=1) > 0:
        t = st.ttest_rel([b[s] for s in seeds], [a[s] for s in seeds])
        out.update(t_stat=float(t.statistic), p_value=float(t.pvalue))
        se = d.std(ddof=1) / math.sqrt(len(d))
        tcrit = st.t.ppf(0.975, len(d) - 1)
        out.update(ci95_t=(float(d.mean() - tcrit * se), float(d.mean() + tcrit * se)))
    return out


# ------------------------------------------------------------------ image-level paired bootstrap (COCO AP)
@app.function
def boot_prepare(gt_json, preds, max_dets=500):
    """Pre-compute per-detection TP/FP flags (all IoU thresholds) with pycocotools so that AP can be re-accumulated
    quickly under image re-sampling."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib_redirect():
        gt = COCO(str(gt_json))
        dt = gt.loadRes(preds_to_coco(preds))
        E = COCOeval(gt, dt, "bbox")
        E.params.maxDets = [max_dets]
        E.params.areaRng = [[0, 1e10]]
        E.params.areaRngLbl = ["all"]
        E.params.imgIds = sorted(gt.getImgIds())
        E.evaluate()
    img_ids = E.params.imgIds
    cat_ids = E.params.catIds
    I, K, T = len(img_ids), len(cat_ids), len(E.params.iouThrs)
    cats = []
    for k in range(K):
        sc, ii, tp, ig, gi = [], [], [], [], []
        for i in range(I):
            e = E.evalImgs[k * I + i]
            if e is None:
                continue
            n = len(e["dtScores"])
            sc.append(np.asarray(e["dtScores"]))
            ii.append(np.full(n, i, dtype=np.int32))
            tp.append(np.asarray(e["dtMatches"]) > 0)
            ig.append(np.asarray(e["dtIgnore"]).astype(bool))
            gi.append(np.full(int((np.asarray(e["gtIgnore"]) == 0).sum()), i, dtype=np.int32))
        if sc:
            sc = np.concatenate(sc)
            order = np.argsort(-sc, kind="mergesort")
            cats.append({"scores": sc[order], "img": np.concatenate(ii)[order], "tp": np.concatenate(tp, axis=1)[:, order],
                         "ig": np.concatenate(ig, axis=1)[:, order], "gt_img": np.concatenate(gi) if gi else np.zeros(0, np.int32)})
        else:
            cats.append({"scores": np.zeros(0), "img": np.zeros(0, np.int32), "tp": np.zeros((T, 0), bool), "ig": np.zeros((T, 0), bool),
                         "gt_img": np.concatenate(gi) if gi else np.zeros(0, np.int32)})
    return {"n_img": I, "T": T, "cats": cats}


@app.function
def contextlib_redirect():
    import contextlib
    return contextlib.redirect_stdout(io.StringIO())


@app.function
def boot_ap(prep, w):
    """COCO AP (mean over IoU .50:.95 and classes) and AP50 under integer image weights `w` (all ones = original sample)."""
    rec = np.linspace(0.0, 1.0, 101)
    aps = np.full((prep["T"], len(prep["cats"])), np.nan)
    for k, c in enumerate(prep["cats"]):
        npig = w[c["gt_img"]].sum()
        if npig == 0:
            continue
        ww = w[c["img"]]
        for t in range(prep["T"]):
            keep_w = ww * (~c["ig"][t])
            tps = np.cumsum(keep_w * c["tp"][t])
            fps = np.cumsum(keep_w * (~c["tp"][t]))
            if len(tps) == 0:
                aps[t, k] = 0.0
                continue
            rc = tps / npig
            pr = tps / (tps + fps + np.spacing(1))
            pr = np.maximum.accumulate(pr[::-1])[::-1]
            idx = np.searchsorted(rc, rec, side="left")
            q = np.zeros(len(rec))
            ok = idx < len(rc)
            q[ok] = pr[idx[ok]]
            aps[t, k] = q.mean()
    return float(np.nanmean(aps) * 100.0), float(np.nanmean(aps[0]) * 100.0)


@app.function
def paired_bootstrap(preps_a, preps_b, B=200, seed=0):
    """Image-level paired bootstrap for the difference in (seed-averaged) AP / AP50 between two conditions.
    preps_a / preps_b: lists (one entry per seed) of boot_prepare() outputs on the SAME images."""
    rng = np.random.default_rng(seed)
    n = preps_a[0]["n_img"]
    ones = np.ones(n, dtype=np.float64)
    base_a = np.mean([boot_ap(p, ones) for p in preps_a], axis=0)
    base_b = np.mean([boot_ap(p, ones) for p in preps_b], axis=0)
    d_ap, d_ap50 = [], []
    for _ in range(B):
        w = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(np.float64)
        a = np.mean([boot_ap(p, w) for p in preps_a], axis=0)
        b = np.mean([boot_ap(p, w) for p in preps_b], axis=0)
        d_ap.append(b[0] - a[0])
        d_ap50.append(b[1] - a[1])
    q = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    return {"A_AP": float(base_a[0]), "B_AP": float(base_b[0]), "A_AP50": float(base_a[1]), "B_AP50": float(base_b[1]),
            "delta_AP": float(base_b[0] - base_a[0]), "delta_AP_ci95": q(d_ap), "delta_AP50": float(base_b[1] - base_a[1]),
            "delta_AP50_ci95": q(d_ap50), "B": B, "frac_boot_positive_AP": float(np.mean(np.asarray(d_ap) > 0)),
            "frac_boot_positive_AP50": float(np.mean(np.asarray(d_ap50) > 0))}


@app.function
def run_pred_path(work, run_id, split="val", size=640):
    return Path(work) / "runs" / run_id / f"preds_{split_dir_name(split)}_{size}_o2o.npz"


# ------------------------------------------------------------------ transfer audit (static, no training)
@app.function
def transfer_audit(work, family="v10", nc=10):
    """Parameters carried over from the COCO checkpoint by every (variant, transfer-mode) pair, plus model statistics."""
    pre, _ = load_pretrained(work, family)
    base = load_base_cfg(family)
    ref = None
    rows = []
    combos = [("base", "role"), ("graft", "role"), ("graft", "key"), ("naive", "key"), ("replace", "role"), ("replace", "key")]
    for v, mode in combos:
        cfg = make_variant_cfg(base, v)
        m = build_detection_model(cfg, nc=nc)
        m.names = {i: n for i, n in enumerate(CLASS_NAMES)}
        acc = transfer_weights(m, pre, v, mode, base_cfg=base, var_cfg=cfg)
        st = model_stats(m, 640)
        if v == "base" and mode == "role":
            ref = acc["params_transferred"]  # the most any nc-class model can inherit
        rows.append({"variant": v, "mode": mode, "tensors_transferred": acc["tensors_transferred"], "params_transferred": acc["params_transferred"],
                     "params_aligned": acc["params_aligned"], "params_pretrained": acc["params_pretrained"], **st})
    for r in rows:
        r["frac_of_inheritable"] = r["params_transferred"] / ref
        r["frac_aligned_of_inheritable"] = r["params_aligned"] / ref
    return {"rows": rows, "inheritable_params": ref}


# ------------------------------------------------------------------ plotting helpers (matplotlib, print-oriented)
@app.function
def viz_palette():
    return {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4", "green": "#008300",
            "violet": "#6250d6", "red": "#e34948", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"}


@app.function
def set_mpl_style():
    import matplotlib as mpl
    mpl.use("Agg")
    P = viz_palette()
    mpl.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": P["axis"], "axes.labelcolor": P["ink2"], "xtick.color": P["muted"], "ytick.color": P["muted"], "text.color": P["ink"],
        "axes.grid": True, "grid.color": P["grid"], "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 6,
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 110,
    })


@app.function
def condition_style(variant, transfer, floor, imgsz):
    """Fixed colour / marker / hatch per experimental condition (identity never depends on how many conditions are shown)."""
    P = viz_palette()
    if variant == "base" and transfer == "role" and imgsz == 640:
        return {"color": P["blue"], "marker": "o", "hatch": "", "ls": "-"}
    if variant == "graft" and transfer == "role":
        return {"color": P["orange"], "marker": "s", "hatch": "//", "ls": "-"}
    if variant == "naive":
        return {"color": P["aqua"], "marker": "^", "hatch": "\\\\", "ls": "--"}
    if variant == "replace":
        return {"color": P["yellow"], "marker": "D", "hatch": "xx", "ls": "--"}
    if transfer == "none":
        return {"color": P["violet"], "marker": "v", "hatch": "..", "ls": ":"}
    if variant == "base" and imgsz != 640:
        return {"color": P["magenta"], "marker": "P", "hatch": "++", "ls": "-"}
    return {"color": P["muted"], "marker": "x", "hatch": "", "ls": "-"}


@app.function
def save_fig(fig, out_dir, name, wide=False, rect=None):
    """Save PDF + PNG.  In the LNCS template a `wide` figure (designed for 7 in) is shown at 4.8 in, so its text is enlarged to stay legible
    (bar-value labels, marked with gid='val', only slightly: they sit side by side).  `rect` is the tight_layout rectangle to use afterwards."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if wide and PAPER_STYLE["template"] == "lncs":
        from matplotlib.text import Text
        for t in fig.findobj(Text):
            t.set_fontsize(t.get_fontsize() * (1.08 if t.get_gid() == "val" else 1.28))
        try:
            fig.tight_layout(rect=rect) if rect else fig.tight_layout()
        except Exception:
            pass
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=200)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return str(out_dir / f"{name}.pdf")


@app.function
def fig_architecture(out_dir):
    """Schematic of the function-preserving graft (new modules in orange, untouched pretrained path in grey)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    set_mpl_style()
    P = viz_palette()
    lncs = PAPER_STYLE["template"] == "lncs"   # LNCS text block is 4.8 in wide: draw at that width (no down-scaling, so the 7 pt text stays 7 pt)
    fig, ax = plt.subplots(figsize=(4.8, 3.05) if lncs else (7.0, 2.9))
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.7 if lncs else 0, 6)
    ax.axis("off")
    ax.grid(False)
    # column geometry (x, width) in data units; the narrow LNCS drawing uses wider boxes and smaller gaps
    bb, nk, hd, gf = (((0.05, 2.75), (3.5, 2.85), (7.05, 2.45), (10.2, 3.75)) if lncs else ((0.2, 2.2), (3.7, 2.4), (7.0, 2.2), (10.3, 3.6)))
    fs_box, fs_head, fs_new = (6.4, 7.0, 6.2) if lncs else (7.2, 7.5, 6.8)

    def box(x, y, w, h, text, fc, ec=None, tc=None, fs=fs_box, ls="-"):
        ec = ec or fc
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.0, ls=ls)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc or P["ink"])

    def arrow(x1, y1, x2, y2, color=None, lw=1.2, style="-|>", ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=8, lw=lw, color=color or P["ink2"], ls=ls,
                                     connectionstyle=f"arc3,rad={rad}"))

    grey, grey_ec = "#ececea", P["axis"]
    bb_r, nk_r = bb[0] + bb[1], nk[0] + nk[1]
    # backbone column
    ax.text(bb[0] + bb[1] / 2, 5.7, "Backbone\n(pretrained)", fontsize=fs_head, ha="center", va="bottom", color=P["ink2"])
    for i, (y, t) in enumerate([(4.3, "stem / C2f\nstride 4 (C2)"), (3.1, "C3\nstride 8"), (1.9, "C4\nstride 16"), (0.7, "C5+SPPF+PSA\nstride 32")]):
        box(bb[0], y, bb[1], 0.9, t, grey, grey_ec)
        if i < 3:
            arrow(bb[0] + bb[1] / 2, y, bb[0] + bb[1] / 2, y - 0.3)
    # neck column
    ax.text(nk[0] + nk[1] / 2, 5.7, "Neck\n(pretrained)" if lncs else "Neck\n(pretrained, unchanged)", fontsize=fs_head, ha="center", va="bottom", color=P["ink2"])
    box(nk[0], 3.1, nk[1], 0.9, "top-down C2f\nP3 feature", grey, grey_ec)
    box(nk[0], 1.9, nk[1], 0.9, "C2f\nP4 feature", grey, grey_ec)
    box(nk[0], 0.7, nk[1], 0.9, "C2fCIB\nP5 feature", grey, grey_ec)
    arrow(bb_r, 0.9, nk[0], 1.0)
    arrow(bb_r, 2.1, nk[0], 2.2)
    arrow(bb_r, 3.3, nk[0], 3.4)
    arrow(nk[0] + nk[1] / 2, 1.6, nk[0] + nk[1] / 2, 1.9)
    arrow(nk[0] + nk[1] / 2, 2.8, nk[0] + nk[1] / 2, 3.1)
    # heads
    ax.text(hd[0] + hd[1] / 2, 5.7, "Detect heads\n(pretrained)", fontsize=fs_head, ha="center", va="bottom", color=P["ink2"])
    for y, t in [(3.1, "head P3\n(stride 8)"), (1.9, "head P4\n(stride 16)"), (0.7, "head P5\n(stride 32)")]:
        box(hd[0], y, hd[1], 0.9, t, grey, grey_ec)
        arrow(nk_r, y + 0.45, hd[0], y + 0.45)
    # graft branch
    og, og_ec = "#fbe3d8", P["orange"]
    ax.text(gf[0] + gf[1] / 2, 5.7, "Grafted branch (new)", fontsize=fs_head, ha="center", va="bottom", color=P["orange"])
    box(gf[0], 4.3, gf[1], 0.9, "upsample P3 feat.,\nconcat C2, one C2f" if lncs else "upsample P3 feature, concat C2,\nthen one C2f block", og, og_ec, fs=fs_new)
    box(gf[0], 3.1, gf[1], 0.9, "head P2 (stride 4)\none-to-many + one-to-one", og, og_ec, fs=fs_new)
    arrow(gf[0] + gf[1] / 2, 4.3, gf[0] + gf[1] / 2, 4.0, color=P["orange"])
    arrow(nk_r, 3.75, gf[0], 4.6, color=P["orange"], rad=-0.12)
    arrow(bb_r, 4.9, gf[0], 5.05, color=P["orange"], rad=-0.12)
    note = ("P3/P4/P5 outputs are bit-identical to the pretrained model at initialisation:\nthe branch only reads, never writes." if lncs
            else "P3/P4/P5 outputs are bit-identical to the pretrained model at initialisation: the branch only reads, never writes.")
    ax.text(7.0, -0.65 if lncs else 0.05, note, fontsize=7.2 if not lncs else 6.8, ha="center", va="bottom", color=P["ink2"], style="italic")
    return save_fig(fig, out_dir, "fig_architecture")


@app.function
def fig_assignability(assign, out_dir):
    import matplotlib.pyplot as plt
    set_mpl_style()
    P = viz_palette()
    bins = assign["by_size"]
    labels = [b["bin"].replace("inf", "∞") for b in bins]
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    x = np.arange(len(bins))
    w = 0.34
    a = [100 * b["P3-P5|floor=0"] for b in bins]
    c = [100 * b["P2-P5|floor=0"] for b in bins]
    ax.bar(x - w / 2 - 0.02, a, w, color=P["blue"], label="P3–P5 (strides 8/16/32)")
    ax.bar(x + w / 2 + 0.02, c, w, color=P["orange"], hatch="//", edgecolor="white", linewidth=0, label="P2–P5 (strides 4/8/16/32)")
    for xi, v in zip(x - w / 2 - 0.02, a):
        if v > 1:
            ax.text(xi, v + 1, f"{v:.0f}%", ha="center", va="bottom", fontsize=6.8, color=P["ink2"])
    for xi, v in zip(x + w / 2 + 0.02, c):
        if v > 1:
            ax.text(xi, v + 1, f"{v:.0f}%", ha="center", va="bottom", fontsize=6.8, color=P["ink2"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"box size $\sqrt{wh}$ at 640 px input (px)")
    ax.set_ylabel("boxes with no anchor centre inside (%)")
    ax.set_ylim(0, max(a) * 1.18)
    ax.legend(loc="upper right")
    ax.set_title("Classic task-aligned assignment: unassignable boxes", loc="left")
    ax.grid(axis="x", visible=False)
    return save_fig(fig, out_dir, "fig_assignability")


@app.function
def fig_main_comparison(df, out_dir, key_a, key_b, names=("YOLOv10n baseline", "P2-Graft")):
    """(a) overall AP / AP50 and (b) AP50 by object-size bin for two conditions, mean +- std with per-seed dots."""
    import matplotlib.pyplot as plt
    set_mpl_style()
    P = viz_palette()

    def sel(k):
        return df[(df.family == k[0]) & (df.variant == k[1]) & (df.transfer == k[2]) & (df.floor == k[3]) & (df.imgsz == k[4])]
    A, Bd = sel(key_a), sel(key_b)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1, 1.25]})
    cols = [("val_AP50", "AP$_{50}$"), ("val_AP", "AP"), ("val_APs", "AP$_S$")]
    ax = axes[0]
    x = np.arange(len(cols))
    w = 0.34
    for off, d, nm, c, h in ((-w / 2 - 0.02, A, names[0], P["blue"], ""), (w / 2 + 0.02, Bd, names[1], P["orange"], "//")):
        m = [d[c_].mean() for c_, _ in cols]
        s = [d[c_].std(ddof=1) if len(d) > 1 else 0 for c_, _ in cols]
        ax.bar(x + off, m, w, color=c, hatch=h, edgecolor="white", linewidth=0, label=nm)
        ax.errorbar(x + off, m, yerr=s, fmt="none", ecolor=P["ink2"], elinewidth=1.0, capsize=2.5)
        for j, (c_, _) in enumerate(cols):
            ax.scatter(np.full(len(d), x[j] + off), d[c_], s=9, color="white", edgecolor=P["ink2"], linewidth=0.7, zorder=4)
        for j, v in enumerate(m):
            ax.text(x[j] + off, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=6.8, color=P["ink2"], gid="val")
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in cols])
    ax.set_ylabel("VisDrone-val (%)")
    ax.legend(loc="upper right")
    ax.set_title("(a) overall", loc="left")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    bins = [("tiny", "<16$^2$"), ("xsmall", "16$^2$–32$^2$"), ("medium", "32$^2$–96$^2$"), ("large", ">96$^2$")]
    x = np.arange(len(bins))
    for off, d, nm, c, h in ((-w / 2 - 0.02, A, names[0], P["blue"], ""), (w / 2 + 0.02, Bd, names[1], P["orange"], "//")):
        m = [d[f"val_AP50_{b}"].mean() for b, _ in bins]
        s = [d[f"val_AP50_{b}"].std(ddof=1) if len(d) > 1 else 0 for b, _ in bins]
        ax.bar(x + off, m, w, color=c, hatch=h, edgecolor="white", linewidth=0)
        ax.errorbar(x + off, m, yerr=s, fmt="none", ecolor=P["ink2"], elinewidth=1.0, capsize=2.5)
        for j, v in enumerate(m):
            ax.text(x[j] + off, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=6.8, color=P["ink2"], gid="val")
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in bins])
    ax.set_xlabel("object area (px$^2$, original resolution)")
    ax.set_ylabel("AP$_{50}$ (%)")
    ax.set_title("(b) by object size", loc="left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save_fig(fig, out_dir, "fig_main", wide=True)


@app.function
def fig_factorial(df, out_dir, metric="val_AP", ylabel="AP on VisDrone-val (%)"):
    """Interaction plot: architecture (base / graft) x small-target window floor (classic TAL vs library default)."""
    import matplotlib.pyplot as plt
    set_mpl_style()
    P = viz_palette()
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    xs = {0: 0, 16: 1}
    for var, c, mk, nm in (("base", P["blue"], "o", "YOLOv10n baseline"), ("graft", P["orange"], "s", "P2-Graft")):
        pts = []
        for fl in (0, 16):
            d = df[(df.variant == var) & (df.transfer == "role") & (df.floor == fl) & (df.imgsz == 640)]
            if len(d):
                pts.append((xs[fl], d[metric].mean(), d[metric].std(ddof=1) if len(d) > 1 else 0.0, d[metric].to_numpy()))
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, marker=mk, markersize=7, markeredgecolor="white", markeredgewidth=1.5, label=nm)
        for p in pts:
            ax.errorbar(p[0], p[1], yerr=p[2], fmt="none", ecolor=c, elinewidth=1.0, capsize=3)
            ax.scatter(np.full(len(p[3]), p[0]) + 0.03, p[3], s=8, color="white", edgecolor=c, linewidth=0.7, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["classic TAL\n(window floor 0)", "library default\n(floor 16 px)"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right")
    ax.set_title("Stride-4 level × small-target assignment", loc="left")
    ax.grid(axis="x", visible=False)
    return save_fig(fig, out_dir, "fig_factorial")


@app.function
def fig_pareto(df, out_dir, metric="val_AP"):
    """Accuracy vs deploy GFLOPs (left) and vs measured latency (right); one marker per condition (mean of seeds)."""
    import matplotlib.pyplot as plt
    set_mpl_style()
    P = viz_palette()
    from matplotlib.lines import Line2D
    d = df.copy()
    d["cond"] = d.apply(lambda r: f"{r['variant']}|{r['transfer']}|{r['floor']}|{r['imgsz']}", axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    handles = {}
    for ax, xcol, xl in ((axes[0], "gflops", "deploy GFLOPs"), (axes[1], "latency_ms", "latency, batch 1 (ms)")):
        for cond, g in d.groupby("cond"):
            if g[xcol].isna().all():
                continue
            v, t, fl, sz = cond.split("|")
            st = condition_style(v, t, int(fl) if fl != "None" else None, int(sz))
            ax.errorbar(g[xcol].mean(), g[metric].mean(), yerr=g[metric].std(ddof=1) if len(g) > 1 else 0, fmt=st["marker"], color=st["color"],
                        markersize=7, markeredgecolor="white", markeredgewidth=1.2, elinewidth=1.0, capsize=2.5)
            lab = {"base": "baseline", "graft": "P2-Graft", "naive": "naive P2 re-wiring", "replace": "P2 replaces P5"}.get(v, v)
            if t == "none":
                lab += ", random init"
            if int(sz) != 640:
                lab += f" @{sz} px"
            if int(fl) == 0:
                lab += ", classic TAL"
            handles[lab] = Line2D([0], [0], marker=st["marker"], color="none", markerfacecolor=st["color"], markeredgecolor="white", markersize=7, label=lab)
            if (v, t, int(fl), int(sz)) in (("base", "role", 16, 640), ("graft", "role", 16, 640)):  # direct labels only for the headline pair
                ax.annotate("baseline" if v == "base" else "P2-Graft", (g[xcol].mean(), g[metric].mean()), textcoords="offset points", xytext=(5, -9), fontsize=6.8, color=P["ink2"])
        ax.set_xlabel(xl)
        ax.set_ylabel("AP on VisDrone-val (%)" if metric == "val_AP" else metric)
    axes[0].set_title("(a) accuracy vs compute", loc="left")
    axes[1].set_title("(b) accuracy vs latency", loc="left")
    fig.legend(handles=list(handles.values()), loc="lower center", ncol=min(3 if PAPER_STYLE["template"] == "lncs" else 4, max(1, len(handles))), fontsize=7,
               bbox_to_anchor=(0.5, -0.01))
    n_rows = math.ceil(len(handles) / (3 if PAPER_STYLE["template"] == "lncs" else 4))
    # room for the legend rows (the text is enlarged by 1.28 in the LNCS version)
    bottom = (0.05 + 0.062 * n_rows * 1.28) if PAPER_STYLE["template"] == "lncs" else 0.16
    fig.tight_layout(rect=(0, bottom, 1, 1))
    return save_fig(fig, out_dir, "fig_pareto", wide=True, rect=(0, bottom, 1, 1))


@app.function
def fig_tide(df, out_dir, groups=(("base", "role", 16, 640), ("graft", "role", 16, 640)), names=("YOLOv10n baseline", "P2-Graft")):
    import matplotlib.pyplot as plt
    set_mpl_style()
    P = viz_palette()
    errs = ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"]
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    x = np.arange(len(errs))
    w = 0.34
    for off, g, nm, c, h in ((-w / 2 - 0.02, groups[0], names[0], P["blue"], ""), (w / 2 + 0.02, groups[1], names[1], P["orange"], "//")):
        d = df[(df.variant == g[0]) & (df.transfer == g[1]) & (df.floor == g[2]) & (df.imgsz == g[3])]
        if not len(d) or "tide_Cls" not in d:
            continue
        m = [d[f"tide_{e}"].mean() for e in errs]
        s = [d[f"tide_{e}"].std(ddof=1) if len(d) > 1 else 0 for e in errs]
        ax.bar(x + off, m, w, color=c, hatch=h, edgecolor="white", linewidth=0, label=nm)
        ax.errorbar(x + off, m, yerr=s, fmt="none", ecolor=P["ink2"], elinewidth=1.0, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(errs)
    ax.set_ylabel("TIDE $\\Delta$AP$_{50}$ (points)")
    ax.legend(loc="upper left")
    ax.set_title("Error decomposition (TIDE), VisDrone-val", loc="left")
    ax.grid(axis="x", visible=False)
    return save_fig(fig, out_dir, "fig_tide")


@app.function
def fig_perclass(df, out_dir, key_a=("base", "role", 16, 640), key_b=("graft", "role", 16, 640)):
    import matplotlib.pyplot as plt
    set_mpl_style()
    P = viz_palette()

    def get(k):
        d = df[(df.variant == k[0]) & (df.transfer == k[1]) & (df.floor == k[2]) & (df.imgsz == k[3])]
        return np.array([np.asarray(v, dtype=float) for v in d["val_perclass_AP50"]]) if len(d) else None
    A, B = get(key_a), get(key_b)
    if A is None or B is None:
        return None
    delta_seed = []
    n = min(len(A), len(B))
    delta = B.mean(0) - A.mean(0)
    order = np.argsort(delta)
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    y = np.arange(len(order))
    ax.barh(y, delta[order], height=0.55, color=[P["orange"] if v >= 0 else P["blue"] for v in delta[order]])
    ax.axvline(0, color=P["ink2"], lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([CLASS_NAMES[i] for i in order])
    ax.set_xlabel("$\\Delta$AP$_{50}$ (graft − baseline), points")
    span = max(1.0, float(np.abs(delta).max()))
    ax.set_xlim(-0.35 * span - 0.4, 1.3 * span + 0.2)
    for yi, v in zip(y, delta[order]):
        ax.text(v + (0.04 * span if v >= 0 else -0.04 * span), yi, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=6.8, color=P["ink2"])
    ax.set_title("Per-class change", loc="left")
    ax.grid(axis="y", visible=False)
    return save_fig(fig, out_dir, "fig_perclass")


@app.function
def fig_curves(work, out_dir, run_ids, labels, styles):
    """Training-loss curves (sum of box+cls+dfl) from each run's results.csv."""
    import matplotlib.pyplot as plt
    import pandas as pd
    set_mpl_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    drawn = False
    for rid, lab, st in zip(run_ids, labels, styles):
        p = Path(work) / "runs" / rid / "results.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d.columns = [c.strip() for c in d.columns]
        cols = [c for c in d.columns if c.startswith("train/") and c.endswith("loss")]
        if not cols:
            continue
        y = d[cols].sum(axis=1)
        ax.plot(d["epoch"], y, color=st["color"], ls=st["ls"], label=lab)
        drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (box+cls+dfl, one-to-one)")
    ax.legend(loc="upper right")
    ax.set_title("Optimisation from the pretrained start", loc="left")
    return save_fig(fig, out_dir, "fig_curves")


@app.function
def fig_qualitative(work, out_dir, run_a, run_b, split="val", size=640, n=3, conf=0.3, seed=0):
    """Rows = images where the second model recovers the most extra confident detections; columns = GT / model A / model B."""
    import cv2
    import matplotlib.pyplot as plt
    set_mpl_style()
    work = Path(work)
    gt = jload(work / "data" / f"gt_{split_dir_name(split)}.json")
    pa = load_preds(run_pred_path(work, run_a, split, size))
    pb = load_preds(run_pred_path(work, run_b, split, size))
    n_img = len(gt["images"])
    ca = np.bincount(pa["img"][pa["score"] >= conf], minlength=n_img)
    cb = np.bincount(pb["img"][pb["score"] >= conf], minlength=n_img)
    ngt = np.array([im["n_gt"] for im in gt["images"]])
    score = (cb - ca) * (ngt >= 25) * (ngt <= 120)
    idx = np.argsort(-score)[:n]
    gts = {}
    for a in gt["annotations"]:
        gts.setdefault(a["image_id"], []).append(a)
    P = viz_palette()
    fig, axes = plt.subplots(n, 3, figsize=(7.0, 2.35 * n))
    axes = np.atleast_2d(axes)
    for r, k in enumerate(idx):
        im = gt["images"][int(k)]
        img = cv2.imread(str(work / "data" / split_dir_name(split, size) / "images" / f"{im['stem']}.jpg"))[:, :, ::-1]
        sx, sy = img.shape[1] / im["width"], img.shape[0] / im["height"]
        # crop: window with the most ground-truth centres
        ws, hs = img.shape[1] // 2, img.shape[0] // 2
        cxs = np.array([(a["bbox"][0] + a["bbox"][2] / 2) * sx for a in gts.get(int(k), [])])
        cys = np.array([(a["bbox"][1] + a["bbox"][3] / 2) * sy for a in gts.get(int(k), [])])
        best = (0, 0, -1)
        for x0 in range(0, img.shape[1] - ws + 1, max(1, ws // 6)):
            for y0 in range(0, img.shape[0] - hs + 1, max(1, hs // 6)):
                cnt = int(((cxs >= x0) & (cxs < x0 + ws) & (cys >= y0) & (cys < y0 + hs)).sum())
                if cnt > best[2]:
                    best = (x0, y0, cnt)
        x0, y0, _ = best
        for c, (title, kind) in enumerate((("ground truth", "gt"), ("YOLOv10n baseline", "a"), ("P2-Graft", "b"))):
            ax = axes[r, c]
            ax.imshow(img[y0:y0 + hs, x0:x0 + ws])
            ax.axis("off")
            ax.grid(False)
            if kind == "gt":
                boxes = [(a["bbox"][0] * sx, a["bbox"][1] * sy, a["bbox"][2] * sx, a["bbox"][3] * sy) for a in gts.get(int(k), [])]
                col = "#ffffff"
            else:
                pr = pa if kind == "a" else pb
                sel = (pr["img"] == k) & (pr["score"] >= conf)
                boxes = [((b[0]) * sx, b[1] * sy, (b[2] - b[0]) * sx, (b[3] - b[1]) * sy) for b in pr["xyxy"][sel]]
                col = P["blue"] if kind == "a" else P["orange"]
            nb = 0
            for (bx, by, bw, bh) in boxes:
                if x0 <= bx + bw / 2 < x0 + ws and y0 <= by + bh / 2 < y0 + hs:
                    ax.add_patch(plt.Rectangle((bx - x0, by - y0), bw, bh, fill=False, ec=col, lw=1.0))
                    nb += 1
            ax.set_title(f"{title} ({nb})" if r == 0 else f"({nb})", fontsize=7.5, loc="left")
    fig.tight_layout()
    return save_fig(fig, out_dir, "fig_qualitative", wide=True)


# ------------------------------------------------------------------ analysis driver
@app.function
def canonical_keys(family="v10", iso_size=896):
    return {
        "base": (family, "base", "role", 16, 640), "graft": (family, "graft", "role", 16, 640),
        "base_f0": (family, "base", "role", 0, 640), "graft_f0": (family, "graft", "role", 0, 640),
        "naive": (family, "naive", "key", 16, 640), "replace": (family, "replace", "role", 16, 640),
        "graft_scratch": (family, "graft", "none", 16, 640), "base_scratch": (family, "base", "none", 16, 640),
        "iso": (family, "base", "role", 16, iso_size),
    }


@app.function
def runs_for_key(df, key):
    if df.empty:
        return []
    sel = df[(df.family == key[0]) & (df.variant == key[1]) & (df.transfer == key[2]) & (df.floor == key[3]) & (df.imgsz == key[4])]
    return [(int(s), r) for s, r in zip(sel.seed, sel.run_id)]


@app.function
def bootstrap_runs(work, items, B=200, seed=0, split="val", log_path=None):
    """items: {run_id: test_size} -> {run_id: array (B+1, 2)} of [AP, AP50] under shared image resamples (row 0 = the original sample).
    Sharing the resamples makes differences between any two runs (or seed-averages of runs) paired."""
    work = Path(work)
    gt = work / "data" / f"gt_{split_dir_name(split)}.json"
    rng = np.random.default_rng(seed)
    out: Dict[str, Any] = {}
    W = None
    for rid, size in items.items():
        t0 = time.time()
        prep = boot_prepare(gt, load_preds(run_pred_path(work, rid, split, size)))
        if W is None:
            n = prep["n_img"]
            W = np.vstack([np.ones(n)] + [np.bincount(rng.integers(0, n, size=n), minlength=n).astype(np.float64) for _ in range(B)])
        out[rid] = np.array([boot_ap(prep, W[b]) for b in range(B + 1)])
        log(f"bootstrap {rid}: {time.time() - t0:.0f}s", log_path)
    return out


@app.function
def ci_from_boot(a, b, idx=0):
    """Difference b - a: point estimate, 95% percentile CI, fraction of resamples > 0.  a, b: arrays (B+1, 2)."""
    d = b[1:, idx] - a[1:, idx]
    return {"delta": float(b[0, idx] - a[0, idx]), "ci95": (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))),
            "frac_positive": float(np.mean(d > 0))}


@app.function
def analyze(work, B=200, do_bootstrap=True, iso_size=896, family="v10", log_path=None):
    """Collect every finished run and compute all statistics used in the paper."""
    import pandas as pd
    work = Path(work)
    df = collect_results(work)
    A: Dict[str, Any] = {"n_runs": int(len(df)), "df": df, "keys": canonical_keys(family, iso_size)}
    if df.empty:
        return A
    metrics = ["val_AP", "val_AP50", "val_AP75", "val_APs", "val_APm", "val_APl", "val_AP_tiny", "val_AP_xsmall", "val_AP50_tiny",
               "val_AP50_xsmall", "val_AP50_medium", "val_AP50_large", "val_ARmax", "test_AP", "test_AP50", "test_APs", "val_AP_ign",
               "val_AP50_ign", "val_AP_o2m", "val_AP50_o2m", "val_ultra_AP50", "val_ultra_AP", "val_AP50_conf05", "val_AP50_conf25", "val_AP_conf25",
               "val_AP50_maxdet100", "val_AP_maxdet100", "val_AP_at800", "val_AP_at960", "val_APs_at800", "val_APs_at960", "val_AP50_at800", "val_AP50_at960", "params_deploy_M", "gflops", "latency_ms", "train_h", "frac_transferred",
               "frac_aligned"] + [c for c in df.columns if c.startswith(("tide_", "ext_"))]
    A["agg"] = aggregate_groups(df, [m for m in metrics if m in df.columns])
    A["external"] = collect_external(work)
    A["epochs"] = int(df["epochs"].mode().iloc[0])
    K = A["keys"]
    present = {n: runs_for_key(df, k) for n, k in K.items()}
    A["present"] = {n: [s for s, _ in v] for n, v in present.items()}
    pairs = {"graft_vs_base": ("base", "graft"), "base_f0_vs_f16": ("base_f0", "base"), "graft_f0_vs_f16": ("graft_f0", "graft"),
             "naive_vs_graft": ("naive", "graft"), "replace_vs_graft": ("replace", "graft"), "scratch_vs_pretrained_graft": ("graft_scratch", "graft"),
             "scratch_vs_pretrained_base": ("base_scratch", "base"), "iso_vs_graft": ("iso", "graft"), "iso_vs_base": ("base", "iso"),
             "graft_f0_vs_base_f0": ("base_f0", "graft_f0")}
    paired = {}
    for name, (a, b) in pairs.items():
        if present.get(a) and present.get(b):
            paired[name] = {m: paired_comparison(df, K[a], K[b], m) for m in ("val_AP", "val_AP50", "val_APs", "val_AP_tiny", "val_AP50_tiny", "test_AP", "test_AP50")
                            if m in df.columns}
    A["paired"] = paired
    # interaction contrast: (graft - base | floor 16) - (graft - base | classic TAL)
    contrast = {}
    if all(present.get(n) for n in ("base", "graft", "base_f0", "graft_f0")):
        for m in ("val_AP", "val_AP50", "val_APs"):
            seeds = sorted(set(s for s, _ in present["base"]) & set(s for s, _ in present["graft"]) & set(s for s, _ in present["base_f0"]) & set(s for s, _ in present["graft_f0"]))
            g = lambda n: {s: float(df[df.run_id == r][m].iloc[0]) for s, r in present[n]}
            b16, g16, b0, g0 = g("base"), g("graft"), g("base_f0"), g("graft_f0")
            c = np.array([(g16[s] - b16[s]) - (g0[s] - b0[s]) for s in seeds])
            contrast[m] = {"per_seed": c.tolist(), "mean": float(c.mean()), "std": float(c.std(ddof=1)) if len(c) > 1 else float("nan"), "n": len(c),
                           "gain_with_floor": float(np.mean([g16[s] - b16[s] for s in seeds])), "gain_classic": float(np.mean([g0[s] - b0[s] for s in seeds]))}
    A["interaction"] = contrast
    # image-level bootstrap on VisDrone-val
    if do_bootstrap:
        need: Dict[str, int] = {}
        for n in ("base", "graft", "base_f0", "graft_f0", "naive", "replace", "iso"):
            sz = iso_size if n == "iso" else 640
            for _s, rid in present.get(n, []):
                if run_pred_path(work, rid, "val", sz).exists():
                    need[rid] = sz
        if need and present.get("base") and present.get("graft"):
            try:
                boots = bootstrap_runs(work, need, B=B, log_path=log_path)
                seeds_of = lambda n: {s for s, rid in present.get(n, []) if rid in boots}

                def seed_avg(n, seeds):
                    arrs = [boots[rid] for s, rid in present[n] if s in seeds and rid in boots]
                    return np.mean(arrs, axis=0)
                A["bootstrap"] = {}
                for name, (a, b) in pairs.items():
                    common = seeds_of(a) & seeds_of(b)
                    if common:
                        xa, xb = seed_avg(a, common), seed_avg(b, common)
                        A["bootstrap"][name] = {"AP": ci_from_boot(xa, xb, 0), "AP50": ci_from_boot(xa, xb, 1), "seeds": sorted(common)}
                common4 = seeds_of("base") & seeds_of("graft") & seeds_of("base_f0") & seeds_of("graft_f0")
                if common4:
                    b16, g16, b0, g0 = (seed_avg(n, common4) for n in ("base", "graft", "base_f0", "graft_f0"))
                    d = (g16[1:] - b16[1:]) - (g0[1:] - b0[1:])
                    d0 = (g16[0] - b16[0]) - (g0[0] - b0[0])
                    A["bootstrap"]["interaction"] = {"AP": {"delta": float(d0[0]), "ci95": (float(np.percentile(d[:, 0], 2.5)), float(np.percentile(d[:, 0], 97.5)))},
                                                     "AP50": {"delta": float(d0[1]), "ci95": (float(np.percentile(d[:, 1], 2.5)), float(np.percentile(d[:, 1], 97.5)))},
                                                     "seeds": sorted(common4)}
                A["bootstrap_B"] = B
            except Exception as e:
                A["bootstrap_error"] = repr(e)
                log(f"bootstrap failed: {e!r}\n{traceback.format_exc()}", log_path)
    return A


# =====================================================================================================
# SECTION H -- paper generation (LaTeX with numbers and verdicts computed from the saved results)
# =====================================================================================================
@app.function
def tex_escape(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    return "".join(rep.get(c, c) for c in str(s))


@app.function
def isnum(x):
    try:
        return x is not None and not (isinstance(x, float) and math.isnan(x))
    except Exception:
        return False


@app.function
def f1(x, d=1):
    return f"{x:.{d}f}" if isnum(x) else "--"


@app.function
def sgn(x, d=1):
    """Signed number for running LaTeX text: a real minus sign (math mode), and never '-0.0' after rounding."""
    if not isnum(x):
        return "--"
    r = round(float(x), d)
    if r == 0:
        return f"${0:.{d}f}$"
    return f"${'+' if r > 0 else '-'}{abs(r):.{d}f}$"


@app.function
def pmtex(m, s, d=1):
    if not isnum(m):
        return "--"
    return f"${m:.{d}f}\\pm{s:.{d}f}$" if isnum(s) else f"${m:.{d}f}$"


@app.function
def cond_row(A, name):
    """Aggregated row (dict) for a canonical condition, or None."""
    agg = A.get("agg")
    if agg is None or len(agg) == 0:
        return None
    k = "|".join(str(x) for x in A["keys"][name])
    sel = agg[agg.key == k]
    return None if sel.empty else sel.iloc[0].to_dict()


@app.function
def verdict_from(delta, ci=None, per_seed=None):
    """('positive'|'negative'|'inconclusive', explanation) from an interval (preferred) or per-seed deltas."""
    n = len(per_seed) if per_seed else 0
    if ci is not None:
        lo, hi = ci
        if lo > 0 or hi < 0:
            sign = lo > 0
            if n > 1 and not all((d > 0) == sign for d in per_seed):
                return "inconclusive", f"image-bootstrap interval excludes zero but the {n} paired seeds disagree in sign"
            how = f"same sign in all {n} paired seeds" if n > 1 else "single seed: training variance not captured"
            return ("positive" if sign else "negative"), f"95% image-bootstrap interval excludes zero; {how}"
        return "inconclusive", f"95% image-bootstrap interval includes zero ({n} paired seed{'s' if n != 1 else ''})"
    if n > 1 and all(d > 0 for d in per_seed):
        return "positive", f"same sign in all {n} paired seeds (no bootstrap interval available)"
    if n > 1 and all(d < 0 for d in per_seed):
        return "negative", f"same sign in all {n} paired seeds (no bootstrap interval available)"
    return "inconclusive", f"insufficient evidence ({n} paired seed{'s' if n != 1 else ''}, no interval)"


@app.function
def build_claims(A):
    """Turn the analysis dictionary into sentences and hypothesis verdicts (nothing is asserted that the data do not show)."""
    C: Dict[str, Any] = {"verdicts": {}, "sentences": {}, "short": {}}  # "short": compact versions of the sentences for the abstract
    P, BS, IA = A.get("paired", {}), A.get("bootstrap", {}), A.get("interaction", {})
    base, graft = cond_row(A, "base"), cond_row(A, "graft")
    # ---- H1: graft vs base
    if "graft_vs_base" in P:
        p = P["graft_vs_base"]
        b = BS.get("graft_vs_base", {})
        ci = b.get("AP", {}).get("ci95")
        v, why = verdict_from(p["val_AP"]["mean_delta"], ci, p["val_AP"]["per_seed_delta"])
        C["verdicts"]["H1"] = (v, why)
        verb = {"positive": "improves", "negative": "reduces", "inconclusive": "does not reliably change"}[v]
        d_ap = p["val_AP"]["mean_delta"]
        if v == "inconclusive":
            s = f"P2-Graft does not reliably change AP ({sgn(d_ap)} points; {f1(base['val_AP'])} $\\rightarrow$ {f1(graft['val_AP'])})"
        else:
            s = f"P2-Graft {verb} AP by {abs(d_ap):.1f} points ({f1(base['val_AP'])} $\\rightarrow$ {f1(graft['val_AP'])})"
        if ci:
            s += f", 95\\% image-bootstrap interval of the difference [{sgn(ci[0])}, {sgn(ci[1])}]"
        s += f"; AP$_{{50}}$ changes by {sgn(p['val_AP50']['mean_delta'])} points"
        if "val_AP_tiny" in p and p["val_AP_tiny"]:
            s += f" and AP on objects below $16^2$ px by {sgn(p['val_AP_tiny']['mean_delta'])}"
        sd = p["val_AP"]["per_seed_delta"]
        if len(sd) > 1:
            same = all(x > 0 for x in sd) or all(x < 0 for x in sd)
            s += f". The per-seed differences are {', '.join(sgn(x) for x in sd)} (the sign is {'identical' if same else 'not identical'} across the {len(sd)} paired seeds)"
        C["sentences"]["main"] = s + "."
        short = (f"P2-Graft does not reliably change AP ({sgn(d_ap)}; {f1(base['val_AP'])}$\\rightarrow${f1(graft['val_AP'])})" if v == "inconclusive"
                 else f"P2-Graft {verb} AP by {abs(d_ap):.1f} points ({f1(base['val_AP'])}$\\rightarrow${f1(graft['val_AP'])})")
        if ci:
            short += f", 95\\% bootstrap interval [{sgn(ci[0])}, {sgn(ci[1])}]"
        C["short"]["main"] = short
    # ---- cost
    if base and graft and isnum(base.get("gflops")) and isnum(graft.get("gflops")):
        s = f"The extra level costs {f1(graft['gflops'])} vs {f1(base['gflops'])} deploy GFLOPs ($\\times${graft['gflops'] / base['gflops']:.2f})"
        if isnum(base.get("latency_ms")) and isnum(graft.get("latency_ms")):
            s += f" and {f1(graft['latency_ms'], 2)} vs {f1(base['latency_ms'], 2)}\\,ms median latency"
        s += f", with {f1(graft['params_deploy_M'], 2)} vs {f1(base['params_deploy_M'], 2)}\\,M deploy parameters."
        C["sentences"]["cost"] = s
    # ---- H3: interaction
    if "val_AP" in IA:
        i = IA["val_AP"]
        ci = BS.get("interaction", {}).get("AP", {}).get("ci95")
        v, why = verdict_from(-i["mean"], None if ci is None else (-ci[1], -ci[0]), [-x for x in i["per_seed"]])  # H3 predicts a NEGATIVE contrast
        C["verdicts"]["H3"] = (v, why)
        s = (f"Under classic task-aligned assignment the stride-4 level adds {sgn(i['gain_classic'])} AP; with the library's small-target window floor "
             f"it adds {sgn(i['gain_with_floor'])} AP, i.e. an interaction contrast of {sgn(i['mean'], 2)}")
        if ci:
            s += f" (95\\% bootstrap interval [{f1(ci[0], 2)}, {f1(ci[1], 2)}])"
        C["sentences"]["interaction"] = s + "."
        C["short"]["interaction"] = (f"the gain of the stride-4 level is {sgn(i['gain_classic'])} AP under classic assignment and "
                                     f"{sgn(i['gain_with_floor'])} AP with the window floor")
    # ---- H2: graft vs naive (usually single seed)
    if "naive_vs_graft" in P:
        p = P["naive_vs_graft"]["val_AP"]
        ci = BS.get("naive_vs_graft", {}).get("AP", {}).get("ci95")
        v, why = verdict_from(p["mean_delta"], ci, p["per_seed_delta"])
        C["verdicts"]["H2"] = (v, why)
        nv = cond_row(A, "naive")
        frac = A.get("audit_naive_frac")
        inh = f"; key matching inherits {100 * frac:.0f}\\% of the inheritable pretrained parameters" if isnum(frac) else ""
        diff = -p["mean_delta"]
        C["sentences"]["naive"] = (f"Re-wiring the neck instead of grafting (key-matched transfer{inh}) gives {f1(nv['val_AP'])} AP, "
                                   f"{abs(diff):.1f} points {'below' if diff < 0 else 'above'} P2-Graft ({p['n']} seed{'s' if p['n'] > 1 else ''}).")
        C["short"]["naive"] = f"naive re-wiring is {abs(diff):.1f} points {'below' if diff < 0 else 'above'} P2-Graft ({p['n']} seed{'s' if p['n'] > 1 else ''})"
    # ---- H4: compute-matched
    if "iso_vs_graft" in P:
        p = P["iso_vs_graft"]["val_AP"]
        ci = BS.get("iso_vs_graft", {}).get("AP", {}).get("ci95")
        v, why = verdict_from(p["mean_delta"], ci, p["per_seed_delta"])
        C["verdicts"]["H4"] = (v, why)
        iso = cond_row(A, "iso")
        diff = -p["mean_delta"]
        C["sentences"]["iso"] = (f"A baseline trained and tested at higher resolution ({iso['imgsz']} px, {f1(iso['gflops'])} GFLOPs) reaches {f1(iso['val_AP'])} AP, "
                                 f"{abs(diff):.1f} points {'below' if diff < 0 else 'above'} P2-Graft at {f1(graft['gflops'])} GFLOPs.")
        C["short"]["iso"] = f"a baseline at higher resolution and matched GFLOPs is {abs(diff):.1f} points {'below' if diff < 0 else 'above'} P2-Graft ({p['n']} seed{'s' if p['n'] > 1 else ''})"
    return C


@app.function
def latex_table(header, rows, caption, label, colspec=None, notes=None, star=False, small=True, tiny=False):
    """Booktabs table.  `rows` is a list of lists of already-formatted LaTeX strings; None entries become horizontal rules.
    tiny: use the smallest size (long reference tables) in the LNCS template."""
    ncol = len(header)
    colspec = colspec or ("l" + "r" * (ncol - 1))
    lncs = PAPER_STYLE["template"] == "lncs"
    env = "table*" if (star and not lncs) else "table"
    size = ("\\scriptsize" if tiny else "\\footnotesize") if lncs else ("\\small" if small else "")
    size = size if (small or tiny) else ""
    # LNCS has a single narrow column: scale a table down only if it is wider than the text block
    open_fit, close_fit = ("\\begin{adjustbox}{max width=\\linewidth}", "\\end{adjustbox}") if lncs else ("", "")
    out = [f"\\begin{{{env}}}[{'!htbp' if lncs else 't'}]", "\\centering", size, f"\\caption{{{caption}}}", f"\\label{{{label}}}", open_fit,
           f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if r is None:
            out.append("\\midrule")
        else:
            out.append(" & ".join(r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", close_fit]
    if notes:
        out.append(f"\\par\\vspace{{2pt}}{{\\footnotesize {notes}}}")
    out.append(f"\\end{{{env}}}")
    return "\n".join(x for x in out if x != "")


@app.function
def table_main(A):
    names = [("base", "YOLOv10n baseline"), ("graft", "P2-Graft (ours)"), ("naive", "naive P2 re-wiring"), ("replace", "P2 replaces P5"),
             ("graft_scratch", "P2-Graft, random init"), ("base_scratch", "baseline, random init"), ("iso", None)]
    rows = []
    for key, nm in names:
        r = cond_row(A, key)
        if not r:
            continue
        if key == "iso":
            nm = f"baseline @ {r['imgsz']} px"
        rows.append([tex_escape(nm), str(r["n"]), f1(r.get("params_deploy_M"), 2), f1(r.get("gflops")), f1(r.get("latency_ms"), 2),
                     pmtex(r.get("val_AP50"), r.get("val_AP50_std")), pmtex(r.get("val_AP"), r.get("val_AP_std")), pmtex(r.get("val_APs"), r.get("val_APs_std")),
                     pmtex(r.get("val_AP_tiny"), r.get("val_AP_tiny_std")), pmtex(r.get("test_AP"), r.get("test_AP_std"))])
    return latex_table(["Model", "$n$", "Par.\\,(M)", "GFLOPs", "ms", "AP$_{50}$", "AP", "AP$_S$", "AP$_{<16^2}$", "AP$^{\\mathrm{test}}$"], rows,
                       f"Main results on VisDrone-DET2019 validation (and test-dev for the last column), mean $\\pm$ std over $n$ seeds, {A['epochs']}-epoch fine-tuning from the COCO checkpoint. "
                       "Parameters and GFLOPs are for the deploy graph (fused, one-to-one head only) at the evaluated resolution; latency is the median batch-1 FP16 forward pass on the GPU of Section~\\ref{sec:setup}.",
                       "tab:main", colspec="lrrrrrrrrr", star=True)


@app.function
def table_factorial(A):
    rows = []
    for floor, lab in ((16, "library default (floor 16\\,px)"), (0, "classic TAL (floor 0)")):
        b = cond_row(A, "base" if floor == 16 else "base_f0")
        g = cond_row(A, "graft" if floor == 16 else "graft_f0")
        if not b or not g:
            continue
        rows.append([lab, pmtex(b["val_AP"], b["val_AP_std"]), pmtex(g["val_AP"], g["val_AP_std"]), f"${g['val_AP'] - b['val_AP']:+.1f}$",
                     pmtex(b["val_APs"], b["val_APs_std"]), pmtex(g["val_APs"], g["val_APs_std"]), f"${g['val_APs'] - b['val_APs']:+.1f}$"])
    if not rows:
        return ""
    return latex_table(["Assignment", "AP base", "AP graft", "$\\Delta$", "AP$_S$ base", "AP$_S$ graft", "$\\Delta$"], rows,
                       "Stride-4 level $\\times$ small-target window floor of the task-aligned assigner (VisDrone-val, mean $\\pm$ std over seeds).",
                       "tab:factorial", colspec="lrrrrrr", star=True)


@app.function
def table_transfer(audit):
    if not audit:
        return ""
    names = {("base", "role"): "baseline (reference)", ("graft", "role"): "P2-Graft, role-based map", ("graft", "key"): "P2-Graft, key matching",
             ("naive", "key"): "naive P2, key matching", ("replace", "role"): "P2 replaces P5, role-based map", ("replace", "key"): "P2 replaces P5, key matching"}
    rows = []
    for r in audit["rows"]:
        rows.append([names.get((r["variant"], r["mode"]), f"{r['variant']}/{r['mode']}"), f"{100 * r['frac_of_inheritable']:.1f}", f"{100 * r['frac_aligned_of_inheritable']:.1f}",
                     f1(r["params_train_graph_M"], 2), f1(r["params_deploy_fused_M"], 2), f1(r["gflops_deploy"])])
    return latex_table(["Variant / transfer rule", "copied (\\%)", "role-aligned (\\%)", "params, train (M)", "params, deploy (M)", "GFLOPs"], rows,
                       "Pretrained parameters inherited by each variant ($100\\%$ = everything a 10-class baseline can inherit from the 80-class COCO checkpoint). "
                       "`Key matching' is Ultralytics' default (same tensor name and shape); a copied tensor is role-aligned only if the whole sub-graph feeding that layer is unchanged.",
                       "tab:transfer", colspec="lrrrrr", star=True)


@app.function
def table_assign(assign):
    if not assign:
        return ""
    rows = []
    for cfg, lab in (("P3-P5", "P3--P5 (8/16/32)"), ("P2-P5", "P2--P5 (4/8/16/32)")):
        rows.append([lab] + [f"{100 * assign[f'{cfg}|floor={f}']['frac_unassigned']:.1f}" for f in (0, 8, 16)] +
                    [f"{assign[f'{cfg}|floor=0']['mean_candidates']:.1f}", f"{assign[f'{cfg}|floor=16']['mean_candidates']:.1f}"])
    return latex_table(["Levels", "floor 0 (\\%)", "floor 8 (\\%)", "floor 16 (\\%)", "cand./box (floor 0)", "cand./box (floor 16)"], rows,
                       f"Share of VisDrone-train boxes (343k, long side resized to {assign['imgsz']} px) with no anchor centre inside the box, i.e. unable to become a positive of a task-aligned assigner, "
                       "for different pyramids and window floors. Exact counts at the true positions; average of the true placement and four random sub-grid shifts.",
                       "tab:assign", colspec="lrrrrr", star=True)


@app.function
def table_tide(A):
    b, g = cond_row(A, "base"), cond_row(A, "graft")
    if not b or not g or "tide_Cls" not in b:
        return ""
    rows = []
    for e in ("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss", "FalsePos", "FalseNeg"):
        if f"tide_{e}" in b and f"tide_{e}" in g:
            rows.append([e, pmtex(b[f"tide_{e}"], b.get(f"tide_{e}_std"), 2), pmtex(g[f"tide_{e}"], g.get(f"tide_{e}_std"), 2), f"${g[f'tide_{e}'] - b[f'tide_{e}']:+.2f}$"])
    return latex_table(["Error type", "baseline", "P2-Graft", "$\\Delta$"], rows,
                       "TIDE $\\Delta$AP$_{50}$ (points) computed from the saved VisDrone-val predictions (500 detections per image, boxes as $[x,y,w,h]$). Lower is better.",
                       "tab:tide", colspec="lrrr")


@app.function
def table_testsize(A):
    """Models trained at 640 px evaluated at larger input sizes without retraining (extra sizes exist for the first seed only)."""
    rows = []
    for key, nm in (("base", "YOLOv10n baseline"), ("graft", "P2-Graft")):
        r = cond_row(A, key)
        if not r or not isnum(r.get("val_AP_at800")):
            continue
        trip = lambda a, b, c: " / ".join(f1(x) for x in (a, b, c))
        rows.append([nm, trip(r["val_AP"], r.get("val_AP_at800"), r.get("val_AP_at960")), trip(r["val_APs"], r.get("val_APs_at800"), r.get("val_APs_at960")),
                     f"{f1(r.get('gflops'))} / {f1(r['gflops'] * (800 / 640) ** 2) if isnum(r.get('gflops')) else '--'} / {f1(r['gflops'] * (960 / 640) ** 2) if isnum(r.get('gflops')) else '--'}"])
    if not rows:
        return ""
    return latex_table(["Model (trained at 640\\,px)", "AP @ 640 / 800 / 960", "AP$_S$ @ 640 / 800 / 960", "GFLOPs @ 640 / 800 / 960"], rows,
                       "Test-time input size without retraining (extra sizes: first seed only; GFLOPs scale with the pixel count).", "tab:testsize", colspec="lccc", star=True)


@app.function
def larger_input_sentence(A):
    """The content of table_testsize as running text (used where the page budget does not allow a 2-row table)."""
    parts = []
    for key, nm in (("base", "the baseline"), ("graft", "P2-Graft")):
        r = cond_row(A, key)
        if not r or not isnum(r.get("val_AP_at800")):
            continue
        g = r.get("gflops")
        gf = (" / ".join(f1(g * (s / 640.0) ** 2) for s in (640, 800, 960)) + " GFLOPs") if isnum(g) else ""
        parts.append(f"{nm} reaches AP {f1(r['val_AP'])} / {f1(r.get('val_AP_at800'))} / {f1(r.get('val_AP_at960'))} and AP$_S$ {f1(r['val_APs'])} / "
                     f"{f1(r.get('val_APs_at800'))} / {f1(r.get('val_APs_at960'))}" + (f" at {gf}" if gf else ""))
    if not parts:
        return ""
    return "Evaluated without retraining at 640 / 800 / 960\\,px (first seed only), " + " and ".join(parts) + "."


@app.function
def table_perseed(A):
    df = A["df"]
    rows = []
    for _, r in df.sort_values(["variant", "transfer", "floor", "imgsz", "seed"]).iterrows():
        rows.append([tex_escape(r["run_id"]), f1(r.get("val_AP50")), f1(r.get("val_AP")), f1(r.get("val_APs")), f1(r.get("test_AP50")), f1(r.get("test_AP")),
                     f1(r.get("train_h"), 2)])
    if PAPER_STYLE["template"] == "lncs":  # narrow page: two half-tables side by side (same content, half the height)
        half = (len(rows) + 1) // 2
        left, right = rows[:half], rows[half:] + [[""] * 7] * (half - len(rows[half:]))
        paired = [[l[0], l[1], l[2], l[5], "~", r[0], r[1], r[2], r[5]] if l[0] or r[0] else None for l, r in zip(left, right)]
        return latex_table(["Run", "val AP$_{50}$", "val AP", "test AP", "", "Run", "val AP$_{50}$", "val AP", "test AP"],
                           [p for p in paired if p is not None],
                           "Every finished run (one row per seed; the full table with all metrics is \\texttt{all\\_runs.csv} in the archive).", "tab:perseed",
                           colspec="lrrrcrrrr", small=True, star=True, tiny=True)
    return latex_table(["Run", "val AP$_{50}$", "val AP", "val AP$_S$", "test AP$_{50}$", "test AP", "train (h)"], rows,
                       "Every finished run (one row per seed).", "tab:perseed", colspec="lrrrrrr", small=True, star=True, tiny=True)


@app.function
def paper_preamble(template="article"):
    if template == "lncs":
        return r"""\documentclass[runningheads]{llncs}
% Springer LNCS (the format of ICPR / ICPRv).  llncs.cls and splncs04.bst are Springer's files and are not part of this bundle:
% take them from Springer's LNCS author kit or from Overleaf's "Springer LNCS" template and keep them next to this file.
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{adjustbox}
\usepackage{xcolor}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\topfraction}{0.9}\renewcommand{\bottomfraction}{0.8}\renewcommand{\textfraction}{0.07}\renewcommand{\floatpagefraction}{0.75}
\setlength{\textfloatsep}{10pt plus 2pt minus 3pt}\setlength{\floatsep}{8pt plus 2pt minus 2pt}\setlength{\intextsep}{8pt plus 2pt minus 2pt}
"""
    return r"""\documentclass[10pt,twocolumn,letterpaper]{article}
% To switch to the conference template, replace the class line above and delete the page-style block below
% (the document body uses only standard sectioning commands, booktabs tables and \includegraphics).
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\usepackage[left=0.7in,right=0.7in,top=0.75in,bottom=0.8in,columnsep=0.28in]{geometry}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{caption}
\usepackage{xcolor}
\usepackage{microtype}
\usepackage[colorlinks=true,linkcolor=blue!50!black,citecolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\usepackage{titlesec}
\usepackage{fancyhdr}
\pagestyle{fancy}\fancyhf{}\fancyfoot[C]{\thepage}\renewcommand{\headrulewidth}{0pt}\flushbottom
\titleformat{\section}{\normalsize\bfseries}{\thesection.}{0.5em}{\MakeUppercase}
\titleformat{\subsection}{\normalsize\bfseries\itshape}{\thesubsection.}{0.5em}{}
\titlespacing*{\section}{0pt}{8pt}{4pt}
\titlespacing*{\subsection}{0pt}{6pt}{3pt}
\captionsetup{font=small,labelfont=bf,skip=4pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{2pt plus 1pt minus 0.5pt}
\newtheorem{proposition}{Proposition}
"""


@app.function
def fig_block(figs_dir, name, caption, label, wide=False, width=None):
    if not (Path(figs_dir) / f"{name}.pdf").exists():
        return ""
    lncs = PAPER_STYLE["template"] == "lncs"
    env = "figure*" if (wide and not lncs) else "figure"
    if lncs:  # single-column figures were drawn for 3.45 in, i.e. 0.72 of the 4.8 in LNCS text width; wide ones are shown at full width
        w = width or (r"\linewidth" if wide else r"0.58\linewidth")
    else:
        w = width or (r"0.98\textwidth" if wide else r"\columnwidth")
    return (f"\\begin{{{env}}}[{'!htbp' if lncs else 't'}]\n\\centering\n\\includegraphics[width={w}]{{figs/{name}.pdf}}\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{{env}}}\n")


@app.function
def verdict_word(v):
    return {"positive": "supported", "negative": "contradicted", "inconclusive": "inconclusive"}.get(v, "not tested")


@app.function
def heading_if_content(out, pos, title):
    """Insert a (sub)section heading at position `pos` only if something non-empty was appended after it."""
    if any(x for x in out[pos:]):
        out.insert(pos, title)


@app.function
def render_abstract(ctx):
    A, C = ctx["A"], ctx["C"]
    a = ctx["assign"]
    n_main = max([len(A.get("present", {}).get(k, [])) for k in ("base", "graft")] + [0])
    sh = C.get("short", {})
    parts = [
        "A stride-4 (P2) prediction level is the standard remedy for small objects in UAV imagery, but it is usually added by "
        "re-wiring the neck of a COCO-pretrained detector, which discards much of the pretrained weights, and its benefit is rarely "
        "separated from label assignment, input resolution and evaluation protocol. "
        f"We study these factors on VisDrone-DET2019 with YOLOv10n under one protocol (saved predictions, COCO AP, TIDE, paired bootstrap, "
        f"{n_main} seeds, {A.get('epochs', '--')} epochs). "
        "P2-Graft appends a stride-4 branch that only \\emph{reads} the pretrained neck: the original levels are "
        f"reproduced exactly at initialisation (max.\\ output difference {ctx['exact_str']}) and all inheritable pretrained parameters are used, versus "
        f"{ctx['naive_copied_str']}\\% for textbook re-wiring. "]
    if a:
        parts.append(f"An exact anchor analysis shows that {100 * a['P3-P5|floor=0']['frac_unassigned']:.1f}\\% of the training boxes own no anchor "
                     f"centre on P3--P5 under classic task-aligned assignment ({100 * a['P2-P5|floor=0']['frac_unassigned']:.1f}\\% with P2), "
                     "which the small-target window floor of the Ultralytics release removes by itself. ")
    res = [sh[k] for k in ("main", "interaction", "naive", "iso") if k in sh]
    if res:
        parts.append("Results: " + "; ".join(res) + ".")
    return "".join(parts).strip()


@app.function
def render_paper_tex(ctx):
    """Full LaTeX source.  Numbers, verdicts and which sections exist are computed from `ctx` (nothing hard-coded)."""
    A, C = ctx["A"], ctx["C"]
    a, audit, cen = ctx["assign"], ctx["audit"], ctx["census"]
    env = ctx.get("env", {})
    figs = ctx["figs_dir"]
    E = A.get("epochs", "--")
    S = C["sentences"]
    base, graft = cond_row(A, "base"), cond_row(A, "graft")
    train = cen.get("train", {})
    med_w = a.get("median_w_px") if a else None
    med_h = a.get("median_h_px") if a else None
    nseeds = max([len(A.get("present", {}).get(k, [])) for k in ("base", "graft")] + [0])
    bs_B = A.get("bootstrap_B", "--")

    tpl = ctx.get("template", "lncs")
    PAPER_STYLE["template"] = tpl
    lncs = tpl == "lncs"
    banner_box = (("\\begin{center}\\fcolorbox{red}{white}{\\parbox{0.9\\textwidth}{\\centering\\bfseries\\color{red}"
                   + tex_escape(ctx["banner"]) + "}}\\end{center}") if ctx.get("banner") else "")
    if tpl == "lncs":
        # Springer LNCS, single-blind review: names and affiliations are mandatory (placeholders below).
        out = [paper_preamble("lncs"),
               (("\\usepackage{fancyhdr}\\pagestyle{fancy}\\fancyhf{}\\fancyhead[C]{\\color{red}\\footnotesize\\bfseries " + tex_escape(ctx["banner"])
                 + "}\\fancyfoot[C]{\\thepage}\\renewcommand{\\headrulewidth}{0pt}\\setlength{\\headheight}{14pt}") if ctx.get("banner") else ""),
               "\\begin{document}", "",
               "\\title{Grafting, Not Rewiring: A Function-Preserving Stride-4 Level for a Pretrained YOLOv10n and Its Interaction with Small-Target Label Assignment}",
               "\\titlerunning{Grafting, Not Rewiring}",
               "\\author{[First Author]\\inst{1} \\and [Second Author]\\inst{2}}",
               "\\authorrunning{[F. Author and S. Author]}",
               "\\institute{[Affiliation, city, country] \\email{[e-mail]} \\and [Affiliation, city, country] \\email{[e-mail]}}",
               "\\maketitle", banner_box,
               "\\begin{abstract}", render_abstract(ctx),
               "\\keywords{small object detection \\and UAV imagery \\and YOLOv10 \\and pretrained initialisation \\and label assignment \\and reproducible evaluation}",
               "\\end{abstract}", ""]
    else:
        out = [paper_preamble("article"),
               (("\\fancyhead[C]{\\color{red}\\footnotesize\\bfseries " + tex_escape(ctx["banner"]) + "}\\setlength{\\headheight}{14pt}") if ctx.get("banner") else ""),
               "\\begin{document}", "",
               "\\title{\\Large\\bfseries Grafting, Not Rewiring: A Function-Preserving Stride-4 Level for a Pretrained YOLOv10n\\\\[2pt]"
               "\\normalsize\\normalfont and Its Interaction with Small-Target Label Assignment on VisDrone}",
               "\\author{\\textit{[Author names, affiliations and e-mail addresses to be completed by the authors]}}",
               "\\date{}", "\\twocolumn[{", "\\maketitle", banner_box,  # braces: the abstract contains ']' (confidence intervals)
               "\\begin{center}\\begin{minipage}{0.92\\textwidth}\\small",
               "\\noindent\\rule{\\textwidth}{0.4pt}\\par\\vspace{4pt}",
               "\\noindent\\textbf{Abstract.} " + render_abstract(ctx),
               "\\par\\vspace{4pt}\\noindent\\textbf{Keywords.} small object detection, UAV imagery, YOLOv10, pretrained initialisation, label assignment, reproducible evaluation.",
               "\\par\\vspace{4pt}\\noindent\\rule{\\textwidth}{0.4pt}", "\\end{minipage}\\end{center}\\vspace{8pt}", "}]", ""]

    # ------------------------------------------------------------------ 1 Introduction
    out.append("\\section{Introduction}")
    out.append(
        "Unmanned aerial vehicles are now routine sensing platforms for traffic monitoring, search and rescue and infrastructure inspection, and the "
        "detectors that run on them must be small and fast~\\cite{ref_yolov10,ref_visdrone,ref_uavdt}. Aerial detection is hard mainly because objects "
        f"are tiny: in the VisDrone-DET2019 training split ({train.get('images', '--')} images, {train.get('objects', 0):,} annotated objects) "
        f"{100 * train.get('frac_area_lt_32sq', float('nan')):.1f}\\% of the boxes have an area below $32^2$ px and "
        f"{100 * train.get('frac_area_lt_16sq', float('nan')):.1f}\\% below $16^2$ px at the original resolution; after the usual resize to a 640\\,px "
        f"network input the median box is only {f1(med_w)}$\\times${f1(med_h)} px~\\cite{{ref_visdrone}}.")
    out.append(
        "The most common remedy is to add a stride-4 (P2) prediction level to a COCO-pretrained detector~\\cite{ref_tphyolov5,ref_querydet,ref_lightuav,ref_rddyolo}. "
        "In practice this is done by editing the neck, so that the bottom-up path and the detection heads receive different inputs than they were pretrained "
        "with. Frameworks then initialise the new model by copying every tensor whose name and shape happen to match, which keeps part of the weights, "
        "silently drops the rest, and can place a tensor in a layer whose role has changed. At the same time, the benefit of the extra level is "
        "entangled with three other factors that papers rarely control: (i) how small ground-truth boxes are assigned to anchors, (ii) the input resolution, "
        "and (iii) the evaluation protocol, including whether baselines are re-trained or quoted from other publications.")
    out.append("We therefore ask four pre-specified questions, answered under one training and evaluation pipeline:")
    out.append("\\begin{itemize}\\setlength\\itemsep{0pt}")
    out.append("\\item[\\textbf{H1}] Does a stride-4 level improve a pretrained YOLOv10n on VisDrone when the library's default assignment is used?")
    out.append("\\item[\\textbf{H2}] Does it matter \\emph{how} the pretrained weights are carried over (grafting vs.\\ re-wiring)?")
    out.append("\\item[\\textbf{H3}] Is the benefit of the stride-4 level smaller when the assigner already enlarges the candidate window of tiny boxes?")
    out.append("\\item[\\textbf{H4}] At matched compute, is a stride-4 level a better use of FLOPs than a higher input resolution?")
    out.append("\\end{itemize}")
    out.append("\\textbf{Contributions.} (1) \\emph{P2-Graft}, a function-preserving way to extend a pretrained detector with a stride-4 level, together with an "
               "explicit role-based weight-transfer rule and transfer accounting (Section~\\ref{sec:method}). (2) An exact anchor-candidate analysis of VisDrone that "
               "quantifies how many boxes cannot be assigned at all under classic task-aligned assignment and how a P2 level or a window floor changes this. "
               "(3) A controlled study with a matched baseline trained by us, a $2\\times2$ factorial over level and assignment, negative controls (naive "
               "re-wiring, removing P5, random initialisation) and a compute-matched resolution baseline, evaluated from saved predictions with COCO AP, "
               "TIDE and an image-level paired bootstrap. All numbers are produced by one open notebook, and the text of Section~\\ref{sec:results} "
               "is generated from its outputs.")
    verdicts = C["verdicts"]
    if verdicts:
        out.append("\\textbf{Outcome in brief.} " + " ".join(
            f"H{k[1:]}~{verdict_word(v[0])}." for k, v in sorted(verdicts.items())) + " Details and caveats are in Sections~\\ref{sec:results} and~\\ref{sec:discussion}.")

    # ------------------------------------------------------------------ 2 Related work
    out.append("\\section{Related Work}")
    out.append(
        "\\textbf{Small objects and high-resolution pyramid levels.} Feature pyramids~\\cite{ref_fpn,ref_panet} are the standard answer to scale variation, and "
        "a fourth prediction head fed by high-resolution backbone features is a frequent remedy for tiny objects in drone imagery~\\cite{ref_tphyolov5,ref_lightuav,ref_rddyolo}; "
        "QueryDet~\\cite{ref_querydet} reduces the cost of high-resolution features with sparse queries, and slicing at inference~\\cite{ref_sahi} trades latency for recall. "
        "Surveys and benchmarks document that objects below $32\\times32$ px remain the dominant failure case~\\cite{ref_nikouei2025,ref_tsod_survey,ref_visdrone,ref_uavdt,ref_aitod,ref_soda}.")
    out.append(
        "\\textbf{YOLO variants for aerial images.} Many papers modify YOLOv8/v10/v11 with attention, lightweight convolutions or extra heads "
        "and compare against numbers taken from other publications~\\cite{ref_bgf,ref_osd,ref_sscw,ref_toe,ref_lightuav,ref_rddyolo}. Such tables combine different training "
        "lengths, augmentations, input sizes and evaluation code, so a gain over a quoted baseline cannot be attributed to the proposed module; we avoid this by training every "
        "compared model ourselves, with identical code, data order seeds and evaluation.")
    out.append(
        "\\textbf{Label assignment for tiny objects.} Task-aligned assignment~\\cite{ref_tood}, in the form used by YOLOv8-style detectors, chooses for every ground-truth box the top-$k$ "
        "anchors among those whose centres lie inside the box. For boxes smaller than the grid stride this set can be empty, which motivated Gaussian receptive-field assignment~\\cite{ref_rfla} and "
        "Wasserstein-based similarity~\\cite{ref_nwd}. The Ultralytics release used in this study~\\cite{ref_ultralytics} enlarges the candidate window of very small boxes to a minimum side "
        "length (Section~\\ref{sec:assign}); we treat this \\emph{window floor} as an experimental factor.")
    out.append(
        "\\textbf{Growing a pretrained network.} Net2Net~\\cite{ref_net2net} introduced function-preserving transformations that widen or deepen a trained network without "
        "changing its output, and ControlNet~\\cite{ref_controlnet} attaches new trainable branches to a frozen network so that the augmented model starts exactly at the "
        "original. P2-Graft applies the same principle to the \\emph{detection pyramid}: the new level only consumes existing activations.")
    out.append(
        "\\textbf{Evaluation practice.} We use COCO AP~\\cite{ref_coco} with the detection cap raised to the VisDrone density, TIDE error analysis~\\cite{ref_tide} on saved predictions, "
        "multiple seeds~\\cite{ref_bouthillier} and an image-level bootstrap~\\cite{ref_efron}.")

    # ------------------------------------------------------------------ 3 Method
    out.append("\\section{Method}\\label{sec:method}")
    out.append("\\subsection{Setting}")
    out.append(
        "We use the nano YOLOv10 detector~\\cite{ref_yolov10} as implemented in Ultralytics " + tex_escape(ULTRALYTICS_PIN) + "~\\cite{ref_ultralytics}: a CSP-style backbone $B$ "
        "(stem, C2f stages, SCDown, SPPF, PSA), a PAN-style neck $N$ producing features $p_3,p_4,p_5$ (strides 8/16/32) and one detection head $H_\\ell$ per level with a "
        "one-to-many and a one-to-one branch. Training uses the end-to-end loss of this release, which mixes a one-to-many loss (top-$k$ task-aligned assignment) and a one-to-one "
        "loss, with weights that move from 0.8/0.2 to 0.1/0.9 over the schedule; inference uses the NMS-free one-to-one branch unless stated otherwise. "
        "Box regression uses distribution focal loss~\\cite{ref_gfl}. The COCO-pretrained checkpoint has 80 classes; the final class-logit convolutions of a 10-class model "
        "are re-initialised, and rows whose class name matches a COCO class (bicycle, car, truck, bus) are copied by name.")
    out.append("\\subsection{P2-Graft}\\label{sec:graft}")
    out.append(
        "Let $b_2$ be the stride-4 backbone feature (output of the first C2f stage) and $p_3$ the pretrained P3 neck feature. P2-Graft adds "
        "\\begin{equation}p_2=G_\\phi\\big([\\,\\mathrm{Up}_{2\\times}(p_3),\\,b_2\\,]\\big),\\qquad y_2=H^{(2)}_\\psi(p_2),\\end{equation} "
        "where $G_\\phi$ is a neck block of the same type and width as the P3 block and $H^{(2)}_\\psi$ is a detection head built like the others "
        "(Figure~\\ref{fig:arch}). The extended model outputs $\\{y_3,y_4,y_5,y_2\\}$; the new level is appended \\emph{last}, so the parameter names of the existing "
        "levels do not change.")
    out.append(
        "\\begin{proposition}[exact preservation] If the parameters of $B$, $N$ and $H_3,H_4,H_5$ equal those of the pretrained detector, the outputs $y_3,y_4,y_5$ of the "
        "grafted model equal those of the pretrained detector for every input, whatever $\\phi,\\psi$ are.\\end{proposition}")
    out.append(
        "\\emph{Proof.} $G_\\phi$ and $H^{(2)}_\\psi$ consume $p_3$ and $b_2$ and produce only new tensors; no existing layer reads their outputs. $\\square$ "
        f"We also verified this numerically on random inputs with the real checkpoint: the maximum absolute difference over the neck features and over the raw outputs of "
        f"all four prediction branches (one-to-many/one-to-one $\\times$ box/class) of the three original levels is {ctx['exact_str']}. "
        "The statement concerns the \\emph{starting point}: all parameters, pretrained ones included, are fine-tuned jointly, so gradients of the new level reach the shared layers.")
    out.append(fig_block(figs, "fig_architecture", "P2-Graft. Grey modules are the pretrained detector and are left untouched; orange modules are new. "
                         "The new branch reads the P3 neck feature and the stride-4 backbone feature and adds a fourth prediction level.", "fig:arch", wide=True))
    out.append("\\subsection{Transfer accounting}")
    out.append(
        "Let $\\Theta_{\\mathrm{inh}}$ be the parameters of the 80-class checkpoint that a 10-class model of the \\emph{same} architecture can inherit (everything except the shape-dependent "
        "class-logit convolutions). For a variant and a transfer rule we report the fraction of $\\Theta_{\\mathrm{inh}}$ that is copied, and the fraction that is copied into a layer whose "
        "entire upstream sub-graph (types, arguments, wiring) is identical to the one in the pretrained model (\\emph{role-aligned}). "
        "Two rules are compared: \\emph{key matching}, which is what Ultralytics does by default (same tensor name and shape), and a \\emph{role-based map} written per variant "
        "(for P2-Graft it only renames the detection module's index, $23\\!\\to\\!26$; for the P5-removal variant it additionally drops the P5 neck and the P5 branch of the head).")
    out.append("\\subsection{Anchor assignability and the window floor}\\label{sec:assign}")
    out.append(
        "In Ultralytics' task-aligned assigner a ground-truth box $b=(x_1,y_1,x_2,y_2)$ can only receive positives from anchors whose centres $((i+\\tfrac12)s,(j+\\tfrac12)s)$ "
        "lie strictly inside the box. The release used here first raises every side shorter than $w_{\\min}$ to $w_{\\min}$ (around the box centre) for this test, where "
        "$w_{\\min}$ defaults to the second entry of the model's stride list (16\\,px for P3--P5); we call $w_{\\min}$ the \\emph{window floor} and set it explicitly in every run. With $\\tilde b$ the floored box and $\\mathcal S$ the set "
        "of strides, the number of candidates is "
        "\\begin{equation}N(b)=\\sum_{s\\in\\mathcal S}\\#\\{(i,j):\\,\\tilde x_1<(i+\\tfrac12)s<\\tilde x_2,\\ \\tilde y_1<(j+\\tfrac12)s<\\tilde y_2\\},\\end{equation} "
        "and $N(b)=0$ means the box can never become a positive. We evaluate $N(b)$ exactly on all training boxes (Section~\\ref{sec:res_assign}).")
    out.append("\\subsection{Systems compared}")
    out.append(
        "\\textbf{Baseline}: YOLOv10n with COCO initialisation (role map = identity). \\textbf{P2-Graft} (ours). \\textbf{Naive P2}: the textbook YOLOv8-P2-style re-wiring "
        "(P2 output feeds a new bottom-up path), initialised by key matching. \\textbf{P2$\\rightarrow$P5}: a P2-Graft whose P5 level and P5 neck are removed. "
        "\\textbf{Random init}: baseline and P2-Graft without pretrained weights. \\textbf{Resolution baseline}: the baseline trained and tested at a higher input size chosen "
        "to match the GFLOPs of P2-Graft. Each can be trained with the library-default window floor (16\\,px) or with classic assignment (floor 0).")

    # ------------------------------------------------------------------ 4 Experimental protocol
    out.append("\\section{Experimental Protocol}\\label{sec:setup}")
    out.append(
        "\\textbf{Data.} VisDrone-DET2019~\\cite{ref_visdrone,ref_visdrone2019}: 6{,}471 training, 548 validation and 1{,}610 test-dev images, ten classes. Ignored regions and "
        "``others'' are not used as objects. All models are trained on the full training split with one fixed recipe, \\emph{without} any hyper-parameter search and "
        "\\emph{without} checkpoint selection (the last-epoch EMA weights are evaluated), so validation and test-dev are both untouched held-out sets. "
        "Images are resized once (INTER\\_AREA) to the training long side and boxes are mapped back to the original resolution for evaluation.")
    out.append(
        f"\\textbf{{Training.}} {E} epochs from the COCO checkpoint, SGD (lr 0.01, momentum 0.937, weight decay $5\\!\\times\\!10^{{-4}}$, linear decay to $0.01\\times$), 3 warm-up epochs, "
        "batch 32 with gradient accumulation to a nominal batch of 64, mosaic augmentation switched off for the last "
        f"{int(min(10, max(1, E // 4))) if isinstance(E, int) else '--'} epochs, mixed precision, EMA, input 640\\,px (unless stated), the Ultralytics default augmentation otherwise. "
        "The schedule is deliberately short (the budget of one GPU session); absolute numbers are therefore lower than 300-epoch results reported in the literature and must not be compared with them.")
    gpu = env.get("gpu", "an NVIDIA GPU")
    odd_batch = sorted(f"{r} (batch {int(b)})" for r, b in zip(A["df"]["run_id"], A["df"]["batch"]) if int(b) not in (32, -1)) if "batch" in A["df"] else []
    lat_seq = bool("latency_seq" in A["df"] and len(A["df"]) and A["df"]["latency_seq"].fillna(False).astype(bool).all())
    lat_txt = ("Latency is the median of 300 batch-1 forward passes of the fused deploy graph in FP16, measured for one model at a time after all training had finished."
               if lat_seq else
               "Latency is the median of 200 batch-1 forward passes of the fused deploy graph in FP16, measured at the end of each run while other trainings shared the GPU and the CPU, "
               "so differences of about a millisecond between models are noise.")
    out.append(
        f"\\textbf{{Hardware and software.}} {tex_escape(gpu)}, PyTorch {tex_escape(env.get('torch', '--'))}, Ultralytics {tex_escape(env.get('ultralytics', ULTRALYTICS_PIN))}, "
        f"{env.get('cpu_count', '--')} CPU threads. " + lat_txt
        + (" After a CUDA out-of-memory error the following runs were repeated with a smaller batch (nominal batch 64 kept by gradient accumulation): "
           + tex_escape(", ".join(odd_batch)) + "." if odd_batch else ""))
    out.append(
        "\\textbf{Evaluation.} Predictions are produced with confidence threshold 0.001 and at most 500 detections per image (VisDrone images contain up to "
        f"{train.get('objects_per_image_max', '--')} objects), saved, and scored against the \\emph{{original-resolution}} boxes with COCO AP (IoU 0.50:0.95), AP$_{{50}}$, AP in the COCO "
        "area bins and two finer bins ($<16^2$ and $16^2$--$32^2$ px). The primary head is the NMS-free one-to-one branch. Secondary results use (i) the one-to-many branch with NMS, "
        "(ii) VisDrone's ignore regions turned into per-class crowd regions, and (iii) larger test sizes. TIDE~\\cite{ref_tide} is computed from the same saved predictions.")
    out.append(
        f"\\textbf{{Statistics.}} Main conditions are repeated with {nseeds} seeds (initialisation of new layers, augmentation and data order). We report mean $\\pm$ sample standard deviation, paired per-seed differences, "
        f"and a paired image-level bootstrap ({bs_B} resamples of the validation images, seed-averaged AP) for differences between conditions. The primary metric of H1--H4 is AP (IoU 0.50:0.95) on VisDrone-val. "
        "With so few seeds, p-values are not reported as evidence; single-seed controls are labelled as such.")

    # ------------------------------------------------------------------ 5 Results
    out.append("\\section{Results}\\label{sec:results}")
    out.append("\\subsection{How many boxes can be assigned at all?}\\label{sec:res_assign}")
    if a:
        u35, u25 = 100 * a["P3-P5|floor=0"]["frac_unassigned"], 100 * a["P2-P5|floor=0"]["frac_unassigned"]
        bs = a["by_size"]
        out.append(
            f"At 640\\,px the median training box is {f1(a['median_w_px'])}$\\times${f1(a['median_h_px'])} px, {100 * a['frac_min_side_lt_8']:.1f}\\% of the boxes have a side below 8\\,px and "
            f"{100 * a['frac_min_side_lt_4']:.1f}\\% below 4\\,px. Under classic task-aligned assignment {u35:.1f}\\% of the boxes own no anchor centre on P3--P5 "
            f"({100 * bs[0]['P3-P5|floor=0']:.1f}\\% of those with $\\sqrt{{wh}}<8$\\,px); a P2 level lowers the overall share to {u25:.1f}\\%, and a window floor of 8 or 16\\,px reduces it to "
            f"{100 * a['P3-P5|floor=8']['frac_unassigned']:.2f}\\% / {100 * a['P3-P5|floor=16']['frac_unassigned']:.2f}\\% even without P2 (Table~\\ref{{tab:assign}}, Figure~\\ref{{fig:assign}}). "
            f"The floor achieves this by admitting anchors that lie \\emph{{outside}} the true box (candidates per box: {a['P3-P5|floor=0']['mean_candidates']:.1f} $\\rightarrow$ {a['P3-P5|floor=16']['mean_candidates']:.1f}). "
            "The two mechanisms therefore address the same problem, which motivates H3.")
        out.append(table_assign(a))
        out.append(fig_block(figs, "fig_assignability", "Boxes without any candidate anchor under classic task-aligned assignment, by box size (VisDrone-train, 640\\,px input).", "fig:assign"))
    _p = len(out)
    if audit:
        rows = {(r["variant"], r["mode"]): r for r in audit["rows"]}
        gk, nk = rows.get(("graft", "key")), rows.get(("naive", "key"))
        out.append(
            f"The grafted model reproduces the three original levels exactly (max.\\ difference {ctx['exact_str']}) and inherits 100\\% of $\\Theta_{{\\mathrm{{inh}}}}$ under the role-based map, whereas key matching "
            f"copies only {100 * gk['frac_of_inheritable']:.1f}\\% for the same architecture (the detection module's index shifts) and {100 * nk['frac_of_inheritable']:.1f}\\% for the naive P2 re-wiring "
            f"(Table~\\ref{{tab:transfer}}); in the P2$\\rightarrow$P5 variant key matching additionally copies {100 * (rows[('replace','key')]['frac_of_inheritable'] - rows[('replace','key')]['frac_aligned_of_inheritable']):.1f}\\% "
            "of $\\Theta_{\\mathrm{inh}}$ into layers whose role has changed.")
        out.append(table_transfer(audit))
    heading_if_content(out, _p, "\\subsection{Exactness and transfer coverage}")
    _p = len(out)
    if base and graft:
        out.append((S.get("main", "") + " " + S.get("cost", "")).strip() + " See Table~\\ref{tab:main} and Figure~\\ref{fig:main}.")
        out.append(fig_block(figs, "fig_main", "Baseline vs.\\ P2-Graft on VisDrone-val: (a) overall metrics and (b) AP$_{50}$ by object area. Bars: mean over seeds; error bars: sample std; dots: individual seeds.", "fig:main", wide=True))
        out.append(table_main(A))
    heading_if_content(out, _p, "\\subsection{Main comparison}")
    _p = len(out)
    tf = table_factorial(A)
    if tf:
        out.append(S.get("interaction", "") + f" Hypothesis H3 is {verdict_word(C['verdicts'].get('H3', ('',))[0])} (basis: {tex_escape(C['verdicts'].get('H3', ('', ''))[1])}); "
                   + ("see Table~\\ref{tab:factorial}." if lncs else "see Table~\\ref{tab:factorial} and Figure~\\ref{fig:factorial}."))
        out.append(tf)
        if not lncs:  # (the 2x2 table carries the same numbers; the plot stays in the bundle's figs/ for the LNCS page budget)
            out.append(fig_block(figs, "fig_factorial", "Interaction between the stride-4 level and the window floor of the assigner.", "fig:factorial"))
    heading_if_content(out, _p, "\\subsection{Stride-4 level $\\times$ assignment}")
    _p = len(out)
    ctrl = []
    if "naive" in S:
        ctrl.append(S["naive"])
    rp = cond_row(A, "replace")
    if rp and graft:
        ctrl.append(f"Removing P5 while keeping the grafted P2 level gives {f1(rp['val_AP'])} AP ({sgn(rp['val_AP'] - graft['val_AP'])} vs.\\ P2-Graft) and "
                    f"{f1(rp['val_APl'])} AP$_L$ ({sgn(rp['val_APl'] - graft['val_APl'])}), which tests the earlier suggestion that dropping the stride-32 head hurts large vehicles.")
    for key, lab in (("graft_scratch", "P2-Graft"), ("base_scratch", "the baseline")):
        sc = cond_row(A, key)
        ref = graft if key == "graft_scratch" else base
        if sc and ref:
            ctrl.append(f"Without pretrained weights {lab} reaches {f1(sc['val_AP'])} AP after {E} epochs ({sgn(sc['val_AP'] - ref['val_AP'])} vs.\\ the pretrained model).")
    if ctrl:
        out.append(" ".join(ctrl) + " Controls other than the main pair use a single seed unless Table~\\ref{tab:main} lists more; the seed-to-seed standard deviation of the main conditions gives the scale of noise.")
    if not lncs:  # (in the LNCS page budget the loss curves stay in the bundle's figs/ only)
        out.append(fig_block(figs, "fig_curves", "Training loss per epoch (box + class + DFL, one-to-one branch) of seed-0 runs; the pretrained start is compared with random initialisation.", "fig:curves"))
    heading_if_content(out, _p, "\\subsection{Controls: re-wiring, removing P5, random initialisation}")
    _p = len(out)
    if "iso" in S:
        out.append(S["iso"] + f" Hypothesis H4 is {verdict_word(C['verdicts'].get('H4', ('',))[0])} (basis: {tex_escape(C['verdicts'].get('H4', ('', ''))[1])}); see Figure~\\ref{{fig:pareto}}.")
    tts = table_testsize(A)
    if tts and lncs:
        ts = larger_input_sentence(A)
        out.append("Spending extra compute at inference time only: " + ts[:1].lower() + ts[1:])
    elif tts:
        out.append("Table~\\ref{tab:testsize} evaluates the models trained at 640\\,px at larger test sizes (800 and 960\\,px) without retraining, "
                   "which spends the same extra compute at inference time only.")
        out.append(tts)
    out.append(fig_block(figs, "fig_pareto", "Accuracy versus deploy GFLOPs (a) and versus measured latency (b); one marker per condition (mean over seeds).", "fig:pareto", wide=True))
    heading_if_content(out, _p, "\\subsection{Compute-matched comparison with input resolution}")
    _p = len(out)
    td = table_tide(A)
    if td:
        b, g = base, graft
        out.append("TIDE attributes the AP$_{50}$ lost relative to an oracle to error types (Table~\\ref{tab:tide}). Change from the baseline to P2-Graft in points of AP$_{50}$ "
                   "(negative = fewer errors): "
                   + ", ".join(f"{e} {sgn(g['tide_' + e] - b['tide_' + e], 2)}" for e in ("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss") if f"tide_{e}" in b and f"tide_{e}" in g) + ".")
        out.append(td)
        if not lncs:
            out.append(fig_block(figs, "fig_tide", "TIDE error decomposition on VisDrone-val.", "fig:tide"))
    qual_caption = ("Qualitative examples from VisDrone-val (images with the largest increase in confident detections; detections with score $\\ge 0.3$ in a dense crop). "
                    "Counts of drawn boxes are shown.")
    has_qual = (Path(figs) / "fig_qualitative.pdf").exists()
    if (Path(figs) / "fig_perclass.pdf").exists():
        out.append("Figure~\\ref{fig:perclass} shows the per-class change in AP$_{50}$"
                   + ((" and Figure~\\ref{fig:qual} (Appendix~\\ref{app:qual}) gives qualitative examples." if lncs else " and Figure~\\ref{fig:qual} gives qualitative examples.")
                      if has_qual else "."))
    out.append(fig_block(figs, "fig_perclass", "Per-class change in AP$_{50}$ (P2-Graft minus baseline, mean over seeds).", "fig:perclass"))
    if not lncs:
        out.append(fig_block(figs, "fig_qualitative", qual_caption, "fig:qual", wide=True))
    heading_if_content(out, _p, "\\subsection{Error analysis}")
    _p = len(out)
    sens = []
    if base and graft and isnum(base.get("test_AP")) and isnum(graft.get("test_AP")):
        sens.append(f"On the 1{{,}}610 test-dev images the AP change is {sgn(graft['test_AP'] - base['test_AP'])} ({f1(base['test_AP'])} $\\rightarrow$ {f1(graft['test_AP'])}), "
                    f"and the AP$_{{50}}$ change is {sgn(graft['test_AP50'] - base['test_AP50'])}.")
    if base and graft and isnum(base.get("val_AP50_ign")):
        sens.append(f"Turning VisDrone's ignore regions into crowd regions raises AP$_{{50}}$ from {f1(base['val_AP50'])} to {f1(base['val_AP50_ign'])} (baseline) and from {f1(graft['val_AP50'])} to "
                    f"{f1(graft['val_AP50_ign'])} (P2-Graft); the difference between the two models changes from {sgn(graft['val_AP50'] - base['val_AP50'])} to {sgn(graft['val_AP50_ign'] - base['val_AP50_ign'])}.")
    if base and graft and isnum(base.get("val_AP_o2m")):
        sens.append(f"Using the one-to-many branch with NMS instead of the NMS-free branch gives {f1(base['val_AP_o2m'])} (baseline) and {f1(graft['val_AP_o2m'])} AP (P2-Graft).")
    if base and graft and isnum(base.get("val_AP50_conf25")) and isnum(graft.get("val_AP50_conf25")):
        sens.append(f"Shortcuts in the evaluation pipeline matter as much as architecture changes: scoring only detections with confidence $\\ge0.25$ (the operating point of "
                    f"precision/recall columns) lowers AP$_{{50}}$ from {f1(base['val_AP50'])} to {f1(base['val_AP50_conf25'])} for the baseline and from {f1(graft['val_AP50'])} to "
                    f"{f1(graft['val_AP50_conf25'])} for P2-Graft, and the difference between the models becomes {sgn(graft['val_AP50_conf25'] - base['val_AP50_conf25'])} "
                    f"instead of {sgn(graft['val_AP50'] - base['val_AP50'])}")
        if isnum(base.get("val_AP50_maxdet100")) and isnum(graft.get("val_AP50_maxdet100")):
            sens[-1] += (f"; capping the output at 100 detections per image (the COCO default) gives {f1(base['val_AP50_maxdet100'])} and {f1(graft['val_AP50_maxdet100'])} "
                         "although VisDrone images contain up to hundreds of objects")
        sens[-1] += "."
    if base and graft and isnum(base.get("val_ultra_AP50")) and isnum(graft.get("val_ultra_AP50")):
        sens.append(f"As an independent cross-check, Ultralytics' own validator (rectangular batches, its own matching rule) gives AP$_{{50}}$ {f1(base['val_ultra_AP50'])} / {f1(graft['val_ultra_AP50'])} "
                    f"for the baseline / P2-Graft, a difference of {sgn(graft['val_ultra_AP50'] - base['val_ultra_AP50'])} against {sgn(graft['val_AP50'] - base['val_AP50'])} with the COCO evaluator; "
                    "absolute values of different evaluators can differ by about a point, so only differences measured with one evaluator are compared in this paper.")
    if sens:
        out.append(" ".join(sens))
    heading_if_content(out, _p, "\\subsection{Test-dev and protocol sensitivity}")
    _p = len(out)
    for nm, info in (A.get("external") or {}).items():
        col = info["column"]
        if base and graft and isnum(base.get(f"ext_{col}_AP50")) and isnum(graft.get(f"ext_{col}_AP50")):
            tn = info.get("target_names") or {}
            cm = ", ".join(f"{tex_escape(CLASS_NAMES[int(k)])}$\\rightarrow${tex_escape(str(tn.get(str(v), v)))}" for k, v in sorted(info["class_map"].items(), key=lambda kv: int(kv[0])))
            nb = sum(1 for r in info["runs"] if str(r).split("-")[1:2] == ["base"])
            ng = sum(1 for r in info["runs"] if str(r).split("-")[1:2] == ["graft"])
            out.append(f"Zero-shot transfer to {tex_escape(nm)} ({info['n_images']} images, never used for training; VisDrone classes mapped as {cm}; "
                       f"detections of unmapped classes are dropped; mean over {nb} baseline and {ng} P2-Graft runs): AP$_{{50}}$ {f1(base[f'ext_{col}_AP50'])} for the baseline "
                       f"and {f1(graft[f'ext_{col}_AP50'])} for P2-Graft ({sgn(graft[f'ext_{col}_AP50'] - base[f'ext_{col}_AP50'])}), AP {f1(base[f'ext_{col}_AP'])} and "
                       f"{f1(graft[f'ext_{col}_AP'])} ({sgn(graft[f'ext_{col}_AP'] - base[f'ext_{col}_AP'])}). This is a single-dataset, mapped-class check without an "
                       "uncertainty estimate, not a full cross-dataset study.")
    heading_if_content(out, _p, "\\subsection{Zero-shot transfer to another dataset}")

    # ------------------------------------------------------------------ 6 Discussion
    out.append("\\section{Discussion}\\label{sec:discussion}")
    rows = []
    desc = {"H1": "stride-4 level improves AP (library-default assignment)", "H2": "grafting transfers better than re-wiring",
            "H3": "benefit of the level is smaller when the window floor is active", "H4": "at matched GFLOPs the level beats higher resolution"}
    for h in ("H1", "H2", "H3", "H4"):
        if h in C["verdicts"]:
            v, why = C["verdicts"][h]
            rows.append([h, tex_escape(desc[h]), verdict_word(v), tex_escape(why)])
        else:
            rows.append([h, tex_escape(desc[h]), "not tested", "required runs missing"])
    out.append(latex_table(["", "Hypothesis", "Verdict", "Basis"], rows, "Pre-specified hypotheses and what the collected runs support (primary metric: AP on VisDrone-val).",
                           "tab:verdicts", colspec="lp{0.42\\textwidth}lp{0.32\\textwidth}", star=True))
    out.append(
        "\\textbf{Reading the evidence.} Verdicts are computed mechanically: ``supported'' or ``contradicted'' requires that the 95\\% image-bootstrap interval of the seed-averaged difference "
        "excludes zero \\emph{and}, with several paired seeds, the same sign in every pair; everything else is ``inconclusive''. "
        "Single-seed verdicts say so, because image-level intervals capture test-set sampling, not training variance. "
        "All verdicts describe this model scale, dataset and schedule.")
    out.append(
        "\\textbf{Context for the resolution comparison.} Most VisDrone images are $960\\times540$ px, so a 640\\,px input down-scales them by a factor of 1.5 whereas 896\\,px is close to native resolution; "
        "part of what a stride-4 level recovers at 640\\,px is detail that a larger input does not discard in the first place. "
        "Likewise, the naive re-wiring differs from P2-Graft both in architecture and in the share of inherited weights; "
        "a graft initialised by key matching, which would separate the two, was not trained here.")
    ext_note = ("the only evaluation outside VisDrone is the mapped-class zero-shot check above, which is not a cross-dataset study. " if A.get("external")
                else "no cross-dataset evaluation is performed. ")
    out.append(
        "\\textbf{Limitations.} (i) Short fine-tuning budget ("
        f"{E} epochs); the new level starts from random weights, so a longer schedule may favour it more than reported here. (ii) One training dataset (VisDrone) and one detector scale (nano); "
        + ("a second detector family (YOLO26n) appears only in the run table of Appendix~\\ref{app:runs}, and " if (A["df"]["family"] != "v10").any() else "") + ext_note +
        "(iii) Controls other than the main pair are single-seed unless Table~\\ref{tab:main} lists more. (iv) The window floor is a property of the Ultralytics release used; other frameworks differ. "
        "(v) Latency is measured on one GPU, not on embedded hardware. (vi) The main protocol follows the YOLO literature (no ignore-region rule); the ignore-region variant is reported. "
        "(vii) No state-of-the-art claim is made.")
    out.append(
        "\\textbf{Recommendations (independent of the accuracy ranking).} When extending a pretrained detector with a pyramid level, report (a) how many pretrained parameters were "
        "carried over and where, (b) the assigner's treatment of boxes smaller than the stride, and (c) a baseline trained under the same pipeline plus a resolution baseline of equal FLOPs.")

    out.append(
        "\\textbf{Ethical considerations.} Aerial detectors have dual uses (traffic monitoring, search and rescue, but also surveillance). This study uses a public research "
        "benchmark, evaluates coarse object categories only, and releases no model or tool aimed at identifying individuals.")
    out.append(
        "\\textbf{Reproducibility.} The study is one marimo notebook (\\texttt{graft\\_p2\\_study.py}, version " + tex_escape(STUDY_VERSION)
        + (", SHA-256 " + tex_escape(ctx["nb_hash"][:16]) + "\\ldots" if ctx.get("nb_hash") else "")
        + ") that downloads VisDrone, trains all runs, evaluates from saved predictions and writes this paper (software and hardware: Section~\\ref{sec:setup}; Python "
        + tex_escape(env.get("python", "--")) + "). Code and saved predictions: \\textit{[URL to be added]}. "
        "Ultralytics is released under AGPL-3.0; VisDrone is distributed for research use.")

    # ------------------------------------------------------------------ 7 Conclusion
    out.append("\\section{Conclusion}")
    concl = []
    if "main" in S:
        concl.append(S["main"])
    if "interaction" in S:
        concl.append(S["interaction"])
    out.append(" ".join(concl) + " P2-Graft shows that an added pyramid level can start exactly at the pretrained detector; "
               "whether it should be preferred over re-wiring, over a window floor, or over a higher input size is answered by Tables~\\ref{tab:main}--\\ref{tab:verdicts} for this setting, "
               "and the released notebook makes the same questions cheap to ask for other detectors and datasets.")

    # ------------------------------------------------------------------ appendix
    out.append("\\appendix")
    out.append("\\section{All runs}\\label{app:runs}")
    if not A["df"].empty:
        out.append(table_perseed(A))
    if lncs and (Path(figs) / "fig_qualitative.pdf").exists():
        out.append("\\section{Qualitative examples}\\label{app:qual}")
        out.append(fig_block(figs, "fig_qualitative", qual_caption, "fig:qual", wide=True))
    out.append("\\bibliographystyle{splncs04}" if tpl == "lncs" else "\\bibliographystyle{plain}")
    out.append("\\bibliography{references}")
    out.append("\\end{document}")
    return "\n\n".join(x for x in out if x)


@app.function
def bib_text():
    """BibTeX for every key cited in the generated paper.  PLEASE VERIFY each entry against the publisher record before submission."""
    return r"""@inproceedings{ref_yolov10,
  author = {Wang, Ao and Chen, Hui and Liu, Lihao and Chen, Kai and Lin, Zijia and Han, Jungong and Ding, Guiguang},
  title = {{YOLOv10}: Real-Time End-to-End Object Detection},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2024},
}
@article{ref_visdrone,
  author = {Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
  title = {Detection and Tracking Meet Drones Challenge},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume = {44}, number = {11}, pages = {7380--7399}, year = {2022},
}
@inproceedings{ref_visdrone2019,
  author = {Du, Dawei and others},
  title = {{VisDrone-DET2019}: The Vision Meets Drone Object Detection in Image Challenge Results},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  year = {2019},
}
@inproceedings{ref_uavdt,
  author = {Du, Dawei and Qi, Yuankai and Yu, Hongyang and Yang, Yifan and Duan, Kaiwen and Li, Guorong and Zhang, Weigang and Huang, Qingming and Tian, Qi},
  title = {The Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)}, pages = {370--386}, year = {2018},
}
@inproceedings{ref_coco,
  author = {Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C. Lawrence},
  title = {Microsoft {COCO}: Common Objects in Context},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)}, pages = {740--755}, year = {2014},
}
@inproceedings{ref_fpn,
  author = {Lin, Tsung-Yi and Doll{\'a}r, Piotr and Girshick, Ross and He, Kaiming and Hariharan, Bharath and Belongie, Serge},
  title = {Feature Pyramid Networks for Object Detection},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, pages = {2117--2125}, year = {2017},
}
@inproceedings{ref_panet,
  author = {Liu, Shu and Qi, Lu and Qin, Haifang and Shi, Jianping and Jia, Jiaya},
  title = {Path Aggregation Network for Instance Segmentation},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, pages = {8759--8768}, year = {2018},
}
@article{ref_nikouei2025,
  author = {Nikouei, Mahya and Baroutian, Bita and Nabavi, Shahriar and Taraghi, Fatemeh and Aghaei, Amirreza and Sajedi, Aydin and Ebrahimi Moghaddam, Mohsen},
  title = {Small Object Detection: A Comprehensive Survey on Challenges, Techniques and Real-World Applications},
  journal = {Intelligent Systems with Applications}, volume = {27}, year = {2025},
}
@article{ref_tsod_survey,
  author = {Rekavandi, Aref Miri and Rashidi, Shima and Boussaid, Farid and Hoefs, Stephen and Akbas, Emre and Bennamoun, Mohammed},
  title = {Transformers in Small Object Detection: A Benchmark and Survey of State-of-the-Art},
  journal = {arXiv preprint arXiv:2309.04902}, year = {2023},
}
@article{ref_bgf,
  author = {Mei, Jianqiang and Zhu, Wei},
  title = {{BGF-YOLOv10}: Small Object Detection Algorithm from Unmanned Aerial Vehicle Perspective Based on Improved {YOLOv10}},
  journal = {Sensors}, volume = {24}, number = {21}, pages = {6911}, year = {2024},
}
@article{ref_osd,
  author = {Zhang, Yang and Chen, Xin and Sun, Shuo and You, Hui and Wang, Yue and Lin, Jian and Wang, Jing},
  title = {Vehicle Detection in Drone Aerial Views Based on Lightweight {OSD-YOLOv10}},
  journal = {Scientific Reports}, volume = {15}, pages = {25155}, year = {2025},
}
@article{ref_sscw,
  author = {He, Zhuolun and She, Rui and Tan, Bo and Li, Jiajian and Lei, Xiaolong},
  title = {{SSCW-YOLO}: A Lightweight and High-Precision Model for Small Object Detection in {UAV} Scenarios},
  journal = {Drones}, volume = {10}, number = {1}, pages = {41}, year = {2026},
}
@article{ref_toe,
  author = {Yan, Hui and Kong, Xiangzhi and Shimada, Tetsuya and Tomiyama, Hiroyuki},
  title = {{TOE-YOLO}: Accurate and Efficient Detection of Tiny Objects in {UAV} Imagery},
  journal = {Journal of Real-Time Image Processing}, volume = {22}, pages = {194}, year = {2025},
}
@article{ref_lightuav,
  author = {Lyu, Yingqi and Zhang, Ting and Li, Xin and Liu, An and Shi, Gang},
  title = {{LightUAV-YOLO}: A Lightweight Object Detection Model for Unmanned Aerial Vehicle Image},
  journal = {Journal of Supercomputing}, volume = {81}, pages = {172}, year = {2025},
}
@inproceedings{ref_rddyolo,
  author = {Song, Chen and Ge, Wei},
  title = {{RDD-YOLO}: An Improved Small Object Detection Model for {UAV} Images Based on {YOLO11}},
  booktitle = {Proceedings of the International Conference on Aerospace System Science and Engineering}, organization = {IEEE}, year = {2025},
}
@inproceedings{ref_sahi,
  author = {Akyon, Fatih Cagatay and Altinuc, Sinan Onur and Temizel, Alptekin},
  title = {Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection},
  booktitle = {Proceedings of the IEEE International Conference on Image Processing (ICIP)}, pages = {966--970}, year = {2022},
}
@inproceedings{ref_aitod,
  author = {Wang, Jinwang and Yang, Wen and Guo, Haowen and Zhang, Ruixiang and Xia, Gui-Song},
  title = {Tiny Object Detection in Aerial Images},
  booktitle = {Proceedings of the International Conference on Pattern Recognition (ICPR)}, pages = {3791--3798}, year = {2021},
}
@article{ref_soda,
  author = {Cheng, Gong and Yuan, Xiang and Yao, Xiwen and Yan, Kebing and Zeng, Qinghua and Xie, Xingxing and Han, Junwei},
  title = {Towards Large-Scale Small Object Detection: Survey and Benchmarks},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence}, volume = {45}, number = {11}, pages = {13364--13387}, year = {2023},
}
@inproceedings{ref_querydet,
  author = {Yang, Chenhongyi and Huang, Zehao and Wang, Naiyan},
  title = {{QueryDet}: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, pages = {13668--13677}, year = {2022},
}
@inproceedings{ref_tide,
  author = {Bolya, Daniel and Foley, Sean and Hays, James and Hoffman, Judy},
  title = {{TIDE}: A General Toolbox for Identifying Object Detection Errors},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)}, pages = {558--573}, year = {2020},
}
@inproceedings{ref_tphyolov5,
  author = {Zhu, Xingkui and Lyu, Shuhao and Wang, Xiao and Zhao, Qi},
  title = {{TPH-YOLOv5}: Improved {YOLOv5} Based on Transformer Prediction Head for Object Detection on Drone-captured Scenarios},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)}, pages = {2778--2788}, year = {2021},
}
@inproceedings{ref_tood,
  author = {Feng, Chengjian and Zhong, Yujie and Gao, Yu and Scott, Matthew R. and Huang, Weilin},
  title = {{TOOD}: Task-aligned One-stage Object Detection},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)}, pages = {3490--3499}, year = {2021},
}
@inproceedings{ref_rfla,
  author = {Xu, Chang and Wang, Jinwang and Yang, Wen and Yu, Huai and Yu, Lei and Xia, Gui-Song},
  title = {{RFLA}: {Gaussian} Receptive Field Based Label Assignment for Tiny Object Detection},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)}, pages = {526--543}, year = {2022},
}
@article{ref_nwd,
  author = {Wang, Jinwang and Xu, Chang and Yang, Wen and Yu, Lei},
  title = {A Normalized {Gaussian} {Wasserstein} Distance for Tiny Object Detection},
  journal = {arXiv preprint arXiv:2110.13389}, year = {2021},
}
@inproceedings{ref_gfl,
  author = {Li, Xiang and Wang, Wenhai and Wu, Lijun and Chen, Shuo and Hu, Xiaolin and Li, Jun and Tang, Jinhui and Yang, Jian},
  title = {Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, year = {2020},
}
@inproceedings{ref_net2net,
  author = {Chen, Tianqi and Goodfellow, Ian and Shlens, Jonathon},
  title = {{Net2Net}: Accelerating Learning via Knowledge Transfer},
  booktitle = {International Conference on Learning Representations (ICLR)}, year = {2016},
}
@inproceedings{ref_controlnet,
  author = {Zhang, Lvmin and Rao, Anyi and Agrawala, Maneesh},
  title = {Adding Conditional Control to Text-to-Image Diffusion Models},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)}, pages = {3836--3847}, year = {2023},
}
@inproceedings{ref_bouthillier,
  author = {Bouthillier, Xavier and Delaunay, Pierre and Bronzi, Mirko and Trofimov, Assya and Nichyporuk, Brennan and Szeto, Justin and Sepah, Naz and Raff, Edward and Madan, Kanika and Voleti, Vikram and others},
  title = {Accounting for Variance in Machine Learning Benchmarks},
  booktitle = {Proceedings of Machine Learning and Systems (MLSys)}, year = {2021},
}
@book{ref_efron,
  author = {Efron, Bradley and Tibshirani, Robert J.},
  title = {An Introduction to the Bootstrap}, publisher = {Chapman \& Hall/CRC}, year = {1993},
}
@misc{ref_ultralytics,
  author = {Jocher, Glenn and Qiu, Jing and Chaurasia, Ayush},
  title = {Ultralytics {YOLO}},
  year = {2026}, howpublished = {\url{https://github.com/ultralytics/ultralytics}}, note = {version 8.4.150, AGPL-3.0},
}
"""


@app.function
def build_paper(work, out_dir=None, B=200, do_bootstrap=True, make_figs=True, log_path=None, iso_size=896, family="v10", banner=None, template="lncs"):
    """Run the whole analysis on the finished runs and write a compilable LaTeX paper bundle.
    template: "lncs" (Springer LNCS, the ICPR/ICPRv format; needs Springer's llncs.cls/splncs04.bst) or "article" (self-contained two-column article).
    Returns a dict with the output directory, the analysis dictionary and the claims."""
    work = Path(work)
    if template not in ("lncs", "article"):
        raise ValueError("template must be 'lncs' or 'article'")
    PAPER_STYLE["template"] = template
    out_dir = Path(out_dir) if out_dir else work / "paper"
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    for stale in figs_dir.glob("fig_*"):  # never let a figure from an earlier (different) state leak into this paper
        stale.unlink()
    A = analyze(work, B=B, do_bootstrap=do_bootstrap, iso_size=iso_size, family=family, log_path=log_path)
    if A["n_runs"] == 0:
        raise RuntimeError("no finished runs found under " + str(work / "runs"))
    df = A["df"]
    # ---- static analyses (cheap, re-computed if missing)
    assign = jload(work / "analysis" / "assignability.json") or assignability_table(work, "train")
    jdump(assign, work / "analysis" / "assignability.json")
    audit = jload(work / "analysis" / "transfer_audit.json")
    if audit is None:
        audit = transfer_audit(work, family)
        jdump(audit, work / "analysis" / "transfer_audit.json")
    _nv = [r for r in audit["rows"] if r["variant"] == "naive" and r["mode"] == "key"]
    A["audit_naive_frac"] = _nv[0]["frac_of_inheritable"] if _nv else float("nan")
    C = build_claims(A)
    exact = jload(work / "analysis" / "exactness.json")
    if exact is None:
        exact = exactness_report(work, family)
        jdump(exact, work / "analysis" / "exactness.json")
    census = {sp: (jload(work / "analysis" / f"census_{sp}.json") or dataset_census(work, sp)) for sp in ("train", "val")}
    for sp, c in census.items():
        jdump(c, work / "analysis" / f"census_{sp}.json")
    d = max(exact["graft"]["max_abs_feature_diff"], exact["graft"]["max_abs_head_output_diff"])
    exact_str = "exactly $0$ (bit-identical)" if d == 0 else f"${d:.1e}$"
    rows = {(r["variant"], r["mode"]): r for r in audit["rows"]}
    naive_copied = f"{100 * rows[('naive', 'key')]['frac_of_inheritable']:.0f}"
    env = (jload(work / "runs" / str(df["run_id"].iloc[0]) / "result.json", {}) or {}).get("env", {})
    # ---- figures
    figs: Dict[str, Any] = {}
    if make_figs:
        keys = A["keys"]
        try:
            figs["architecture"] = fig_architecture(figs_dir)
            figs["assignability"] = fig_assignability(assign, figs_dir)
            if A["present"].get("base") and A["present"].get("graft"):
                figs["main"] = fig_main_comparison(df, figs_dir, keys["base"], keys["graft"])
                figs["tide"] = fig_tide(df, figs_dir)
                figs["perclass"] = fig_perclass(df, figs_dir)
            if all(A["present"].get(n) for n in ("base", "graft", "base_f0", "graft_f0")):
                figs["factorial"] = fig_factorial(df, figs_dir)
            if len(A["agg"]) >= 3:
                figs["pareto"] = fig_pareto(df, figs_dir)
            sel = [("base", "YOLOv10n baseline"), ("graft", "P2-Graft"), ("naive", "naive P2"), ("graft_scratch", "P2-Graft, random init")]
            rr = [(runs_for_key(df, keys[k]), lab, k) for k, lab in sel if runs_for_key(df, keys[k])]
            if len(rr) >= 2:
                fig_curves(work, figs_dir, [r[0][0][1] for r in rr], [r[1] for r in rr], [condition_style(keys[r[2]][1], keys[r[2]][2], keys[r[2]][3], keys[r[2]][4]) for r in rr])
            ra, rb = runs_for_key(df, keys["base"]), runs_for_key(df, keys["graft"])
            if ra and rb and run_pred_path(work, ra[0][1]).exists() and run_pred_path(work, rb[0][1]).exists():
                figs["qualitative"] = fig_qualitative(work, figs_dir, ra[0][1], rb[0][1], n=2 if template == "lncs" else 3)
        except Exception as e:
            log(f"figure generation error: {e!r}\n{traceback.format_exc()}", log_path)
    try:
        nb_hash = hashlib.sha256(Path(NOTEBOOK_FILE).read_bytes()).hexdigest() if NOTEBOOK_FILE else None
    except Exception:
        nb_hash = None
    ctx = {"A": A, "C": C, "assign": assign, "audit": audit, "census": census, "env": env, "figs_dir": figs_dir, "exact_str": exact_str,
           "naive_copied_str": naive_copied, "banner": banner, "nb_hash": nb_hash, "template": template}
    tex = render_paper_tex(ctx)
    (out_dir / "main.tex").write_text(tex, encoding="utf-8")
    (out_dir / "references.bib").write_text(bib_text(), encoding="utf-8")
    summary = {"claims": C, "analysis": {k: v for k, v in A.items() if k not in ("df", "agg")}, "exactness": exact, "assignability": assign,
               "transfer_audit": audit, "census": census, "env": env, "n_runs": A["n_runs"], "notebook_sha256": nb_hash, "study_version": STUDY_VERSION}
    jdump(summary, out_dir / "results_summary.json")
    df.to_csv(out_dir / "all_runs.csv", index=False)
    A["agg"].to_csv(out_dir / "conditions_summary.csv", index=False)
    # bundle (paper + figures + csv); predictions are zipped separately because they are large
    bundle = shutil.make_archive(str(work / "paper_bundle"), "zip", root_dir=str(out_dir))
    return {"out_dir": str(out_dir), "bundle": bundle, "A": A, "C": C, "figs": figs, "tex_path": str(out_dir / "main.tex")}


@app.function
def zip_predictions(work, out_zip=None):
    """Optional: bundle all saved predictions (npz) and per-run result.json files for release."""
    work = Path(work)
    out_zip = Path(out_zip) if out_zip else work / "predictions_and_results.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted((work / "runs").glob("*/*.npz")) + sorted((work / "runs").glob("*/result.json")) + sorted((work / "runs").glob("*/train_info.json")) \
                + sorted((work / "runs").glob("*/results.csv")) + sorted((work / "runs").glob("*/args.yaml")):
            z.write(p, p.relative_to(work))
    return str(out_zip)


# =====================================================================================================
# SECTION I -- helpers used by the notebook cells
# =====================================================================================================
@app.function
def run_fn_in_subprocess(notebook_file, fn_name, kwargs, log_path, env_extra=None):
    """Run module-level function `fn_name(**kwargs)` of this notebook in a fresh interpreter; output goes to log_path."""
    code = ("import sys, json, importlib.util; "
            f"spec = importlib.util.spec_from_file_location('graft_p2_study_mod', r'{notebook_file}'); "
            "m = importlib.util.module_from_spec(spec); sys.modules['graft_p2_study_mod'] = m; spec.loader.exec_module(m); "
            f"kw = json.loads(sys.argv[1]); r = getattr(m, '{fn_name}')(**kw); "
            "print('RESULT_JSON', json.dumps(r, default=str))")
    env = dict(os.environ)
    env.update({"PYTHONUNBUFFERED": "1"})
    env.update(env_extra or {})
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as lf:
        p = subprocess.run([sys.executable, "-c", code, json.dumps(kwargs, default=str)], stdout=lf, stderr=subprocess.STDOUT, env=env)
    return p.returncode


@app.function
def last_result_json(log_path):
    try:
        for line in reversed(Path(log_path).read_text(errors="ignore").splitlines()):
            if line.startswith("RESULT_JSON "):
                return json.loads(line[len("RESULT_JSON "):])
    except Exception:
        pass
    return None


@app.function
def tail_text(path, n=25):
    try:
        return "\n".join(Path(path).read_text(errors="ignore").splitlines()[-n:])
    except Exception:
        return ""


@app.function
def synthetic_preds_from_gt(gt_json, seed=0, jitter=0.12, p_correct_cls=0.85, fp_per_img=25, miss_rate=0.2):
    """Noisy copies of the ground truth (+ random false positives) with random scores: a detector-free test input whose AP is
    far from 0 and 1, used to verify that every evaluator / the bootstrap accumulation agree on non-trivial data."""
    rng = np.random.default_rng(seed)
    g = jload(gt_json)
    img, box, sc, cl = [], [], [], []
    by_img: Dict[int, List[Any]] = {}
    for a in g["annotations"]:
        if not a.get("iscrowd", 0):
            by_img.setdefault(a["image_id"], []).append(a)
    for im in g["images"]:
        i = im["id"]
        for a in by_img.get(i, []):
            if rng.random() < miss_rate:
                continue
            x, y, w, h = a["bbox"]
            dx, dy, dw, dh = rng.normal(0, jitter, 4) * np.array([w, h, w, h])
            bw, bh = max(1.0, w + dw), max(1.0, h + dh)
            img.append(i)
            box.append([x + dx, y + dy, x + dx + bw, y + dy + bh])
            sc.append(float(np.clip(rng.beta(4, 2), 0.01, 0.999)))
            c = a["category_id"] - 1
            cl.append(c if rng.random() < p_correct_cls else int(rng.integers(0, 10)))
        for _ in range(fp_per_img):
            x, y = rng.uniform(0, im["width"] * 0.95), rng.uniform(0, im["height"] * 0.95)
            w, h = rng.uniform(5, 60), rng.uniform(5, 60)
            img.append(i)
            box.append([x, y, x + w, y + h])
            sc.append(float(rng.beta(1.5, 4)))
            cl.append(int(rng.integers(0, 10)))
    return {"img": np.asarray(img, np.int32), "xyxy": np.asarray(box, np.float32).reshape(-1, 4), "score": np.asarray(sc, np.float32), "cls": np.asarray(cl, np.int8)}


@app.function
def evaluator_selftest(gt_json, seed=0, B=3):
    """Cross-check on non-trivial synthetic detections: pycocotools == faster-coco-eval == own bootstrap accumulation (unit weights);
    plus a sanity check that re-sampling images changes AP smoothly."""
    p = synthetic_preds_from_gt(gt_json, seed)
    prep = boot_prepare(gt_json, p)
    ap, ap50 = boot_ap(prep, np.ones(prep["n_img"]))
    ref = coco_evaluate(gt_json, p, backend="pycocotools")
    alt = coco_evaluate(gt_json, p)
    rng = np.random.default_rng(seed)
    res = [boot_ap(prep, np.bincount(rng.integers(0, prep["n_img"], prep["n_img"]), minlength=prep["n_img"]).astype(float))[0] for _ in range(B)]
    return {"bootstrap_AP": ap, "pycocotools_AP": ref["AP"], "faster_AP": alt["AP"], "bootstrap_AP50": ap50, "pycocotools_AP50": ref["AP50"], "faster_AP50": alt["AP50"],
            "resampled_AP": res, "n_dets": int(len(p["img"])),
            "agree": bool(abs(ap - ref["AP"]) < 1e-6 and abs(ap50 - ref["AP50"]) < 1e-6 and abs(alt["AP"] - ref["AP"]) < 1e-6 and abs(alt["AP50"] - ref["AP50"]) < 1e-6)}


@app.function
def smoke_test(work, limit=48, size=320, device=None):
    """Tiny end-to-end check (a few minutes): data -> exactness -> train/eval both models -> bootstrap self-test -> TIDE.
    The numbers it produces are NOT results (a handful of images and one epoch)."""
    work = Path(work)
    steps: List[Dict[str, Any]] = []

    def step(name, fn):
        t0 = time.time()
        try:
            val = fn()
            steps.append({"step": name, "ok": True, "seconds": round(time.time() - t0, 1), "detail": val})
        except Exception as e:
            steps.append({"step": name, "ok": False, "seconds": round(time.time() - t0, 1), "detail": repr(e) + "\n" + traceback.format_exc()[-600:]})

    step("environment", lambda: {k: env_info().get(k) for k in ("python", "torch", "ultralytics", "cuda_available", "gpu")})
    step("exactness of the graft (random inputs, real COCO checkpoint)", lambda: {v: {"max_feature_diff": r["max_abs_feature_diff"],
                                                                                    "max_head_diff": r["max_abs_head_output_diff"]} for v, r in exactness_report(work).items()})
    for sp in ("train", "val"):
        step(f"prepare {sp} subset ({limit} images @ {size}px)", lambda sp=sp: bool(prepare_split(work, sp, sizes=(size,), limit=limit)))
    results = {}
    for variant in ("base", "graft"):
        spec = RunSpec(run_id=f"smoke-{variant}", group=variant, variant=variant, epochs=1, imgsz=size, batch=8, seed=0)

        def go(spec=spec):
            shutil.rmtree(work / "runs" / spec.run_id, ignore_errors=True)
            ok = run_one(str(work), asdict(spec), workers=0, cache=False, device=device, data_suffix=f"_lim{limit}", limit=limit, tide=True, test_dev=False)
            r = jload(work / "runs" / spec.run_id / "result.json", {})
            results[variant] = r
            m = r.get("metrics", {}).get(f"val|{size}|o2o|std", {})
            return {"run_one_ok": ok, "AP50": m.get("AP50"), "n_dets": m.get("n_dets"), "tide_ok": "main" in (r.get("tide|val") or {})}
        step(f"train + evaluate {variant} (1 epoch, tiny)", go)

    def selftest():
        return evaluator_selftest(work / "data" / f"gt_val_lim{limit}.json")
    step("evaluator self-test on synthetic detections (bootstrap accumulation == pycocotools == faster-coco-eval)", selftest)
    return steps


# =====================================================================================================
# SECTION J -- environment probe / repair, worker plan, autopilot, diagnostics
# (molab keeps only files uploaded through its file browser, shuts notebooks down after 90 idle minutes and may lack some of the packages)
# =====================================================================================================
@app.function
def required_modules():
    """(import name, pip requirement) of everything the study needs besides torch / torchvision / marimo."""
    return [("ultralytics", f"ultralytics=={ULTRALYTICS_PIN}"), ("pycocotools", "pycocotools"), ("faster_coco_eval", "faster-coco-eval"), ("tidecv", "tidecv"),
            ("psutil", "psutil"), ("tqdm", "tqdm"), ("scipy", "scipy"), ("yaml", "pyyaml"), ("requests", "requests"), ("matplotlib", "matplotlib"),
            ("PIL", "pillow"), ("pandas", "pandas"), ("numpy", "numpy"), ("cv2", "opencv-python-headless")]


@app.function
def libgl_hints():
    return ("libGL", "libgthread", "libglib", "libSM", "libXext", "libXrender", "libxcb")


@app.function
def probe_environment(work=None, timeout=300):
    """Inspect the Python environment in a SUBPROCESS, so that a broken import (e.g. OpenCV without libGL, a segfault) cannot take the notebook kernel down
    and nothing heavy is imported into the kernel.  Returns versions (None = cannot be imported), the import errors, CUDA facts, disk and RAM."""
    mods = [m for m, _ in required_modules()] + ["torch", "torchvision", "marimo"]
    code = (
        "import json, sys, platform, importlib\n"
        "out = {'python': sys.version.split()[0], 'executable': sys.executable, 'platform': platform.platform(), 'modules': {}, 'errors': {}}\n"
        f"for mod in {mods!r}:\n"
        "    try:\n"
        "        m = importlib.import_module(mod)\n"
        "        out['modules'][mod] = str(getattr(m, '__version__', '?'))\n"
        "    except BaseException as e:\n"
        "        out['modules'][mod] = None\n"
        "        out['errors'][mod] = (type(e).__name__ + ': ' + str(e))[:300]\n"
        "try:\n"
        "    import torch\n"
        "    out['cuda_available'] = bool(torch.cuda.is_available())\n"
        "    out['torch_cuda'] = torch.version.cuda\n"
        "    if out['cuda_available']:\n"
        "        p = torch.cuda.get_device_properties(0)\n"
        "        out.update(gpu=p.name, gpu_gb=round(p.total_memory / 1e9, 1), gpu_capability='%d.%d' % (p.major, p.minor),\n"
        "                   torch_arch_list=torch.cuda.get_arch_list(), n_gpus=torch.cuda.device_count())\n"
        "        x = torch.randn(256, 256, device='cuda')\n"
        "        out['cuda_matmul_ok'] = bool((x @ x).isfinite().all().item())\n"
        "except BaseException as e:\n"
        "    out['torch_error'] = (type(e).__name__ + ': ' + str(e))[:300]\n"
        "if sys.platform.startswith('linux'):\n"  # containers often have a tiny /dev/shm: data-loader workers then die with 'Bus error'
        "    try:\n"
        "        import shutil, torch\n"
        "        out['shm_gb'] = round(shutil.disk_usage('/dev/shm').total / 1e9, 2)\n"
        "        from torch.utils.data import DataLoader, Dataset\n"
        "        class _D(Dataset):\n"
        "            def __len__(self): return 96\n"
        "            def __getitem__(self, i): return torch.full((3, 640, 640), i % 255, dtype=torch.uint8)\n"
        "        out['dataloader_workers_ok'] = sum(1 for _ in DataLoader(_D(), batch_size=32, num_workers=3, prefetch_factor=2)) == 3\n"
        "    except BaseException as e:\n"
        "        out['dataloader_workers_ok'] = False\n"
        "        out['dataloader_error'] = (type(e).__name__ + ': ' + str(e))[:300]\n"
        "print('PROBE_JSON ' + json.dumps(out))\n")
    info: Dict[str, Any]
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout)
        line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith("PROBE_JSON ")), None)
        info = json.loads(line[len("PROBE_JSON "):]) if line else {"probe_failed": (r.stderr or r.stdout)[-400:] or f"exit code {r.returncode}"}
    except Exception as e:
        info = {"probe_failed": repr(e)}
    try:  # what the driver sees, independent of PyTorch (a GPU the driver sees but torch does not = a CUDA build / driver mismatch)
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        info["nvidia_smi"] = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        info["nvidia_smi"] = None
    p = Path(work) if work else default_work_dir()
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        info["disk_free_gb"] = round(shutil.disk_usage(p).free / 1e9, 1)
    except Exception:
        info["disk_free_gb"] = None
    ram_total, ram_avail = effective_ram_gb()
    info.update(ram_gb=round(ram_total, 1), ram_available_gb=round(ram_avail, 1), cpus=effective_cpus(), cpus_host=os.cpu_count())
    return info


@app.function
def diagnose_environment(info, need_cuda=True):
    """-> list of {level: 'error'|'warn', code, msg, fixable}.  'fixable' problems are handled by ensure_environment()."""
    out: List[Dict[str, Any]] = []

    def add(level, code, msg, fixable=False):
        out.append({"level": level, "code": code, "msg": msg, "fixable": fixable})

    if info.get("probe_failed"):
        add("error", "probe", f"the environment probe crashed: {info['probe_failed']}")
        return out
    try:
        if tuple(int(x) for x in str(info["python"]).split(".")[:2]) < (3, 10):
            add("error", "python", f"Python {info['python']} is too old (3.10+ needed)")
    except Exception:
        pass
    mods, errs = info.get("modules", {}), info.get("errors", {})
    libgl = any(any(h in e for h in libgl_hints()) for e in errs.values())
    if libgl:
        add("error", "cv2_libgl", "OpenCV cannot load its system libraries (" + next(e for e in errs.values() if any(h in e for h in libgl_hints()))[:160]
            + "); the headless OpenCV build fixes this", True)
    for mod, req in required_modules():
        if mods.get(mod) is None and not (libgl and mod in ("cv2", "ultralytics", "tidecv")):
            add("error", f"missing:{mod}", f"cannot import `{mod}` ({errs.get(mod, 'not installed')[:160]})", True)
    for mod in ("torch", "torchvision"):
        if mods.get(mod) is None:
            add("error", f"missing:{mod}", f"cannot import `{mod}` ({errs.get(mod, 'not installed')[:160]})", True)
    if mods.get("ultralytics") and mods["ultralytics"] != ULTRALYTICS_PIN:
        add("warn", "pin", f"Ultralytics {mods['ultralytics']} is installed but the study was validated with {ULTRALYTICS_PIN} "
            "(the assigner's window floor, the loss defaults and the model YAML are version-specific)", True)
    if need_cuda and mods.get("torch"):
        if not info.get("cuda_available"):
            if info.get("nvidia_smi"):
                add("error", "no_cuda_driver", f"the driver sees a GPU ({info['nvidia_smi']}) but PyTorch does not: the installed build (torch {mods['torch']}, CUDA {info.get('torch_cuda')}) "
                    f"probably needs a newer driver. Use the button *Install PyTorch for CUDA 12.8* below, then restart the kernel. {info.get('torch_error', '')}")
            else:
                add("error", "no_cuda", "no GPU is visible (nvidia-smi finds none). In molab: click the notebook-specs button in the header and toggle the GPU, then restart. "
                    f"(torch {mods['torch']}, built for CUDA {info.get('torch_cuda')}; {info.get('torch_error', '')})")
        elif not info.get("cuda_matmul_ok", True):
            add("error", "cuda_broken", "a CUDA matrix multiplication failed - the PyTorch build does not support this GPU")
        else:
            cap, arch = info.get("gpu_capability", "0.0"), info.get("torch_arch_list") or []
            if cap.startswith("12") and not any(a in ("sm_120", "compute_120") for a in arch):
                add("error", "arch", f"this GPU has compute capability {cap} but the installed PyTorch lists {arch}: install a CUDA 12.8+ build "
                    "(`pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu128`) and restart the kernel")
            if (info.get("gpu_gb") or 0) < 20:
                add("warn", "gpu_small", f"only {info.get('gpu_gb')} GB of GPU memory: the P2 model needs roughly 20 GB at batch 32 (the OOM back-off will halve the batch)")
    if info.get("dataloader_workers_ok") is False:
        add("warn", "shm", f"data-loader worker processes fail on this machine (/dev/shm = {info.get('shm_gb')} GB; {str(info.get('dataloader_error', ''))[:120]}): "
            "training will use 0 workers, which is much slower - the planner will fit fewer epochs")
    disk = info.get("disk_free_gb")
    if disk is not None and disk < 8:
        add("error", "disk", f"only {disk} GB of free disk space (≈ 10 GB are needed for data, runs and predictions)")
    elif disk is not None and disk < 15:
        add("warn", "disk_low", f"{disk} GB of free disk space (≈ 10 GB are needed for data, runs and predictions)")
    if info.get("ram_available_gb") == info.get("ram_available_gb") and (info.get("ram_available_gb") or 99) < 8:
        add("warn", "ram", f"only {info.get('ram_available_gb')} GB of free RAM")
    return out


@app.function
def pip_install(reqs, no_deps=False, reinstall=False, extra_args=(), dry_run=False, log=print):
    """Install into the interpreter that runs this notebook (uv if it is on PATH, else pip, bootstrapping pip if needed). Returns the exit code."""
    reqs = list(reqs)
    uv = shutil.which("uv")
    cmd = ([uv, "pip", "install", "--python", sys.executable] if uv else [sys.executable, "-m", "pip", "install"])
    if no_deps:
        cmd.append("--no-deps")
    if reinstall:
        cmd.append("--reinstall" if uv else "--force-reinstall")
    cmd += list(extra_args) + reqs
    log("$ " + " ".join(cmd))
    if dry_run:
        return 0
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 and not uv and "No module named pip" in (r.stderr + r.stdout):
        subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], capture_output=True, text=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
    tail = (r.stdout + "\n" + r.stderr).strip().splitlines()[-5:]
    log("\n".join("    " + ln for ln in tail))
    return r.returncode


@app.function
def matching_torchvision(torch_version):
    """('0.24.1', 'https://download.pytorch.org/whl/cu128' or None) for torch '2.9.1+cu128' (torchvision's minor = torch's minor + 15)."""
    base, _, tag = str(torch_version).partition("+")
    parts = base.split(".")
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    if major != 2:
        raise ValueError(f"unsupported torch version {torch_version}")
    return f"0.{minor + 15}.{patch}", (f"https://download.pytorch.org/whl/{tag}" if tag.startswith("cu") else None)


@app.function
def install_no_opencv(dist, requirement, fallback_deps, dry_run=False, log=print):
    """Install `requirement` WITHOUT its dependencies, then its other dependencies except torch / torchvision / opencv-python (the full-fat OpenCV needs libGL,
    which containers often lack; the headless build is installed separately)."""
    rc = pip_install([requirement], no_deps=True, dry_run=dry_run, log=log)
    if rc != 0:
        return rc
    deps = list(fallback_deps)
    if not dry_run:
        try:
            import importlib
            import importlib.metadata as md
            importlib.invalidate_caches()
            deps = [r for r in (md.requires(dist) or []) if "extra ==" not in r and re.split(r"[ ;<>=!~\[(]", r.strip(), maxsplit=1)[0].lower().replace("_", "-")
                    not in ("torch", "torchvision", "opencv-python")]
        except Exception:
            pass
    return pip_install(deps, dry_run=dry_run, log=log) if deps else 0


@app.function
def ensure_environment(info=None, dry_run=False, log=print, need_cuda=True):
    """Repair what diagnose_environment() marks as fixable: missing packages, the headless OpenCV, the pinned Ultralytics, a missing torchvision.
    Idempotent. Never replaces an installed torch.  Returns {"actions": [...], "probe": new probe, "problems": diagnose(new probe)}."""
    info = info or probe_environment()
    mods, errs = info.get("modules", {}), info.get("errors", {})
    libgl = any(any(h in e for h in libgl_hints()) for e in errs.values())
    actions: List[str] = []
    # 1. torch / torchvision (only if absent; the preinstalled GPU build is kept)
    if mods.get("torch") is None:
        actions.append("install torch + torchvision (PyPI default build)")
        pip_install(["torch", "torchvision"], dry_run=dry_run, log=log)
    elif mods.get("torchvision") is None:
        try:
            tv, index = matching_torchvision(mods["torch"])
            actions.append(f"install torchvision=={tv} matching torch {mods['torch']}")
            pip_install([f"torchvision=={tv}"], no_deps=True, extra_args=(["--index-url", index] if index else []), dry_run=dry_run, log=log)
        except Exception as e:
            log(f"cannot derive a matching torchvision for torch {mods['torch']}: {e!r}")
    # 2. Ultralytics (pinned) and tidecv without their OpenCV requirement
    ultra_fallback = ["cloudpickle", "filelock", "numpy", "matplotlib", "pillow", "pyyaml", "requests", "psutil", "polars", "nvidia-ml-py", "ultralytics-thop"]
    if mods.get("ultralytics") != ULTRALYTICS_PIN:
        actions.append(f"install ultralytics=={ULTRALYTICS_PIN} (without OpenCV/torch requirements)")
        install_no_opencv("ultralytics", f"ultralytics=={ULTRALYTICS_PIN}", ultra_fallback, dry_run=dry_run, log=log)
    if mods.get("tidecv") is None:
        actions.append("install tidecv (without its OpenCV requirement)")
        install_no_opencv("tidecv", "tidecv", ["appdirs", "numpy", "pycocotools", "seaborn", "pandas", "matplotlib"], dry_run=dry_run, log=log)
    # 3. plain packages
    plain = [req for mod, req in required_modules() if mods.get(mod) is None and mod not in ("ultralytics", "tidecv", "cv2")]
    if plain:
        actions.append("install " + ", ".join(plain))
        pip_install(plain, dry_run=dry_run, log=log)
    # 4. headless OpenCV: when the normal one cannot load its system libraries, or is absent
    if libgl or mods.get("cv2") is None:
        actions.append("install opencv-python-headless (replaces the OpenCV build that needs libGL)")
        pip_install(["opencv-python-headless"], no_deps=True, reinstall=True, dry_run=dry_run, log=log)
    if dry_run:
        return {"actions": actions, "probe": info, "problems": diagnose_environment(info, need_cuda)}
    import importlib
    importlib.invalidate_caches()
    new = probe_environment()
    return {"actions": actions, "probe": new, "problems": diagnose_environment(new, need_cuda)}


@app.function
def install_cuda_torch(backend="cu128", dry_run=False, log=print):
    """Replace torch/torchvision by the build for CUDA `backend` (a Blackwell GPU needs CUDA >= 12.8; cu128 also runs on drivers that predate CUDA 13).
    Heavy (≈ 3 GB) - only on request.  Restart the kernel afterwards."""
    uv = shutil.which("uv")
    if uv:
        return pip_install(["torch", "torchvision"], extra_args=["--reinstall-package", "torch", "--reinstall-package", "torchvision", "--torch-backend", backend],
                           dry_run=dry_run, log=log)
    return pip_install(["torch", "torchvision"], reinstall=True, extra_args=["--index-url", f"https://download.pytorch.org/whl/{backend}", "--extra-index-url", "https://pypi.org/simple"],
                       dry_run=dry_run, log=log)


@app.function
def make_worker_plan(work, specs, par, cache, budget_h, s_per_epoch, limit=None, paper=None, notebook_file=None):
    """The dict that worker_main() consumes (shared by the numbered step ⑦ and the autopilot)."""
    now = time.time()
    plan = {"work": str(work), "specs": [asdict(s) for s in specs], "n_parallel": par["n_parallel"], "workers": par["workers"], "cache": cache, "device": None,
            "deadline": now + budget_h * 3600, "hard_deadline": now + min(budget_h * 3600 + 1800, 11.5 * 3600), "s_per_epoch": float(s_per_epoch),
            "eval_s": 240.0, "tide": True, "test_dev": limit is None, "paper": paper or {}, "auto_paper": limit is None}
    if limit is not None:
        plan.update(limit=limit, data_suffix=f"_lim{limit}")
    if notebook_file:
        plan["notebook_file"] = str(notebook_file)
    return plan


@app.function
def autopilot(work, budget_h=8.5, seeds=(0, 1, 2), tiers=(0, 1, 2), epochs=0, iso_size=896, families=("v10", "yolo26"), notebook_file=None,
              skip_smoke=False, skip_calibration=False, need_cuda=True, limit=None, template="lncs", bootstrap=300, log_path=None, launch=True, recalibrate=False):
    """Everything between an EMPTY work directory and a RUNNING training worker (notebook steps ②-⑦), in one call.  Idempotent: finished parts are skipped.
    `limit` runs the whole thing on a tiny subset of the data (test mode: no analyses, no calibration, no automatic paper).
    Returns {"ok": bool, "steps": [{step, ok, seconds, detail}], "plan": dict | None, "pid": int | None}."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    nb = Path(notebook_file) if notebook_file else NOTEBOOK_FILE
    steps: List[Dict[str, Any]] = []
    state: Dict[str, Any] = {}

    def step(name, fn):
        t0 = time.time()
        log(f"[autopilot] {name} ...", log_path)
        try:
            detail = fn()
            ok = True
        except Exception as e:
            detail, ok = f"{e!r}\n{traceback.format_exc()[-700:]}", False
        steps.append({"step": name, "ok": ok, "seconds": round(time.time() - t0, 1), "detail": detail})
        log(f"[autopilot] {name}: {'ok' if ok else 'FAILED'} ({time.time() - t0:.0f}s) {str(detail)[:300] if not ok else ''}", log_path)
        return ok

    def fail():
        return {"ok": False, "steps": steps, "plan": None, "pid": None}

    def environment():
        probe = probe_environment(work)
        probs = diagnose_environment(probe, need_cuda)
        if any(p["fixable"] for p in probs):
            rep = ensure_environment(probe, log=lambda m: log(m, log_path), need_cuda=need_cuda)
            probe, probs = rep["probe"], rep["problems"]
        state["probe"] = probe
        errors = [p for p in probs if p["level"] == "error"]
        if errors:
            raise RuntimeError("; ".join(p["msg"] for p in errors))
        return {"gpu": probe.get("gpu"), "gpu_gb": probe.get("gpu_gb"), "cpus": probe.get("cpus"), "ram_gb": probe.get("ram_gb"), "disk_free_gb": probe.get("disk_free_gb"),
                "torch": probe["modules"].get("torch"), "ultralytics": probe["modules"].get("ultralytics"), "warnings": [p["msg"] for p in probs if p["level"] == "warn"]}

    if not step("environment check / repair", environment):
        return fail()
    sizes = (320,) if limit else (640, int(iso_size))   # test mode: tiny images, no resolution baseline

    def data():
        prepare_all_data(work, imgszs=sizes, limit=limit, log_path=log_path)
        return f"VisDrone splits prepared at {list(sizes)} px" + (f" (first {limit} images only)" if limit else "")
    if not step("data (download, convert, pre-resize)", data):
        return fail()
    if limit is None:
        def analyses():
            for sp in ("train", "val"):
                jdump(dataset_census(work, sp), work / "analysis" / f"census_{sp}.json")
            ex = exactness_report(work)
            jdump(ex, work / "analysis" / "exactness.json")
            jdump(transfer_audit(work), work / "analysis" / "transfer_audit.json")
            jdump(assignability_table(work, "train"), work / "analysis" / "assignability.json")
            return {v: f"max feature diff {r['max_abs_feature_diff']:.1e}, max head diff {r['max_abs_head_output_diff']:.1e}" for v, r in ex.items()}
        if not step("static analyses (exactness, transfer audit, assignability)", analyses):
            return fail()
    if not skip_smoke:
        def smoke():
            lp = work / "logs" / "smoke.log"
            rc = run_fn_in_subprocess(str(nb), "smoke_test", {"work": str(work)}, lp, {"GRAFT_WORKDIR": str(work), "YOLO_CONFIG_DIR": str(work / "ultralytics_cfg")})
            res = last_result_json(lp) or []
            bad = [s for s in res if not s["ok"]]
            if rc != 0 or not res or bad:
                raise RuntimeError(f"smoke test failed (rc={rc}): " + "; ".join(f"{s['step']}: {str(s['detail'])[:200]}" for s in bad) + f"\n{tail_text(lp, 12)}")
            return f"{len(res)} checks passed in {sum(s['seconds'] for s in res):.0f}s"
        if not step("smoke test (tiny end-to-end run on this machine)", smoke):
            return fail()
    par = plan_parallel(state["probe"])
    spe = 60.0
    if limit is None and not skip_calibration:
        def calib():
            old = jload(work / "calibration.json", {})
            if old.get("s_per_epoch") and not recalibrate:  # a restarted autopilot must not repeat a 5-10 minute measurement
                state["calib"] = old
                return f"reused the earlier measurement: baseline {old.get('base', float('nan')):.0f} s/epoch, graft {old.get('graft', float('nan')):.0f} s/epoch (delete calibration.json to measure again)"
            lp = work / "logs" / "calibrate.log"
            cache = choose_cache(par["n_parallel"])
            rc = run_fn_in_subprocess(str(nb), "calibrate", {"work": str(work), "workers": par["workers"], "cache": cache}, lp,
                                      {"GRAFT_WORKDIR": str(work), "YOLO_CONFIG_DIR": str(work / "ultralytics_cfg")})
            c = jload(work / "calibration.json", {})
            if rc != 0 or not c:
                raise RuntimeError(f"calibration failed (rc={rc})\n{tail_text(lp, 12)}")
            state["calib"] = c
            return f"baseline {c.get('base', float('nan')):.0f} s/epoch, graft {c.get('graft', float('nan')):.0f} s/epoch"
        if not step("calibration (seconds per epoch on this machine)", calib):
            return fail()
        spe = float(state["calib"]["s_per_epoch"])
    elif limit is None:
        spe = float((jload(work / "calibration.json", {}) or {}).get("s_per_epoch", 60.0))
    seeds, tiers = tuple(seeds), tuple(tiers)
    template_specs = make_plan(40, seeds=seeds, tiers=tiers, iso_size=int(iso_size), families=tuple(families))
    if epochs:
        n_ep = int(epochs)
    elif limit is not None:
        n_ep = 1
    else:
        n_ep, _ = choose_epochs(budget_h * 3600, template_specs, spe, tiers=tuple(t for t in tiers if t <= 1) or (0,))
    specs = make_plan(n_ep, seeds=seeds, tiers=tiers, iso_size=int(iso_size), families=tuple(families))
    if limit is not None:
        specs = [dataclasses.replace(s, batch=8, imgsz=min(s.imgsz, 320), eval_sizes=()) for s in specs]
    cache = choose_cache(par["n_parallel"], need_gb=9.0 * (max(s.imgsz for s in specs) / 640.0) ** 2)
    plan = make_worker_plan(work, specs, par, cache, budget_h, spe, limit=limit, notebook_file=nb,
                            paper={"B": int(bootstrap), "iso_size": int(iso_size), "template": template})
    jdump(plan, work / "plan_preview.json")
    summary = f"{len(specs)} runs x {n_ep} epochs, {spe:.0f} s/epoch, {par['n_parallel']} parallel x {par['workers']} loader workers, cache={cache}"
    steps.append({"step": "plan", "ok": True, "seconds": 0.0, "detail": summary})

    def shared_inputs():
        """Everything the training processes would otherwise fetch/prepare themselves, possibly all at once: the pretrained checkpoints and the evaluation
        resolutions (e.g. 800 and 960 px) that are not prepared yet."""
        for fam in sorted({s.family for s in specs}):
            load_pretrained(work, fam)
        train_sizes = {s.imgsz for s in specs}
        extra = sorted(({s.imgsz for s in specs} | {e for s in specs for e in s.eval_sizes}) - set(sizes))
        for sz in extra:
            prepare_all_data(work, imgszs=[sz], splits=(("train", "val", "test-dev") if sz in train_sizes else ("val", "test-dev")), limit=limit, log_path=log_path)
        return f"pretrained checkpoints for {sorted({s.family for s in specs})}; extra image sizes prepared: {extra or 'none'}"
    if not step("shared inputs (checkpoints, extra evaluation sizes)", shared_inputs):
        return fail()
    if not launch:
        return {"ok": True, "steps": steps, "plan": plan, "pid": None}

    def go():
        w = jload(work / "worker.json", {})
        if w and pid_alive(w.get("pid", -1)):
            return f"a worker (pid {w['pid']}) is already running - not starting another"
        strays = stray_run_pids(work)
        if strays:
            raise RuntimeError(f"training processes of an earlier worker are still alive: {strays}")
        state["pid"] = launch_worker(work, plan, notebook_file=str(nb))
        return f"worker started, pid {state['pid']} ({summary})"
    if not step("launch the background worker", go):
        return fail()
    return {"ok": True, "steps": steps, "plan": plan, "pid": state.get("pid")}


@app.function
def zip_diagnostics(work, out_zip=None, include_predictions=False, max_log_bytes=400_000):
    """Everything needed to understand a run from the outside (statuses, logs - tails of the big ones -, per-run results and curves, analyses, the paper
    sources).  Predictions (hundreds of MB) only if asked."""
    work = Path(work)
    out_zip = Path(out_zip) if out_zip else work / "diagnostics.zip"
    files = [work / n for n in ("status.json", "worker.json", "plan.json", "plan_preview.json", "calibration.json")]
    files += sorted((work / "analysis").glob("*.json")) + sorted((work / "paper").glob("*.csv")) + sorted((work / "paper").glob("*.json")) + sorted((work / "paper").glob("*.tex"))
    for pat in ("result.json", "train_info.json", "results.csv", "args.yaml", "state.json", "external_*.json"):
        files += sorted((work / "runs").glob(f"*/{pat}"))
    if include_predictions:
        files += sorted((work / "runs").glob("*/*.npz"))
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            if p.is_file():
                z.write(p, p.relative_to(work))
        for p in sorted((work / "logs").glob("*")):
            if p.is_file():
                data = p.read_bytes()
                z.writestr(str(p.relative_to(work)), data if len(data) <= max_log_bytes else b"[... head cut ...]\n" + data[-max_log_bytes:])
        z.writestr("environment_probe.json", json.dumps(probe_environment(work), indent=1, default=str))
    return str(out_zip)


# <<<SECTION-J>>>


# =====================================================================================================
# NOTEBOOK CELLS
# =====================================================================================================
@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # P2-Graft — a controlled study of a function-preserving stride-4 level (YOLOv10n, VisDrone-DET2019)

    This notebook is the **whole study**: it downloads VisDrone, builds the model variants, trains every run
    in a detached background worker, evaluates from *saved predictions*, runs the statistics and **writes the LaTeX paper**
    (numbers, tables, figures and hypothesis verdicts are generated from the results — nothing is typed in by hand).

    **Quick path (molab, RTX PRO 6000):** attach the GPU (notebook-specs button in the header) → ① settings (the defaults are fine) → ② environment (press *Fix automatically* if it offers it) →
    ⓪ **Autopilot** (environment → data → analyses → smoke test → calibration → launch; ≈ 45–60 min, then the training runs in the background) → watch ⑧ →
    when the worker has finished, ⑩ **download the paper bundle** and ⑩b **the diagnostics zip**.

    **Numbered path:** ① settings → ② environment → ③ data → ④ static analyses → ⑤ *(optional)* smoke test → ⑥ calibrate + plan → ⑦ launch →
    ⑧ monitor → ⑧b *(optional)* zero-shot check on another dataset → ⑨ build the paper → ⑩ download.  The worker builds the paper by itself when it finishes. Re-launching is safe: finished runs are skipped,
    a partly trained run resumes from its last checkpoint.

    > **molab facts that matter.** Only files uploaded through molab's file browser survive a shutdown — **everything this notebook writes is lost when the session ends** (12 h maximum, or 90 minutes without a connected
    > browser). Keep this tab open (and your computer awake) while the worker runs, leave the ⑧ auto-refresh on, and **download ⑩ / ⑩b before the session ends** (⑧ also offers the results so far as CSV).
    >
    > **Honesty note.** Hypotheses are fixed in advance (H1–H4 in the paper). If a result is negative or inconclusive, the generated text says so.
    """)
    return


@app.cell
def _(mo):
    work_input = mo.ui.text(value=str(default_work_dir()), label="Work directory (data, runs, results, paper)", full_width=True)
    budget_h = mo.ui.slider(2.0, 10.0, value=8.5, step=0.5, label="Budget for training + evaluation (hours, counted from the launch; the 12 h session also has to fit ~1 h of set-up before it and ~0.7 h of paper building after it)", show_value=True)
    seeds_ui = mo.ui.multiselect(options=["0", "1", "2", "3", "4"], value=["0", "1", "2"], label="Seeds of the main contrast")
    tiers_ui = mo.ui.multiselect(options=["0", "1", "2", "3"], value=["0", "1", "2"],
                                 label="Run tiers (0 = headline pair x seeds, 1 = one seed of: compute-matched, naive, classic-TAL, P5-removal, random-init, 2 = more seeds, 3 = YOLO26 family)")
    epochs_ui = mo.ui.number(0, 300, step=5, value=0, label="Epochs per run (0 = choose automatically from the budget)")
    iso_ui = mo.ui.dropdown(options=["800", "896", "960"], value="896", label="Compute-matched resolution baseline (px)")
    boot_ui = mo.ui.slider(50, 1000, value=300, step=50, label="Bootstrap resamples", show_value=True)
    template_ui = mo.ui.dropdown(options=["lncs", "article"], value="lncs",
                                 label="Paper template (lncs = Springer LNCS, the ICPR/ICPRv format; article = self-contained two-column article)")
    mo.vstack([mo.md("## ① Settings"), work_input, budget_h, mo.hstack([seeds_ui, tiers_ui], justify="start"), mo.hstack([epochs_ui, iso_ui, boot_ui], justify="start"), template_ui])
    return boot_ui, budget_h, epochs_ui, iso_ui, seeds_ui, template_ui, tiers_ui, work_input


@app.cell
def _(mo):
    auto_ok = mo.ui.checkbox(value=True, label="Start automatically when the notebook is run (Run all). I understand: this downloads ≈ 2 GB, runs ≈ 30–60 min of checks and then starts a long background training job with the budget set in ①")
    auto_btn = mo.ui.run_button(label="⓪ AUTOPILOT: environment → data → analyses → smoke test → calibration → launch")
    mo.vstack([mo.md("## ⓪ Autopilot (recommended)\nRuns the numbered steps ② – ⑦ in one go and stops at the first failure with the reason. Idempotent: if something fails, fix it and press it again — "
                     "finished parts are skipped. While it runs the notebook is busy (progress goes to the console below and to `logs/autopilot.log`)."), auto_ok, auto_btn])
    return auto_btn, auto_ok


@app.cell
def _(mo):
    recheck_btn = mo.ui.run_button(label="Re-check the environment")
    return (recheck_btn,)


@app.cell
def _(mo, recheck_btn, work_input):
    recheck_btn  # pressing the button re-runs this cell (a fresh probe)
    WORK = Path(work_input.value).expanduser()
    WORK.mkdir(parents=True, exist_ok=True)
    PROBE = probe_environment(WORK)
    PROBLEMS = diagnose_environment(PROBE)
    PAR = plan_parallel(PROBE)
    _rows = "\n".join(f"| {k} | {v} |" for k, v in PROBE.items() if k not in ("modules", "errors", "torch_arch_list"))
    _mods = ", ".join(f"`{k}` {v}" if v else f"**`{k}` MISSING**" for k, v in (PROBE.get("modules") or {}).items())
    _issues = [mo.callout(mo.md(("**Error:** " if p["level"] == "error" else "**Warning:** ") + p["msg"]), kind="danger" if p["level"] == "error" else "warn") for p in PROBLEMS]
    fix_btn = mo.ui.run_button(label="Fix automatically: install missing packages / headless OpenCV / pinned Ultralytics")
    torch_btn = mo.ui.run_button(label="Install PyTorch for CUDA 12.8 (≈ 3 GB, 3–6 min; restart the kernel afterwards)")
    mo.vstack([mo.md("## ② Environment"), mo.md("| item | value |\n|---|---|\n" + _rows), mo.md("Packages: " + _mods),
               mo.md(f"Parallel plan: **{PAR['n_parallel']}** training process(es) × **{PAR['workers']}** data-loader workers "
                     f"(chosen from {PROBE.get('cpus')} usable CPUs and {PROBE.get('gpu_gb')} GB of GPU memory; the data pipeline, not the GPU, is the bottleneck for nano models)."),
               *_issues, (fix_btn if any(p["fixable"] for p in PROBLEMS) else mo.md("*Nothing to fix automatically.*" if PROBLEMS else "*Environment OK.*")),
               (torch_btn if any(p["code"] in ("no_cuda_driver", "arch") for p in PROBLEMS) else mo.md("")), recheck_btn])
    return PAR, PROBE, PROBLEMS, WORK, fix_btn, torch_btn


@app.cell
def _(mo, torch_btn):
    mo.stop(not torch_btn.value, mo.md("*PyTorch has not been touched.*"))
    _log = []
    _rc = install_cuda_torch("cu128", log=lambda m: _log.append(m))
    mo.vstack([mo.callout(mo.md(f"Installer finished with exit code {_rc}. **Restart the kernel now** (menu ▸ Restart), then press *Re-check the environment*."),
                          kind="success" if _rc == 0 else "danger"),
               mo.accordion({"installer output": mo.md("```\n" + "\n".join(_log)[-5000:] + "\n```")})])
    return


@app.cell
def _(PROBE, fix_btn, mo):
    mo.stop(not fix_btn.value, mo.md("*Nothing has been changed.*"))
    _log = []
    _rep = ensure_environment(PROBE, log=lambda m: _log.append(m))
    mo.vstack([mo.md("### Repair report\n" + ("\n".join(f"- {a}" for a in _rep["actions"]) or "- (nothing to do)")),
               mo.accordion({"installer output": mo.md("```\n" + "\n".join(_log)[-5000:] + "\n```")}),
               mo.md("**Remaining problems:** " + ("none" if not _rep["problems"] else "; ".join(p["msg"] for p in _rep["problems"]))),
               mo.callout(mo.md("Press **Re-check the environment** (②). If PyTorch or Ultralytics were changed after this notebook had imported them, restart the kernel first "
                                "(it is safe: files and the background worker are untouched)."), kind="info")])
    return


@app.cell
def _(WORK, auto_btn, auto_ok, boot_ui, budget_h, epochs_ui, iso_ui, mo, seeds_ui, template_ui, tiers_ui):
    mo.stop(not (auto_ok.value or auto_btn.value), mo.md("*Autopilot not started. Tick the box and press the button — or use the numbered steps below.*"))
    _nb = NOTEBOOK_FILE or Path(mo.notebook_location() / "graft_p2_study.py")
    AUTO = autopilot(WORK, budget_h=float(budget_h.value), seeds=tuple(int(s) for s in seeds_ui.value) or (0, 1, 2), tiers=tuple(int(t) for t in tiers_ui.value) or (0, 1),
                     epochs=int(epochs_ui.value or 0), iso_size=int(iso_ui.value), notebook_file=str(_nb), template=template_ui.value, bootstrap=int(boot_ui.value),
                     log_path=WORK / "logs" / "autopilot.log")
    _lines = [f"- {'✅' if s['ok'] else '❌'} **{s['step']}** ({s['seconds']} s): `{str(s['detail'])[:600]}`" for s in AUTO["steps"]]
    mo.vstack([mo.callout(mo.md("**Autopilot finished: the worker is running — follow it in ⑧.**" if AUTO["ok"] else "**Autopilot stopped at a failed step** — read the last line, fix it, press the button again."),
                          kind="success" if AUTO["ok"] else "danger"), mo.md("\n".join(_lines))])
    return (AUTO,)


@app.cell
def _(mo):
    data_btn = mo.ui.run_button(label="Download + prepare VisDrone-DET2019 (≈ 1.9 GB, 10–25 min)")
    mo.vstack([mo.md("## ③ Data"), mo.md("Downloads the three official splits (Ultralytics' GitHub mirror), converts them to YOLO format, pre-resizes the training images once and builds "
                                          "COCO ground-truth JSON files in **original-resolution pixel coordinates** (with and without VisDrone ignore regions)."), data_btn])
    return (data_btn,)


@app.cell
def _(WORK, data_btn, iso_ui, mo):
    mo.stop(not data_btn.value, mo.md("*Press the button above to prepare the data (skipped automatically if already prepared).*"))
    _lp = WORK / "logs" / "notebook.log"
    prepare_all_data(WORK, imgszs=(640, int(iso_ui.value)), log_path=_lp)
    CENSUS = {sp: dataset_census(WORK, sp) for sp in ("train", "val")}
    for _sp, _c in CENSUS.items():
        jdump(_c, WORK / "analysis" / f"census_{_sp}.json")
    mo.md("**Data ready.**  \n" + "  \n".join(
        f"- {sp}: {c['images']} images, {c['objects']:,} objects, median box side {c['median_side_px']:.1f}px, "
        f"{100 * c['frac_area_lt_32sq']:.1f}% below 32², max {c['objects_per_image_max']} objects/image" for sp, c in CENSUS.items()))
    return (CENSUS,)


@app.cell
def _(mo):
    static_btn = mo.ui.run_button(label="Run static analyses (exactness test, transfer audit, assignability) — ≈ 1 min")
    mo.vstack([mo.md("## ④ Static analyses (no training)"), static_btn])
    return (static_btn,)


@app.cell
def _(WORK, mo, static_btn):
    mo.stop(not static_btn.value, mo.md("*Press the button above.*"))
    EXACT = exactness_report(WORK)
    jdump(EXACT, WORK / "analysis" / "exactness.json")
    AUDIT = transfer_audit(WORK)
    jdump(AUDIT, WORK / "analysis" / "transfer_audit.json")
    ASSIGN = assignability_table(WORK, "train")
    jdump(ASSIGN, WORK / "analysis" / "assignability.json")
    _t1 = pd.DataFrame([{"variant": r["variant"], "rule": r["mode"], "copied %": round(100 * r["frac_of_inheritable"], 1),
                          "role-aligned %": round(100 * r["frac_aligned_of_inheritable"], 1), "params train (M)": round(r["params_train_graph_M"], 2),
                          "params deploy (M)": round(r["params_deploy_fused_M"], 2), "GFLOPs @640": round(r["gflops_deploy"], 2)} for r in AUDIT["rows"]])
    _t2 = pd.DataFrame([{"levels": k.split("|")[0], "window floor": k.split("=")[1], "boxes with no candidate anchor %": round(100 * v["frac_unassigned"], 2),
                          "candidates / box": round(v["mean_candidates"], 1)} for k, v in ASSIGN.items() if "|floor=" in k])
    _fig = fig_assignability(ASSIGN, WORK / "paper" / "figs")
    fig_architecture(WORK / "paper" / "figs")
    mo.vstack([mo.md("**Exactness** (max abs. difference of the original levels, grafted vs. pretrained): "
                     + ", ".join(f"`{v}`: features {r['max_abs_feature_diff']:.1e}, raw head outputs {r['max_abs_head_output_diff']:.1e}" for v, r in EXACT.items())),
               mo.md("**Transfer audit**"), mo.ui.table(_t1, selection=None), mo.md("**Assignability** (VisDrone-train @ 640 px)"), mo.ui.table(_t2, selection=None),
               mo.image(src=str(WORK / "paper" / "figs" / "fig_assignability.png"), width=480),
               mo.image(src=str(WORK / "paper" / "figs" / "fig_architecture.png"), width=900)])
    return ASSIGN, AUDIT, EXACT


@app.cell
def _(mo):
    smoke_btn = mo.ui.run_button(label="Run the smoke test (≈ 5–8 min): tiny data, 1 epoch — verifies the whole pipeline on THIS machine")
    mo.vstack([mo.md("## ⑤ Smoke test (recommended before the long run)"),
               mo.md("Trains the baseline and the grafted model for one epoch on 48 images, evaluates them (COCO AP, TIDE), and checks that the bootstrap accumulation "
                     "reproduces the COCO evaluator exactly. The numbers are meaningless by design."), smoke_btn])
    return (smoke_btn,)


@app.cell
def _(WORK, mo, smoke_btn):
    mo.stop(not smoke_btn.value, mo.md("*Not run.*"))
    _nb = NOTEBOOK_FILE or Path(mo.notebook_location() / "graft_p2_study.py")
    _lp = WORK / "logs" / "smoke.log"
    _rc = run_fn_in_subprocess(str(_nb), "smoke_test", {"work": str(WORK)}, _lp, {"GRAFT_WORKDIR": str(WORK), "YOLO_CONFIG_DIR": str(WORK / "ultralytics_cfg")})
    _res = last_result_json(_lp) or []
    _lines = [f"- {'✅' if s['ok'] else '❌'} **{s['step']}** ({s['seconds']}s): `{json.dumps(s['detail'], default=str)[:300]}`" for s in _res]
    mo.vstack([mo.md(f"Smoke test finished with return code {_rc}."), mo.md("\n".join(_lines) if _lines else "No result — see the log below."),
               mo.accordion({"log tail": mo.md("```\n" + tail_text(_lp, 40) + "\n```")})])
    return


@app.cell
def _(mo):
    calib_btn = mo.ui.run_button(label="Calibrate speed (≈ 4–8 min) and build the run plan")
    mo.vstack([mo.md("## ⑥ Calibrate + plan"),
               mo.md("Trains the baseline and the grafted model for 3 epochs on 10 % of the training set to measure seconds/epoch on **this** machine, "
                     "then chooses the largest epoch count for which all tier-0/1 runs fit the budget."), calib_btn])
    return (calib_btn,)


@app.cell
def _(PAR, WORK, calib_btn, mo):
    _msg = ""
    if calib_btn.value:
        _nb = NOTEBOOK_FILE or Path(mo.notebook_location() / "graft_p2_study.py")
        _lp = WORK / "logs" / "calibrate.log"
        prepare_all_data(WORK, imgszs=(640,), splits=("train", "val"), log_path=_lp)
        _cache = choose_cache(PAR["n_parallel"])
        _rc = run_fn_in_subprocess(str(_nb), "calibrate", {"work": str(WORK), "workers": PAR["workers"], "cache": _cache}, _lp,
                                   {"GRAFT_WORKDIR": str(WORK), "YOLO_CONFIG_DIR": str(WORK / "ultralytics_cfg")})
        _msg = f"Calibration finished (rc={_rc}). "
    CALIB = jload(WORK / "calibration.json", {})
    if CALIB:
        _msg += (f"Baseline **{CALIB.get('base', float('nan')):.0f} s/epoch**, graft **{CALIB.get('graft', float('nan')):.0f} s/epoch** "
                 f"(start-up overhead ≈ {CALIB.get('base_startup_s', float('nan')):.0f} s per run).")
    else:
        _msg += "No calibration yet - a default of 60 s/epoch is assumed until you calibrate."
    mo.md(_msg)
    return (CALIB,)


@app.cell
def _(CALIB, PAR, WORK, boot_ui, budget_h, epochs_ui, iso_ui, mo, seeds_ui, tiers_ui):
    _seeds = tuple(int(s) for s in seeds_ui.value) or (0, 1, 2)
    _tiers = tuple(int(t) for t in tiers_ui.value) or (0, 1)
    _spe = float(CALIB.get("s_per_epoch", float("nan"))) if CALIB else float("nan")
    _spe = _spe if _spe == _spe else (jload(WORK / "calibration.json", {}) or {}).get("s_per_epoch", 60.0)
    _template = make_plan(40, seeds=_seeds, tiers=_tiers, iso_size=int(iso_ui.value), families=("v10", "yolo26"))
    _auto, _info = choose_epochs(budget_h.value * 3600, _template, _spe, tiers=tuple(t for t in _tiers if t <= 1) or (0,))
    EPOCHS = int(epochs_ui.value) if epochs_ui.value else _auto
    PLAN = make_plan(EPOCHS, seeds=_seeds, tiers=_tiers, iso_size=int(iso_ui.value), families=("v10", "yolo26"))
    _tot = sum(estimate_run_seconds(s, _spe) for s in PLAN)
    _tab = pd.DataFrame([{"order": i + 1, "run": s.run_id, "tier": s.tier, "imgsz": s.imgsz, "epochs": s.epochs, "est. hours": round(estimate_run_seconds(s, _spe) / 3600, 2)}
                         for i, s in enumerate(PLAN)])
    mo.vstack([mo.md(f"## Plan: **{EPOCHS} epochs** per run, {len(PLAN)} runs, est. total {_tot / 3600:.1f} h of sequential work "
                     f"({_tot / 3600 / PAR['n_parallel']:.1f} h wall-clock with {PAR['n_parallel']} parallel process(es)); budget {budget_h.value:.1f} h. "
                     "Runs that cannot finish inside the budget are skipped from the end of the queue."),
               mo.ui.table(_tab, selection=None, page_size=25)])
    return EPOCHS, PLAN


@app.cell
def _(mo):
    confirm = mo.ui.checkbox(label="I understand that this starts a long background job on this machine")
    launch_btn = mo.ui.run_button(label="⑦ Launch the background worker")
    mo.vstack([mo.md("## ⑦ Launch"), confirm, launch_btn])
    return confirm, launch_btn


@app.cell
def _(CALIB, PAR, PLAN, WORK, boot_ui, budget_h, confirm, iso_ui, launch_btn, mo, template_ui):
    mo.stop(not (launch_btn.value and confirm.value), mo.md("*Tick the box and press the button to launch. Nothing is running yet.*"))
    _nb = NOTEBOOK_FILE or Path(mo.notebook_location() / "graft_p2_study.py")
    _w = jload(WORK / "worker.json", {})
    if _w and pid_alive(_w.get("pid", -1)):
        mo.stop(True, mo.callout(mo.md(f"A worker (pid {_w['pid']}) is already running. Use the monitor below."), kind="info"))
    _strays = stray_run_pids(WORK)
    if _strays:
        mo.stop(True, mo.callout(mo.md(f"Training processes from an earlier worker are still alive: {_strays}. Wait for them to finish (or stop them) before launching."), kind="danger"))
    _cache = choose_cache(PAR["n_parallel"], need_gb=9.0 * (max(sp_.imgsz for sp_ in PLAN) / 640.0) ** 2)
    _plan = make_worker_plan(WORK, PLAN, PAR, _cache, float(budget_h.value), float((CALIB or jload(WORK / "calibration.json", {})).get("s_per_epoch", 60.0)),
                             paper={"B": int(boot_ui.value), "iso_size": int(iso_ui.value), "template": template_ui.value})
    PID = launch_worker(WORK, _plan, notebook_file=str(_nb))
    mo.callout(mo.md(f"Worker started (pid **{PID}**, cache={_cache}). It keeps running if you close this tab (but see the molab note at the top). Progress: step ⑧."), kind="success")
    return (PID,)


@app.cell
def _(mo):
    refresh = mo.ui.refresh(options=["15s", "30s", "1m", "5m"], default_interval="30s", label="Auto-refresh the monitor")
    mo.vstack([mo.md("## ⑧ Monitor"), refresh])
    return (refresh,)


@app.cell
def _(WORK, mo, refresh):
    refresh
    _st = jload(WORK / "status.json", {})
    _w = jload(WORK / "worker.json", {})
    if not _st:
        mo.stop(True, mo.md("*No worker status yet.*"))
    _runs = _st.get("runs", {})
    _now = time.time()
    _rows = []
    for rid, r in _runs.items():
        el = (_now - r["started"]) if r.get("state") == "running" and r.get("started") else r.get("seconds")
        _rows.append({"run": rid, "tier": r.get("tier"), "state": r.get("state"), "elapsed": fmt_hours(el) if el else "", "estimate": fmt_hours(r.get("est_s", 0)) if r.get("est_s") else ""})
    _df = pd.DataFrame(_rows)
    _done = int((_df.state == "done").sum()) if len(_df) else 0
    _alive = pid_alive(_w.get("pid", -1)) if _w else False
    mo.vstack([mo.md(f"**Worker {'running' if _alive else 'not running'}** · {_done}/{len(_df)} runs done · time left in budget: {fmt_hours(_st.get('deadline', _now) - _now)}"),
               mo.ui.table(_df, selection=None, page_size=30), mo.accordion({"worker log (tail)": mo.md("```\n" + tail_text(WORK / "logs" / "worker.log", 30) + "\n```")})])
    return


@app.cell
def _(WORK, mo, refresh):
    refresh
    _df = collect_results(WORK)
    if _df.empty:
        mo.stop(True, mo.md("*No finished run yet.*"))
    _cols = [c for c in ("run_id", "val_AP50", "val_AP", "val_APs", "test_AP", "gflops", "latency_ms", "train_h") if c in _df.columns]
    _csv = _df.drop(columns=[c for c in _df.columns if c.endswith("_perclass_AP50") or c.endswith("_perclass_AP")], errors="ignore").to_csv(index=False).encode("utf-8")
    mo.vstack([mo.md("### Finished runs so far (VisDrone-val unless noted)"), mo.ui.table(_df[_cols].round(2), selection=None, page_size=30),
               mo.download(data=_csv, filename="all_runs_so_far.csv", mimetype="text/csv", label="Download the results so far (CSV, small) — a cheap backup")])
    return


@app.cell
def _(mo):
    keep_auto = mo.ui.checkbox(value=True, label="Keep the session busy automatically until the worker has finished (Run all will then stay running)")
    keep_btn = mo.ui.run_button(label="Keep this session busy until the worker has finished (blocks the notebook; press the stop button to interrupt)")
    mo.vstack([keep_auto, mo.md("### Keep-alive (optional)\nmolab shuts a notebook down after 90 minutes without activity. The auto-refresh above keeps an open tab active; if you will close the tab, "
                     "this loop keeps the kernel itself busy while the worker trains. Whether molab counts it as activity is not documented — keeping the tab open is the safe option."), keep_btn])
    return keep_auto, keep_btn


@app.cell
def _(AUTO, WORK, keep_auto, keep_btn, mo):
    mo.stop(not (keep_auto.value or keep_btn.value), mo.md("*Keep-alive not running.*"))
    _t0 = time.time()
    while True:
        _w = jload(WORK / "worker.json", {})
        if not (_w and pid_alive(_w.get("pid", -1))):
            break
        time.sleep(30)
        if int((time.time() - _t0) // 600) != int((time.time() - _t0 - 30) // 600):
            print(f"[keep-alive] worker still running after {fmt_hours(time.time() - _t0)}", flush=True)
    mo.md("The worker has finished (or is not running).")
    return


@app.cell
def _(mo):
    ext_dir = mo.ui.text(value="", label="Folder of another dataset in YOLO format (contains images/ and labels/)", full_width=True)
    ext_name = mo.ui.text(value="", label="Dataset name (appears in the paper)")
    ext_classes = mo.ui.text(value="car, truck, bus", label="Its class names, in class-index order (comma-separated)", full_width=True)
    ext_map = mo.ui.text(value="3:0, 5:1, 8:2", full_width=True,
                         label="Class map  VisDrone class index : class index in that dataset  (VisDrone: 0 pedestrian, 1 people, 2 bicycle, 3 car, 4 van, 5 truck, 6 tricycle, 7 awning-tricycle, 8 bus, 9 motor)")
    ext_btn = mo.ui.run_button(label="⑧b Zero-shot evaluation of the finished baseline / P2-Graft runs on this dataset")
    mo.vstack([mo.md("## ⑧b Optional: zero-shot check on another dataset"),
               mo.md("Not needed for the main study. Cheap (minutes): the finished main-pair runs are applied **without any training** to a dataset you provide, "
                     "through an explicit class map; the paper then gets one extra paragraph. You supply and convert the dataset yourself (this notebook does not download it), "
                     "and you are responsible for its licence. Leave the folder empty to skip."),
               ext_dir, mo.hstack([ext_name, ext_classes], justify="start"), ext_map, ext_btn])
    return ext_btn, ext_classes, ext_dir, ext_map, ext_name


@app.cell
def _(WORK, ext_btn, ext_classes, ext_dir, ext_map, ext_name, iso_ui, mo):
    mo.stop(not (ext_btn.value and ext_dir.value.strip()), mo.md("*Optional step not run.*"))
    _names = [s.strip() for s in ext_classes.value.split(",") if s.strip()]
    try:
        _done = run_external_check(WORK, ext_dir.value, ext_name.value.strip() or "other dataset", _names, ext_map.value, log_path=WORK / "logs" / "external.log")
        _out = mo.callout(mo.md(f"Evaluated {len(_done)} runs: {', '.join(_done) or '(none finished yet)'}. Rebuild the paper (⑨) to include the paragraph."), kind="success")
    except Exception as _e:
        _out = mo.callout(mo.md(f"Could not run the check: `{type(_e).__name__}: {_e}`"), kind="danger")
    _out
    return


@app.cell
def _(mo):
    build_btn = mo.ui.run_button(label="⑨ Build the paper from the finished runs (bootstrap, TIDE tables, figures, LaTeX)")
    mo.vstack([mo.md("## ⑨ Analysis + paper"),
               mo.md("Can be run at any time on the runs that have finished (partial results give a partial paper, clearly marked by missing sections). "
                     "Takes a few minutes (image-level bootstrap)."), build_btn])
    return (build_btn,)


@app.cell
def _(WORK, boot_ui, build_btn, iso_ui, mo, template_ui):
    mo.stop(not build_btn.value, mo.md("*Not built yet.*"))
    PAPER = build_paper(WORK, B=int(boot_ui.value), iso_size=int(iso_ui.value), template=template_ui.value, log_path=WORK / "logs" / "analysis.log")
    _A, _C = PAPER["A"], PAPER["C"]
    _v = "\n".join(f"- **{h}**: {verdict_word(v[0])} — {v[1]}" for h, v in sorted(_C["verdicts"].items())) or "- (no hypothesis testable yet)"
    _s = "\n\n".join(f"{k}: {v}" for k, v in _C["sentences"].items())
    _imgs = [mo.image(src=str(Path(PAPER["out_dir"]) / "figs" / f"{n}.png"), width=760) for n in ("fig_main", "fig_factorial", "fig_pareto", "fig_tide", "fig_perclass")
             if (Path(PAPER["out_dir"]) / "figs" / f"{n}.png").exists()]
    mo.vstack([mo.md(f"### Verdicts\n{_v}"), mo.md("### Generated sentences (LaTeX source)\n```\n" + _s + "\n```"), mo.md("### Condition summary"),
               mo.ui.table(_A["agg"][[c for c in ("group", "n", "val_AP50", "val_AP", "val_APs", "gflops", "latency_ms", "test_AP") if c in _A["agg"].columns]].round(2), selection=None),
               *_imgs])
    return (PAPER,)


@app.cell
def _(WORK, mo, refresh):
    refresh  # re-read the disk periodically: the worker builds the paper by itself when it finishes
    _zip = WORK / "paper_bundle.zip"
    _tex = WORK / "paper" / "main.tex"
    if not (_zip.exists() and _tex.exists()):
        mo.stop(True, mo.md("## ⑩ Download\n*No paper bundle yet — the worker builds it when it finishes; ⑨ builds it from the runs finished so far.*"))
    mo.vstack([mo.md("## ⑩ Download"),
               mo.md(f"Paper bundle `{_zip.name}` (built {time.strftime('%Y-%m-%d %H:%M', time.localtime(_zip.stat().st_mtime))}) — contains `main.tex`, `references.bib`, `figs/`, `all_runs.csv`, "
                     "`conditions_summary.csv`, `results_summary.json`. With the `lncs` template add Springer's `llncs.cls` and `splncs04.bst` next to `main.tex`, then "
                     "`pdflatex main && bibtex main && pdflatex main && pdflatex main` (e.g. on Overleaf). **Author block, code URL and every reference must be checked by the authors.**"),
               mo.hstack([mo.download(data=_zip.read_bytes(), filename="paper_bundle.zip", mimetype="application/zip", label="Download paper bundle"),
                          mo.download(data=_tex.read_bytes(), filename="main.tex", mimetype="text/plain", label="Download main.tex")], justify="start")])
    return


@app.cell
def _(mo):
    diag_pred = mo.ui.checkbox(label="include the saved predictions (hundreds of MB; only needed to recompute the bootstrap elsewhere)")
    diag_btn = mo.ui.run_button(label="⑩b Build the diagnostics zip (statuses, logs, per-run results, analyses)")
    mo.vstack([mo.md("## ⑩b Diagnostics — download this if anything looks wrong (and as a backup)"), diag_pred, diag_btn])
    return diag_btn, diag_pred


@app.cell
def _(WORK, diag_btn, diag_pred, mo):
    mo.stop(not diag_btn.value, mo.md("*Not built yet.*"))
    _z = Path(zip_diagnostics(WORK, include_predictions=bool(diag_pred.value)))
    mo.vstack([mo.md(f"`{_z.name}`: {_z.stat().st_size / 1e6:.1f} MB"),
               mo.download(data=_z.read_bytes(), filename="diagnostics.zip", mimetype="application/zip", label="Download diagnostics.zip")])
    return


if __name__ == "__main__":
    app.run()
