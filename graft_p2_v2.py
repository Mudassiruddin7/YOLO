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
#     "huggingface_hub",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="P2-Graft v2: confirmatory runs")

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

    STUDY_VERSION = "2.0.0"
    ULTRALYTICS_PIN = "8.4.150"
    FAMILY_SCALE = {"v10": "n", "v10s": "s"}  # v10 = YOLOv10n (the main scale), v10s = YOLOv10s (the second scale)
    FAMILY_CKPT = {"v10": "yolov10n.pt", "v10s": "yolov10s.pt"}
    try:  # path of this notebook file (needed to start the detached worker); marimo provides __file__
        NOTEBOOK_FILE = Path(__file__).resolve()
    except NameError:  # pragma: no cover
        NOTEBOOK_FILE = None

    HF_PUSHED = {}  # run id -> time of the last checkpoint push of this process
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
def load_base_cfg(family="v10"):
    """Nano- (family v10) or small-scale (v10s) YAML of the base detector as a python dict (taken from the installed Ultralytics package)."""
    import yaml
    import ultralytics
    scale = FAMILY_SCALE[family]
    f = Path(ultralytics.__file__).parent / "cfg" / "models" / "v10" / f"yolov10{scale}.yaml"
    cfg = yaml.safe_load(open(f, encoding="utf-8"))
    cfg["scales"] = {scale: cfg["scales"][scale]}  # keep only this scale: no filename-based scale guessing
    cfg["scale"] = scale
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
    graftd  : graft with a twice as deep stride-4 neck block.
    injC    : no stride-4 level; the P3 head also sees C channels of a strided conv of the stride-4 backbone feature (detail injection).
    assistC : no stride-4 level; the stride-4 neck block is computed and C channels of its strided conv go into the P3 head (P2-assisted P3).
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
    m_gd = re.fullmatch(r"graftd(\d*)", variant)
    if m_gd:  # graft with a 2x (graftd) or k x (graftd<k>) deeper stride-4 neck block
        p2_block[1] = int(p2_block[1]) * int(m_gd.group(1) or 2)
        variant = "graft"
    m_inj = re.fullmatch(r"inj(\d+)", variant)
    if m_inj:  # detail injection: the P3 head additionally sees a strided conv of the stride-4 backbone feature (new input channels, zero weights at start)
        c_arg = int(m_inj.group(1)) * 4  # yaml width of the nano scale is 0.25
        cfg["head"] = body + [[p2_backbone_idx, 1, "Conv", [c_arg, 3, 2]], [[p3_idx, -1], 1, "Concat", [1]],
                              [[n_before + 1, p4_idx, p5_idx], 1, det[2], det[3]]]
        return cfg
    m_as = re.fullmatch(r"assist(\d+)", variant)
    if m_as:  # P2-assisted P3: the stride-4 neck block (no stride-4 prediction level) is folded into the P3 head input
        c_arg = int(m_as.group(1)) * 4
        cfg["head"] = body + [up_spec, cat_spec, p2_block, [-1, 1, "Conv", [c_arg, 3, 2]], [[p3_idx, -1], 1, "Concat", [1]],
                              [[n_before + 4, p4_idx, p5_idx], 1, det[2], det[3]]]
        return cfg
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
    p = Path(work) / "cfg" / f"{family}-{variant}.yaml"
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
    if variant == "graft" or re.fullmatch(r"graftd\d*", variant) or re.fullmatch(r"(inj|assist)\d+", variant):
        det_new = n_base_layers + (3 if variant.startswith("graft") else (2 if variant.startswith("inj") else 5))

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
def warm_start_p2_entries(csd, msd, variant, n_base_layers=23, p3_block_idx=16):
    """Extra checkpoint entries that initialise the NEW stride-4 level of a graft from the pretrained P3 level instead of at random
    (the P3 neck block -> the P2 neck block where the shapes agree, the level-0 detection convs -> the level-3 ones).  The original levels are not touched,
    so the function-preserving property of the graft is unchanged."""
    det_new = n_base_layers + 3
    blk_new = n_base_layers + 2
    extra: Dict[str, Any] = {}
    for k, v in csd.items():
        if k.startswith(f"model.{p3_block_idx}."):
            tail = k[len(f"model.{p3_block_idx}."):]
            cands = [f"model.{blk_new}.{tail}"]
            m_gd = re.fullmatch(r"graftd(\d*)", variant)
            if m_gd and tail.startswith("m.0."):  # the deeper block's further bottlenecks start as copies of the pretrained one
                cands += [f"model.{blk_new}.m.{j}.{tail[len('m.0.'):]}" for j in range(1, int(m_gd.group(1) or 2))]
            for nk in cands:
                if nk in msd and tuple(msd[nk].shape) == tuple(v.shape):
                    extra[nk] = v
        m = re.match(rf"model\.{det_new}\.(cv2|cv3|one2one_cv2|one2one_cv3)\.0\.(.*)", k)
        if m:
            nk = f"model.{det_new}.{m.group(1)}.3.{m.group(2)}"
            if nk in msd and tuple(msd[nk].shape) == tuple(v.shape):
                extra[nk] = v
    return extra


@app.function
def fit_state(msd, csd):
    """Entries of the checkpoint `csd` that fit the model state `msd`.  Tensors of equal shape are copied; a tensor that is LARGER in the model (the P3 head of an
    injection variant receives extra input channels) starts from the model's own initialisation with the pretrained values written into the leading block,
    and the weights of the extra INPUT channels of a convolution are set to zero, so that the extended network computes exactly what the pretrained one did.
    Returns (state to load, number of source parameters used)."""
    upd: Dict[str, Any] = {}
    n_src = 0
    for k, v in csd.items():
        if k not in msd:
            continue
        t = msd[k]
        if tuple(v.shape) == tuple(t.shape):
            upd[k] = v
            n_src += int(v.numel())
        elif v.dim() == t.dim() and v.dim() > 0 and all(a <= b for a, b in zip(v.shape, t.shape)):
            new = t.detach().clone()
            new[tuple(slice(0, a) for a in v.shape)] = v.to(new.dtype)
            if v.dim() == 4 and t.shape[1] > v.shape[1]:
                new[:, v.shape[1]:] = 0
            upd[k] = new
            n_src += int(v.numel())
    return upd, n_src


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
    ws_keys: set = set()
    if mode in ("role", "role_ws"):
        rename, drop = role_key_rename(variant)
        csd = {rename(k): v for k, v in pre_sd.items() if not drop(k)}
        if mode == "role_ws":
            ws = warm_start_p2_entries(csd, model.state_dict(), variant)
            ws_keys = set(ws)
            csd.update(ws)
        aligned_keys = set(csd)
    elif mode == "key":
        csd = dict(pre_sd)
        aligned_keys = None
    else:
        raise ValueError(mode)
    cls_remapped = model._remap_cls_by_names(csd, pre_model, verbose=False)
    if mode == "key":
        upd = intersect_dicts(csd, model.state_dict())
        n_src = int(sum(v.numel() for v in upd.values()))
    else:
        upd, n_src = fit_state(model.state_dict(), csd)
    model.load_state_dict(upd, strict=False)
    acc["cls_rows_remapped"] = int(cls_remapped)
    acc["tensors_transferred"] = len(upd) + int(cls_remapped)
    acc["params_transferred"] = n_src
    if ws_keys:
        acc["params_warm_started"] = int(sum(int(upd[k].numel()) for k in ws_keys if k in upd))
        acc["params_transferred"] = n_src - acc["params_warm_started"]
    if mode in ("role", "role_ws"):
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
    name = FAMILY_CKPT[family]
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
def exactness_report(work, family="v10", variants=("graft", "replace"), seed=0, modes=None):
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
        mode = (modes or {}).get(v, "role")
        acc = transfer_weights(m, pre, v, mode)
        m.eval()
        h = m.model[-1].register_forward_pre_hook(lambda mod, a: store.__setitem__("var", [t.clone() for t in a[0]]))
        with torch.no_grad():
            m(x)
        h.remove()
        det_p, det_v = pre.model[-1], m.model[-1]
        # level correspondence: graft keeps P3,P4,P5 as levels 0,1,2 ; replace keeps P3,P4 as levels 0,1
        n_cmp = 2 if v == "replace" else 3
        # (an injection variant feeds the P3 head extra input channels: compare the pretrained channels, the head comparison below uses them all)
        feat_diff = [float((store["pre"][i] - store["var"][i][:, :store["pre"][i].shape[1]]).abs().max()) for i in range(n_cmp)]
        raw = []
        with torch.no_grad():
            for i in range(n_cmp):
                for nm in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
                    raw.append(float((getattr(det_p, nm)[i](store["pre"][i]) - getattr(det_v, nm)[i](store["var"][i])).abs().max()))
        rep[v] = {"levels_compared": n_cmp, "max_abs_feature_diff": max(feat_diff), "max_abs_head_output_diff": max(raw),
                  "transfer": acc, "var_feature_shapes": [tuple(t.shape[1:]) for t in store["var"]]}
    return rep


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
def push_checkpoint(work, run_id, epoch, log_path=None, min_interval=600.0):
    """Called by the training process right after Ultralytics has SAVED a checkpoint (so the file is consistent): at most every `min_interval` seconds it copies last.pt (+ the
    training curve) to a staging folder, checks that the checkpoint is an intact zip archive and pushes it to the Hugging Face dataset folder of this session.  The environment
    carries the repo, the folder and the token (set by the autopilot).  Never raises: a failed push must not cost a training run."""
    repo, token, tag = os.environ.get("GRAFT_HF_REPO", ""), os.environ.get("HF_TOKEN", ""), os.environ.get("GRAFT_HF_TAG", "")
    if not (repo and token and tag):
        return False
    now = time.time()
    if now - HF_PUSHED.get(run_id, 0.0) < min_interval:
        return False
    HF_PUSHED[run_id] = now
    try:
        from huggingface_hub import HfApi
        rdir = Path(work) / "runs" / run_id
        stage = Path(work) / "_hf_stage" / run_id
        shutil.rmtree(stage, ignore_errors=True)
        (stage / "weights").mkdir(parents=True)
        shutil.copy2(rdir / "weights" / "last.pt", stage / "weights" / "last.pt")
        with zipfile.ZipFile(stage / "weights" / "last.pt") as z:
            if z.testzip() is not None:
                raise RuntimeError("checkpoint is not an intact archive")
        for f in ("results.csv", "args.yaml"):
            if (rdir / f).exists():
                shutil.copy2(rdir / f, stage / f)
        HfApi(token=token).upload_folder(folder_path=str(stage), repo_id=repo, repo_type="dataset", path_in_repo=f"{tag}/runs/{run_id}",
                                         commit_message=f"{run_id}: checkpoint after epoch {epoch}")
        shutil.rmtree(stage, ignore_errors=True)
        log(f"[{run_id}] checkpoint of epoch {epoch} pushed to Hugging Face", log_path)
        return True
    except Exception as e:
        HF_PUSHED[run_id] = now - min_interval + 120.0  # try again in 2 minutes
        log(f"[{run_id}] checkpoint push failed: {str(e).replace(token, '***')[:200]}", log_path)
        return False


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
    y.add_callback("on_model_save", lambda trainer: push_checkpoint(work, spec.run_id, int(trainer.epoch) + 1, log_path))
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
def latency_benchmark(weights, imgsz=640, iters=200, warmup=50, device=None, repeats=3):
    """Deploy-graph speed (fused, one-to-one head, FP16 on GPU, FP32 on CPU): batch-1 latency as the median over `repeats` blocks of `iters` forward passes (the spread
    between blocks is reported: at batch 1 these small models are launch-bound and a single block varies by +-0.2 ms) and the throughput at batch 16, which does depend on the compute."""
    import torch
    from ultralytics import YOLO
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
    m = YOLO(str(weights)).model.float().eval()
    m.end2end = True
    m.fuse()
    half = dev.type == "cuda"
    m = m.to(dev)
    if half:
        m = m.half()

    def timeit(batch, n, w):
        x = torch.rand(batch, 3, imgsz, imgsz, device=dev)
        x = x.half() if half else x
        ts = []
        with torch.no_grad():
            for i in range(w + n):
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t = time.perf_counter()
                m(x)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                if i >= w:
                    ts.append((time.perf_counter() - t) * 1000.0)
        return ts
    blocks_ms = [float(np.median(timeit(1, iters, warmup if i == 0 else 10))) for i in range(repeats)]
    b16 = float(np.median(timeit(16, 20, 5)))
    return {"latency_ms_median": float(np.median(blocks_ms)), "latency_ms_blocks": blocks_ms, "latency_ms_spread": float(max(blocks_ms) - min(blocks_ms)),
            "throughput_img_s_b16": 16000.0 / b16, "device": str(dev), "precision": "fp16" if half else "fp32", "imgsz": imgsz}


@app.function
def tag_suffix(limit):
    return "" if limit is None else f"_lim{limit}"


@app.function
def evaluate_run(work, spec, device=None, splits=("val", "test-dev"), tide=True, data_suffix="", limit=None,
                 log_path=None, skip_existing=True):
    """Post-training evaluation of one run: saved predictions + COCO metrics (confidence 0.001, 500 detections/image, ORIGINAL-resolution boxes) at the training size and at the
    larger test sizes of the spec, TIDE on validation, deploy-graph statistics and a latency probe.  (v1 also ran protocol-sensitivity variants, the one-to-many head with NMS,
    ignore regions and the Ultralytics validator; they cost several minutes per run and are not needed for the claims.)"""
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
            out["metrics"][key + "|std"] = coco_evaluate(work / "data" / f"gt_{split_dir_name(split)}{tag}.json", p)
            out["metrics"][key + "|std"]["predict_s"] = t_pred
            log(f"[{spec.run_id}] {split}@{size}: AP50={out['metrics'][key + '|std']['AP50']:.2f} AP={out['metrics'][key + '|std']['AP']:.2f}", log_path)
            jdump(out, res_path)
    if tide and "tide|val" not in out:
        try:
            p = load_preds(run / f"preds_val_{spec.imgsz}_o2o.npz")
            out["tide|val"] = tide_evaluate(work / "data" / f"gt_val{tag}.json", p)
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


@app.function
def spec_from_dict(d):
    d = dict(d)
    d["eval_sizes"] = tuple(d.get("eval_sizes", ()))
    return RunSpec(**d)


@app.function
def run_id_for(family, variant, tag, seed):
    return f"{family}-{variant}-{tag}-s{seed}"


@app.function
def leaderboard(work):
    """One row per condition (family, variant, transfer, input size, epochs): mean +- std over seeds; the TEST-DEV gain over the baseline of the same family, size and schedule, paired
    by seed (mean, spread, number of pairs, whether every pair has the same sign); and the distance to the baseline's resolution frontier (the baseline's AP interpolated at this model's
    GFLOPs, needs the baseline at >= 2 sizes with the same schedule).  Returns (DataFrame, markdown)."""
    import pandas as pd
    df = collect_results(Path(work))
    if df.empty:
        return pd.DataFrame(), "(no finished runs yet)"
    g = ["family", "variant", "transfer", "imgsz", "epochs"]
    cols = ["val_AP", "val_AP50", "val_APs", "test_AP", "test_AP50", "test_APs", "gflops", "params_deploy_M", "latency_ms", "throughput_b16", "train_h"]
    rows = []
    for key, d in df.groupby(g, dropna=False):
        row = dict(zip(g, key))
        row["label"] = str(d["group"].iloc[0])
        row["n"] = len(d)
        row["seeds"] = ",".join(str(int(x)) for x in sorted(d["seed"]))
        for c in cols:
            if c in d:
                row[c] = float(d[c].mean())
                row[c + "_std"] = float(d[c].std(ddof=1)) if len(d) > 1 else float("nan")
        for c in [x for x in d.columns if re.fullmatch(r"(val|test)_AP_at\d+", x)]:
            if d[c].notna().any():
                row[c] = float(d[c].mean())
        rows.append(row)
    lb = pd.DataFrame(rows)
    for c in ("d_test", "d_test_sd", "d_test_pairs", "d_test_same_sign", "d_val", "d_frontier_test"):
        lb[c] = float("nan")
    for i, r in lb.iterrows():
        sel = lambda v, t: df[(df.family == r["family"]) & (df.variant == v) & (df.transfer == t) & (df.imgsz == r["imgsz"]) & (df.epochs == r["epochs"])]
        b, d = sel("base", "role"), sel(r["variant"], r["transfer"])
        if b.empty or r["variant"] == "base":
            continue
        for m, name in (("test_AP", "d_test"), ("val_AP", "d_val")):
            common = sorted(set(b["seed"]) & set(d["seed"]))
            if common:
                diff = [float(d[d.seed == s_][m].iloc[0]) - float(b[b.seed == s_][m].iloc[0]) for s_ in common]
                lb.loc[i, name] = float(np.mean(diff))
                if name == "d_test":
                    lb.loc[i, "d_test_sd"] = float(np.std(diff, ddof=1)) if len(diff) > 1 else float("nan")
                    lb.loc[i, "d_test_pairs"] = len(diff)
                    lb.loc[i, "d_test_same_sign"] = float(all(x > 0 for x in diff) or all(x < 0 for x in diff))
            else:
                lb.loc[i, name] = float(d[m].mean() - b[m].mean())
    for (fam_, ep_), grp in lb.groupby(["family", "epochs"]):
        pts = grp[(grp.variant == "base") & (grp.transfer == "role")].sort_values("gflops")
        if len(pts) < 2:
            continue
        xs = np.log(pts["gflops"].to_numpy(dtype=float))
        ys = pts["test_AP"].to_numpy(dtype=float)
        for i, r in grp.iterrows():
            if isnum(r.get("gflops")) and isnum(r.get("test_AP")):
                x = float(np.log(r["gflops"]))
                k = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
                lb.loc[i, "d_frontier_test"] = r["test_AP"] - float(ys[k] + (ys[k + 1] - ys[k]) * (x - xs[k]) / (xs[k + 1] - xs[k]))
    lb = lb.sort_values(["family", "imgsz", "epochs", "test_AP"], ascending=[True, True, True, False]).reset_index(drop=True)
    head = ["label", "family", "imgsz", "ep", "n", "test AP", "d test (sd; pairs; same sign)", "d frontier", "val AP", "d val", "test AP50", "GFLOPs", "img/s b16", "ms b1"]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    pm = lambda r, c: (f"{r[c]:.2f}" + (f"±{r[c + '_std']:.2f}" if isnum(r.get(c + "_std")) else "")) if isnum(r.get(c)) else "-"
    sg = lambda x: f"{x:+.2f}" if isnum(x) else "-"
    for _, r in lb.iterrows():
        dt = "-" if not isnum(r["d_test"]) else (f"{r['d_test']:+.2f}" + ((f" ({r['d_test_sd']:.2f}; {int(r['d_test_pairs'])}; {'yes' if r['d_test_same_sign'] else 'NO'})" if isnum(r["d_test_sd"]) else " (1 pair)") if isnum(r["d_test_pairs"]) else ""))
        md.append("| " + " | ".join([r["label"], r["family"], str(int(r["imgsz"])), str(int(r["epochs"])), str(int(r["n"])), pm(r, "test_AP"), dt, sg(r["d_frontier_test"]), pm(r, "val_AP"),
                                      sg(r["d_val"]), pm(r, "test_AP50"), f"{r['gflops']:.1f}" if isnum(r.get("gflops")) else "-",
                                      f"{r['throughput_b16']:.0f}" if isnum(r.get("throughput_b16")) else "-", f"{r['latency_ms']:.2f}" if isnum(r.get("latency_ms")) else "-"]) + " |")
    return lb, "\n".join(md)


@app.function
def write_leaderboard(work):
    """Leaderboard files for the downloads (CSV + Markdown) and the table of results so far."""
    work = Path(work)
    lb, md = leaderboard(work)
    out = work / "results"
    out.mkdir(parents=True, exist_ok=True)
    if len(lb):
        lb.to_csv(out / "leaderboard.csv", index=False)
    (out / "leaderboard.md").write_text(md + "\n", encoding="utf-8")
    try:
        collect_results(work).to_csv(out / "all_runs.csv", index=False)
    except Exception:
        pass
    return md


@app.function
def cost_factor(spec):
    """Relative training cost of a run (pixels dominate: data loading + activations)."""
    f = {"graft": 1.12, "graftd": 1.15, "graftd3": 1.18, "naive": 1.12, "replace": 1.12}.get(spec.variant)
    if f is None:
        f = 1.06 if spec.variant.startswith("inj") else (1.12 if spec.variant.startswith("assist") else 1.0)
    # a smaller batch is slower per epoch (measured: four-level model at 896 px, batch 16 = 203 s/epoch vs 140 s at batch 32)
    return (spec.imgsz / 640.0) ** 2 * f * (2.0 if spec.family == "v10s" else 1.0) * {32: 1.0, 16: 1.45, 8: 2.0}.get(spec.batch, 1.0)


@app.function
def estimate_gpu_gb(spec):
    """GPU memory one training process occupies (the figure Ultralytics reports, batch 32).  Measured on the RTX PRO 6000: baseline 17.5 GB at 640 px, 19-25 GB at 896 px,
    47 GB at 1280 px; four-level models 33-37 GB at 640 px and 76.5 GB at 896 px (they scale with the pixel count).  Used only to keep the scheduler from starting a run
    that cannot fit next to the running ones."""
    base = float(np.interp(spec.imgsz, [640, 896, 1280, 1536], [17.5, 25.0, 47.0, 68.0]))
    if spec.variant.startswith(("graft", "naive", "replace")):
        gb = 37.0 * (spec.imgsz / 640.0) ** 2
    elif spec.variant.startswith("assist"):
        gb = base + 8.0 * (spec.imgsz / 640.0) ** 2
    elif spec.variant.startswith("inj"):
        gb = base + 1.5
    else:
        gb = base
    return gb * (1.8 if spec.family == "v10s" else 1.0) * max(0.25, min(1.0, spec.batch / 32.0))


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
        y = YOLO(str(rdir / "weights" / "last.pt"))
        y.add_callback("on_model_save", lambda trainer: push_checkpoint(work, spec.run_id, int(trainer.epoch) + 1, rdir.parent.parent / "logs" / f"{spec.run_id}.log"))
        y.train(trainer=Trainer, resume=True)
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
def hf_check(repo, token, tag):
    """Fail EARLY (before hours of training) if the Hugging Face token cannot write to the dataset repo.  The token is only ever read from the environment / a password field,
    never written to a file, a log or the plan."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        pip_install(["huggingface_hub"])
        from huggingface_hub import HfApi
    api = HfApi(token=token)
    try:
        who = api.whoami()["name"]
        try:
            api.create_repo(repo, repo_type="dataset", exist_ok=True)
        except Exception:
            pass  # the repo exists already, or the token may not create repos but may still write to it
        api.upload_file(path_or_fileobj=f"started {time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode(), path_in_repo=f"{tag}/START.txt", repo_id=repo, repo_type="dataset")
    except Exception as e:
        raise RuntimeError(f"Hugging Face check failed for {repo}: {str(e).replace(token, '***')[:300]} (the token needs WRITE access to this dataset)")
    return f"write access to dataset {repo} confirmed as '{who}'; results go to the folder '{tag}/'"


@app.function
def hf_restore(repo, tag, work, token=""):
    """Bring a dead session back: download the finished runs' results and every run's latest checkpoint from the folder `tag` of the dataset into this work directory.  Finished runs are
    then skipped, partly trained runs resume from their last epoch (run_one -> resume_training; a damaged checkpoint falls back to a fresh start)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        pip_install(["huggingface_hub"])
        from huggingface_hub import snapshot_download
    work = Path(work)
    tmp = work / "_hf_restore"
    shutil.rmtree(tmp, ignore_errors=True)
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(tmp), token=token or None,
                      allow_patterns=[f"{tag}/runs/*/result.json", f"{tag}/runs/*/train_info.json", f"{tag}/runs/*/results.csv", f"{tag}/runs/*/args.yaml",
                                      f"{tag}/runs/*/weights/last.pt", f"{tag}/runs/*/preds_*.npz"])
    src = tmp / tag / "runs"
    if not src.exists():
        raise RuntimeError(f"folder '{tag}' of {repo} has no runs to restore")
    shutil.copytree(src, work / "runs", dirs_exist_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    fin = [r.parent.name for r in (work / "runs").glob("*/result.json") if jload(r, {}).get("finished_at")]
    part = [d.parent.parent.name for d in (work / "runs").glob("*/weights/last.pt") if d.parent.parent.name not in fin]
    return f"restored {len(fin)} finished runs and {len(part)} partly trained runs ({', '.join(part) or '-'})"


@app.function
def sync_to_hf(work, tag, repo, token, log_path=None):
    """Push everything small and valuable of the work directory to a Hugging Face dataset repo (folder `tag`): results, leaderboard, status, logs, the per-run result files and
    training curves, the latest checkpoint of every run (running ones too: a dead session can be resumed from them) and - for FINISHED runs only - the saved predictions.  Files that did not change are not uploaded again.  Never raises."""
    if not (repo and token):
        return "disabled"
    work = Path(work)
    try:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            pip_install(["huggingface_hub"])
            from huggingface_hub import HfApi
        done = []
        for rf in (work / "runs").glob("*/result.json"):
            r = jload(rf, {})
            if r.get("finished_at") and r.get("spec") and not rf.parent.name.startswith(("calib", "smoke")):
                done.append(rf.parent.name)
        allow = ["results/*", "status.json", "plan.json", "calibration.json", "environment_probe.json", "logs/worker.log", "logs/autopilot.log", "logs/smoke.log", "logs/calibrate.log",
                 "runs/*/result.json", "runs/*/results.csv", "runs/*/train_info.json", "runs/*/args.yaml", "runs/*/state.json"]
        for rid in done:  # (the checkpoints of RUNNING runs are pushed by the training processes themselves, right after each save: push_checkpoint)
            allow += [f"runs/{rid}/weights/last.pt", f"runs/{rid}/preds_*.npz"]
        api = HfApi(token=token)
        api.upload_folder(folder_path=str(work), repo_id=repo, repo_type="dataset", path_in_repo=tag, allow_patterns=allow,
                          ignore_patterns=["runs/calib*/*", "runs/smoke*/*"], commit_message=f"{tag}: {len(done)} finished runs")
        # verify: every finished run's result file, weights and results folder must now be on the hub
        remote = {f.path for f in api.list_repo_tree(repo, path_in_repo=tag, recursive=True, repo_type="dataset") if hasattr(f, "size")}
        need = [f"{tag}/runs/{rid}/result.json" for rid in done] + [f"{tag}/runs/{rid}/weights/last.pt" for rid in done] + ([f"{tag}/results/leaderboard.csv"] if done else [])
        missing = [n for n in need if n not in remote]
        if missing:
            return f"UPLOADED BUT NOT VERIFIED, missing on the hub: {missing[:4]}"
        return f"uploaded and verified on the hub ({len(done)} finished runs, {len(remote)} files)"
    except Exception as e:
        msg = f"sync failed: {str(e).replace(token, '***')[:200]}"
        if log_path:
            log(msg, log_path)
        return msg


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
    waiting_logged: set = set()
    stagger_s = float(plan.get("stagger_s", 60.0))
    last_start = 0.0
    hf_repo, hf_token, hf_tag = os.environ.get("GRAFT_HF_REPO", ""), os.environ.get("HF_TOKEN", ""), plan.get("hf_tag", "run")
    last_sync = time.time()  # first sync after 10 minutes, then after every finished run and at least every 10 minutes
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
                try:  # results so far, readable from outside at any time
                    write_leaderboard(work)
                except Exception as e:
                    log(f"leaderboard failed: {e!r}", lp)
                last_sync = 0.0  # a run has finished: push it to Hugging Face on this pass
            elif time.time() > hard_deadline:
                proc.terminate()
                status[rid].update(state="killed (hard deadline)")
        # start new children if time allows (staggered: simultaneous starters would race on shared downloads and caches)
        while queue and len(running) < n_par:
            if time.time() - last_start < stagger_s:
                break
            gpu_cap = float(plan.get("gpu_gb") or 0.0)
            used_gb = sum(estimate_gpu_gb(sp) for _, sp, _ in running.values())
            idx = None  # first queued run that fits next to the running ones.  A big run that has to wait is protected (backfilling): a smaller run behind it
            blocked, head_start = None, 0.0  # may only start if it leaves room for the big one or will be finished by the time that room exists
            for j, cand in enumerate(queue):
                gb_c = estimate_gpu_gb(cand)
                fits = not (gpu_cap and running) or used_gb + gb_c <= gpu_cap
                if blocked is None:
                    if fits:
                        idx = j
                        break
                    blocked = cand
                    if cand.run_id not in waiting_logged:
                        waiting_logged.add(cand.run_id)
                        log(f"{cand.run_id} waits: ~{gb_c:.0f} GB would not fit next to the running jobs "
                            f"({', '.join(f'{r} ~{estimate_gpu_gb(sp):.0f} GB' for r, (_, sp, _) in running.items())}) on a {gpu_cap:.0f} GB GPU", lp)
                    t_free, left = time.time(), used_gb  # when will enough memory be free?  (estimated end of the running jobs, earliest first)
                    for end_t, gb_r in sorted((t0_ + float(status[r_].get("est_s") or 0.0), estimate_gpu_gb(sp_)) for r_, (_, sp_, t0_) in running.items()):
                        if left + gb_c <= gpu_cap:
                            break
                        left -= gb_r
                        t_free = max(t_free, end_t)
                    head_start = t_free
                    continue
                est_c = estimate_run_seconds(cand, s_per_epoch, plan.get("eval_s", 240.0), ratio) * (1.0 + 0.15 * max(0, len(running)))
                if fits and (used_gb + gb_c + estimate_gpu_gb(blocked) <= gpu_cap or time.time() + est_c <= head_start):
                    idx = j
                    break
            if idx is None:
                break
            spec = queue[idx]
            est = estimate_run_seconds(spec, s_per_epoch, plan.get("eval_s", 240.0), ratio) * (1.0 + 0.15 * max(0, len(running)))
            if time.time() + est > deadline:
                status[spec.run_id]["state"] = "skipped (would exceed time budget)"
                queue.pop(idx)
                log(f"skip {spec.run_id}: est {fmt_hours(est)} exceeds remaining {fmt_hours(deadline - time.time())}", lp)
                continue
            queue.pop(idx)
            args = {"work": str(work), "spec_dict": asdict(spec), "workers": plan["workers"], "cache": (plan["cache"] if spec.imgsz <= 960 else False),
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
        if hf_repo and hf_token and time.time() - last_sync > 600:
            last_sync = time.time()
            log(f"hugging face sync: {sync_to_hf(work, hf_tag, hf_repo, hf_token, lp)}", lp)
        time.sleep(5)
    write_status()
    log("worker finished", lp)
    if plan.get("limit"):
        return
    for name, fn in (("leaderboard", lambda: "\n" + write_leaderboard(work)),
                     ("sequential latency re-measurement (runs)", lambda: remeasure_latency(work, lp)),
                     ("leaderboard", lambda: write_leaderboard(work)[:0]),
                     ("contrasts with image-bootstrap intervals", lambda: headline_contrasts(work, B=int(plan.get("bootstrap_B", 200)), log_path=lp)),
                     ("diagnostics zip", lambda: zip_diagnostics(work)),
                     ("hugging face upload (final)", lambda: sync_to_hf(work, hf_tag, hf_repo, hf_token, lp))):
        try:
            log(f"{name}: {fn()}", lp)
        except Exception as e:
            log(f"{name} failed: {e!r}\n{traceback.format_exc()}", lp)


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
    for key_, m in r.get("metrics", {}).items():  # the same model tested at other input sizes (no retraining)
        mm_ = re.fullmatch(r"(val|test-dev)\|(\d+)\|o2o\|std", key_)
        if mm_ and isinstance(m, dict) and int(mm_.group(2)) != sp["imgsz"]:
            pre_, sz = ("val" if mm_.group(1) == "val" else "test"), int(mm_.group(2))
            row[f"{pre_}_AP50_at{sz}"], row[f"{pre_}_AP_at{sz}"], row[f"{pre_}_APs_at{sz}"] = m["AP50"], m["AP"], m["by_area"]["small"]["AP"]
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
    row["throughput_b16"] = lat.get("throughput_img_s_b16")
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
def run_pred_path(work, run_id, split="val", size=640):
    return Path(work) / "runs" / run_id / f"preds_{split_dir_name(split)}_{size}_o2o.npz"


# ------------------------------------------------------------------ transfer audit (static, no training)


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
def isnum(x):
    try:
        return x is not None and not (isinstance(x, float) and math.isnan(x))
    except Exception:
        return False


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
    def exact():
        reps = exactness_report(work, variants=("graft", "graftd", "inj16"), modes={"graftd": "role_ws"})
        reps["graftd (v10s)"] = exactness_report(work, family="v10s", variants=("graftd",), modes={"graftd": "role_ws"})["graftd"]
        bad = [v for v, r in reps.items() if r["max_abs_feature_diff"] > 1e-4 or r["max_abs_head_output_diff"] > 1e-4]
        if bad:
            raise RuntimeError(f"variants that do NOT reproduce the pretrained detector at initialisation: {bad}")
        return {v: {"max_feature_diff": r["max_abs_feature_diff"], "max_head_diff": r["max_abs_head_output_diff"]} for v, r in reps.items()}
    step("exactness of the grafted / injected variants (random inputs, real COCO checkpoint)", exact)
    for sp in ("train", "val"):
        step(f"prepare {sp} subset ({limit} images @ {size}px)", lambda sp=sp: bool(prepare_split(work, sp, sizes=(size,), limit=limit)))
    results = {}
    for variant, transfer, fam in (("base", "role", "v10"), ("graft", "role_ws", "v10"), ("inj16", "role", "v10"), ("graftd", "role_ws", "v10"),
                                   ("base", "role", "v10s"), ("graftd", "role_ws", "v10s")):
        variant_key = (variant if transfer == "role" else f"{variant}-{transfer}") + ("" if fam == "v10" else f"-{fam}")
        spec = RunSpec(run_id=f"smoke-{variant_key}", group=variant_key, family=fam, variant=variant, transfer=transfer, epochs=1, imgsz=size, batch=8, seed=0)

        def go(spec=spec):
            shutil.rmtree(work / "runs" / spec.run_id, ignore_errors=True)
            ok = run_one(str(work), asdict(spec), workers=0, cache=False, device=device, data_suffix=f"_lim{limit}", limit=limit, tide=True, test_dev=False)
            r = jload(work / "runs" / spec.run_id / "result.json", {})
            results[variant_key] = r
            m = r.get("metrics", {}).get(f"val|{size}|o2o|std", {})
            return {"run_one_ok": ok, "AP50": m.get("AP50"), "n_dets": m.get("n_dets"), "tide_ok": "main" in (r.get("tide|val") or {})}
        step(f"train + evaluate {variant_key} (1 epoch, tiny)", go)

    def selftest():
        return evaluator_selftest(work / "data" / f"gt_val_lim{limit}.json")
    step("evaluator self-test on synthetic detections (bootstrap accumulation == pycocotools == faster-coco-eval)", selftest)
    return steps


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
def make_worker_plan(work, specs, par, cache, budget_h, s_per_epoch, limit=None, notebook_file=None, bootstrap_B=200, hf_tag=""):
    """The dict that worker_main() consumes."""
    now = time.time()
    plan = {"work": str(work), "specs": [asdict(s) for s in specs], "n_parallel": par["n_parallel"], "workers": par["workers"], "cache": cache, "device": None,
            "deadline": now + budget_h * 3600, "hard_deadline": now + min(budget_h * 3600 + 1800, 11.5 * 3600), "s_per_epoch": float(s_per_epoch),
            "eval_s": 240.0, "gpu_gb": float(par.get("gpu_gb") or 0.0), "tide": True, "test_dev": limit is None, "bootstrap_B": int(bootstrap_B), "hf_tag": hf_tag}
    if limit is not None:
        plan.update(limit=limit, data_suffix=f"_lim{limit}")
    if notebook_file:
        plan["notebook_file"] = str(notebook_file)
    return plan


@app.function
def autopilot(work, session="A", budget_h=10.0, epochs=90, notebook_file=None, skip_smoke=False, skip_calibration=False, need_cuda=True, limit=None,
              log_path=None, launch=True, recalibrate=False, hf_repo="", hf_token="", hf_resume=""):
    """Everything between an EMPTY work directory and a RUNNING training worker, in one call.  Idempotent: finished parts are skipped.  `limit` runs it on a tiny subset (test mode).
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
    hf_repo, hf_token = (hf_repo or "").strip(), (hf_token or "").strip()
    hf_tag = f"session{session}-{time.strftime('%Y%m%d-%H%M')}"
    old_plan = jload(work / "plan.json", {})
    if old_plan.get("hf_tag") and (work / "runs").exists() and any((work / "runs").glob("*/result.json")):
        hf_tag = old_plan["hf_tag"]  # a restarted autopilot continues in the same folder
    hf_resume = (hf_resume or "").strip().strip("/")
    if hf_resume:
        hf_tag = hf_resume  # continue in the same folder
    if hf_repo and hf_token:
        if not step("Hugging Face access (results are pushed there during the run)", lambda: hf_check(hf_repo, hf_token, hf_tag)):
            return fail()
    if hf_resume and hf_repo:
        if not step(f"restore the session '{hf_resume}' from Hugging Face", lambda: hf_restore(hf_repo, hf_resume, work, hf_token)):
            return fail()
    specs = make_plan_v2(session, epochs=int(epochs))
    if limit is not None:
        specs = [dataclasses.replace(s, epochs=1, batch=8, imgsz=min(s.imgsz, 320), eval_sizes=()) for s in specs[:2]]
    train_sizes = sorted({s.imgsz for s in specs} | {640})
    eval_only = sorted({e for s in specs for e in s.eval_sizes} - set(train_sizes))
    sizes = (320,) if limit else tuple(train_sizes)

    def data():
        prepare_all_data(work, imgszs=sizes, limit=limit, log_path=log_path)
        if not limit:
            for sz in eval_only:
                prepare_all_data(work, imgszs=[sz], splits=("val", "test-dev"), log_path=log_path)
        return f"VisDrone prepared at training sizes {list(sizes)} px and evaluation-only sizes {eval_only}" + (f" (first {limit} images only)" if limit else "")
    if not step("data (download, convert, pre-resize)", data):
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
        if not step("smoke test (tiny end-to-end run of every model type on this machine)", smoke):
            return fail()
    par = plan_parallel(state["probe"])
    spe = 60.0
    if limit is None and not skip_calibration:
        def calib():
            old = jload(work / "calibration.json", {})
            if old.get("s_per_epoch") and not recalibrate:
                state["calib"] = old
                return f"reused the earlier measurement: {old.get('s_per_epoch'):.0f} s/epoch"
            lp = work / "logs" / "calibrate.log"
            rc = run_fn_in_subprocess(str(nb), "calibrate", {"work": str(work), "workers": par["workers"], "cache": choose_cache(par["n_parallel"])}, lp,
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
    cache = choose_cache(par["n_parallel"], need_gb=9.0 * (max(s.imgsz for s in specs if s.imgsz <= 960) / 640.0) ** 2) if any(s.imgsz <= 960 for s in specs) else False
    plan = make_worker_plan(work, specs, par, cache, budget_h, spe, limit=limit, notebook_file=nb, hf_tag=hf_tag)
    jdump(plan, work / "plan_preview.json")
    summary = f"session {session}: {len(specs)} runs, {spe:.0f} s/epoch, {par['n_parallel']} parallel x {par['workers']} loader workers, cache={cache}"
    steps.append({"step": "plan", "ok": True, "seconds": 0.0, "detail": summary})

    def shared_inputs():
        for fam in sorted({s.family for s in specs}):
            load_pretrained(work, fam)
        return f"pretrained checkpoints for {sorted({s.family for s in specs})}"
    if not step("shared inputs (checkpoints)", shared_inputs):
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
        state["pid"] = launch_worker(work, plan, notebook_file=str(nb), extra_env=({"HF_TOKEN": hf_token, "GRAFT_HF_REPO": hf_repo, "GRAFT_HF_TAG": hf_tag} if (hf_repo and hf_token) else {}))
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
    files = [work / n for n in ("status.json", "worker.json", "plan.json", "plan_preview.json", "calibration.json", "search_selection.json")]
    files += sorted((work / "results").glob("*"))
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

@app.function
def bigger_eval_sizes(imgsz):
    """Test sizes above the training size (no retraining): 640 -> 800 and 960; 896 -> 1024; larger sizes none."""
    return (800, 960) if imgsz <= 640 else ((int(round(imgsz * 1.15 / 32.0)) * 32,) if imgsz <= 960 else ())


@app.function
def default_batch(family, variant, imgsz):
    """Batch size per run (the nominal batch stays 64 through gradient accumulation).  The four-level models need about 37 GB at batch 32 and 640 px and grow with the pixel
    count (76 GB at 896 px), so they use batch 16 from 896 px and batch 8 from 1280 px; everything else uses 32.  (A different batch changes the batch-norm statistics slightly:
    compare runs only within one session / one batch policy.)"""
    graft_like = variant.startswith(("graft", "assist"))
    if graft_like and imgsz >= 1280:
        return 8
    if graft_like and (imgsz >= 896 or family == "v10s"):
        return 16
    return 32


@app.function
def make_plan_v2(session="A", epochs=90, long_epochs=150, family="v10"):
    """Confirmatory plans (the hypotheses are fixed before the runs; nothing is selected on the way).  Model under test: M = 'graftd' + 'role_ws' (a stride-4 level with a 2x deeper
    block, initialised from the pretrained P3 level), chosen on validation in an earlier search; B = the unmodified YOLOv10 baseline.
      Session A (seeds): C1 M beats B at 896 px, 90 epochs, with 3 seeds each.
      Session B (best resolution, long schedule): C2 the gain survives a 150-epoch schedule (640 px); C4 M and B at 1280 px, 90 epochs.
      Session C (generality): C3 the gain replicates at a second model scale (YOLOv10s, 640 px, 90 epochs, 3 seeds).
    Runs are ordered longest / most memory-hungry first; the scheduler drops what no longer fits the budget from the end of the queue."""
    P: List[RunSpec] = []
    M = ("graftd", "role_ws", "ws", "P2-Graft deep, warm-started")
    B = ("base", "role", "f16", "YOLOv10 baseline")

    def add(model, fam, seed, imgsz=640, ep=epochs, notes=""):
        variant, transfer, tag, lab = model
        rid = run_id_for(fam, variant, f"{tag}-{imgsz}-e{ep}", seed)
        if any(x.run_id == rid for x in P):
            return
        P.append(RunSpec(run_id=rid, group=f"{lab}, {fam}, {imgsz} px, {ep} ep", family=fam, variant=variant, transfer=transfer, seed=seed, epochs=ep, imgsz=imgsz, floor=16,
                         batch=default_batch(fam, variant, imgsz), tier={"A": 1, "B": 2, "C": 3}[session], eval_sizes=bigger_eval_sizes(imgsz), notes=notes))
    if session == "A":  # interleaved so that a partial session still has paired seeds
        for sd in (0, 1, 2):
            add(M, family, sd, 896, notes="C1")
            add(B, family, sd, 896, notes="C1")
    elif session == "B":
        add(M, family, 0, 1280, notes="C4")
        add(B, family, 0, 1280, notes="C4")
        add(M, family, 0, 640, long_epochs, notes="C2")
        add(B, family, 0, 640, long_epochs, notes="C2")
    elif session == "C":
        for sd in (0, 1, 2):
            add(M, "v10s", sd, 640, notes="C3")
            add(B, "v10s", sd, 640, notes="C3")
    else:
        raise ValueError(session)
    return P


@app.function
def headline_contrasts(work, B=200, log_path=None):
    """Image-level bootstrap intervals (test-dev, the primary split) of every model-minus-baseline difference with the same family, input size and schedule: the per-image resamples are
    shared by all runs (paired), the AP of the seeds that both conditions have is averaged.  Writes results/contrasts.csv.  (The interval captures sampling of the test images, the seed
    spread is in the leaderboard; both are needed.)"""
    import pandas as pd
    work = Path(work)
    df = collect_results(work)
    if df.empty:
        return 0
    items, pairs = {}, []
    for (fam, imgsz, ep), grp in df.groupby(["family", "imgsz", "epochs"]):
        base = grp[(grp.variant == "base") & (grp.transfer == "role")]
        for (v, t), g2 in grp[grp.variant != "base"].groupby(["variant", "transfer"]):
            common = sorted(set(base["seed"]) & set(g2["seed"]))
            if not common:
                continue
            a = {int(s): r for s, r in zip(base["seed"], base["run_id"])}
            b = {int(s): r for s, r in zip(g2["seed"], g2["run_id"])}
            for s in common:
                for r in (a[s], b[s]):
                    if run_pred_path(work, r, "test-dev", int(imgsz)).exists():
                        items[r] = int(imgsz)
            pairs.append((fam, v, t, int(imgsz), int(ep), [s for s in common if a[s] in items and b[s] in items], a, b))
    if not items:
        return 0
    boots = bootstrap_runs(work, items, B=B, split="test-dev", log_path=log_path)
    rows = []
    for fam, v, t, imgsz, ep, seeds, a, b in pairs:
        if not seeds:
            continue
        xa = np.mean([boots[a[s]] for s in seeds], axis=0)
        xb = np.mean([boots[b[s]] for s in seeds], axis=0)
        ci = ci_from_boot(xa, xb, 0)
        ci50 = ci_from_boot(xa, xb, 1)
        rows.append({"family": fam, "model": f"{v}/{t}", "imgsz": imgsz, "epochs": ep, "seeds": ",".join(map(str, seeds)), "test_dAP": ci["delta"], "test_dAP_lo": ci["ci95"][0],
                     "test_dAP_hi": ci["ci95"][1], "test_dAP50": ci50["delta"], "test_dAP50_lo": ci50["ci95"][0], "test_dAP50_hi": ci50["ci95"][1], "B": B})
    out = work / "results"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "contrasts.csv", index=False)
    return len(rows)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # P2-Graft v2 — confirmatory runs (YOLOv10, VisDrone-DET2019)

    **What this does when you press Run all** (nothing else has to be clicked): environment check/repair → data download (≈ 2 GB) → smoke test of every model type → calibration →
    a background worker trains the fixed plan of the chosen session → results table (`results/leaderboard.csv`, `contrasts.csv`, `all_runs.csv`) → `diagnostics.zip`.

    * **Session A (seeds)** – the model under test (*P2-Graft deep, warm-started*) against the unmodified baseline at 896 px, 3 seeds each.
    * **Session B (best resolution)** – the same pair at 1280 px, and at 640 px with a 150-epoch schedule.
    * **Session C (generality)** – the same pair at a second model scale (YOLOv10s, 640 px, 3 seeds).

    The primary metric is **AP on VisDrone test-dev** (no decision was ever taken on it). Download **diagnostics.zip** from the last cell **before the session ends** — molab deletes
    everything else (12 h maximum, 90 min without a connected browser: keep this tab open).
    """)
    return


@app.cell
def _(mo):
    work_input = mo.ui.text(value=str(default_work_dir()), label="Work directory", full_width=True)
    session_ui = mo.ui.dropdown(options={"Session A - seeds: 896 px, 3 seeds each (about 8 h)": "A", "Session B - best resolution: 1280 px and a 150-epoch schedule (about 8 h)": "B",
                                         "Session C - second model scale: YOLOv10s, 3 seeds (about 8 h)": "C"},
                                value="Session A - seeds: 896 px, 3 seeds each (about 8 h)", label="What to run", full_width=True)
    budget_h = mo.ui.slider(2.0, 10.0, value=10.0, step=0.5, show_value=True, label="Budget for training + evaluation (hours from the launch; the 12 h session also needs ~1 h of set-up and ~0.5 h of final analysis)")
    hf_repo = mo.ui.text(value="mudassiruddin/yolo", label="Hugging Face DATASET repo that receives the results during the run (user/name)", full_width=True)
    hf_token = mo.ui.text(value="", kind="password", label="Hugging Face token with WRITE access to that dataset (kept in memory only: never written to a file or a log; revoke it afterwards)", full_width=True)
    hf_resume = mo.ui.text(value="", label="RESUME a dead session: its folder name in the dataset (e.g. sessionA-20261103-0052). Empty = fresh start. Finished runs are skipped, partly trained runs continue from their last checkpoint.", full_width=True)
    auto_ok = mo.ui.checkbox(value=False, label="Start automatically when the notebook is run (Run all) - only sensible with an EMPTY token field")
    auto_btn = mo.ui.run_button(label="START (paste the token first)")
    mo.vstack([mo.md("## Settings"), work_input, session_ui, budget_h, hf_repo, hf_token, hf_resume, auto_ok, auto_btn])
    return auto_btn, auto_ok, budget_h, hf_repo, hf_resume, hf_token, session_ui, work_input


@app.cell
def _(work_input):
    WORK = Path(work_input.value).expanduser()
    WORK.mkdir(parents=True, exist_ok=True)
    return (WORK,)


@app.cell
def _(WORK, auto_btn, auto_ok, budget_h, hf_repo, hf_resume, hf_token, mo, session_ui):
    mo.stop(not (auto_ok.value or auto_btn.value), mo.md("*Not started. Paste the Hugging Face token, then press START.*"))
    _nb = NOTEBOOK_FILE or Path(mo.notebook_location() / "graft_p2_v2.py")
    AUTO = autopilot(WORK, session=session_ui.value, budget_h=float(budget_h.value), notebook_file=str(_nb), log_path=WORK / "logs" / "autopilot.log",
                     hf_repo=hf_repo.value, hf_token=hf_token.value, hf_resume=hf_resume.value)
    _lines = [f"- {'✅' if s['ok'] else '❌'} **{s['step']}** ({s['seconds']} s): `{str(s['detail'])[:500]}`" for s in AUTO["steps"]]
    _nohf = not (hf_repo.value.strip() and hf_token.value.strip())
    mo.vstack([mo.callout(mo.md("**No Hugging Face token: results exist only on molab's disk and are lost if the session ends. Stop and paste a token if you will not be here.**"), kind="warn") if _nohf else mo.md(""),
               mo.callout(mo.md("**The worker is running — follow it below.**" if AUTO["ok"] else "**Stopped at a failed step** — read the last line; pressing Start again resumes."),
                          kind="success" if AUTO["ok"] else "danger"), mo.md("\n".join(_lines))])
    return (AUTO,)


@app.cell
def _(mo):
    refresh = mo.ui.refresh(options=["30s", "1m", "5m"], default_interval="30s", label="Auto-refresh")
    mo.vstack([mo.md("## Monitor"), refresh])
    return (refresh,)


@app.cell
def _(WORK, mo, refresh):
    refresh
    _st = jload(WORK / "status.json", {})
    _w = jload(WORK / "worker.json", {})
    if not _st:
        mo.stop(True, mo.md("*No worker status yet.*"))
    _now = time.time()
    _rows = [{"run": rid, "state": r.get("state"), "elapsed": fmt_hours((_now - r["started"]) if r.get("state") == "running" and r.get("started") else r.get("seconds")) if (r.get("started") or r.get("seconds")) else "",
              "estimate": fmt_hours(r.get("est_s", 0)) if r.get("est_s") else ""} for rid, r in _st.get("runs", {}).items()]
    _df = pd.DataFrame(_rows)
    _alive = pid_alive(_w.get("pid", -1)) if _w else False
    mo.vstack([mo.md(f"**Worker {'running' if _alive else 'not running'}** · {int((_df.state == 'done').sum())}/{len(_df)} runs done · time left in budget: {fmt_hours(_st.get('deadline', _now) - _now)}"),
               mo.ui.table(_df, selection=None, page_size=30), mo.accordion({"worker log (tail)": mo.md("```\n" + tail_text(WORK / "logs" / "worker.log", 30) + "\n```")})])
    return


@app.cell
def _(WORK, mo, refresh):
    refresh
    _lb, _md = leaderboard(WORK)
    mo.stop(len(_lb) == 0, mo.md("*No finished run yet.*"))
    _csv = collect_results(WORK).drop(columns=["val_perclass_AP50", "val_perclass_AP"], errors="ignore").to_csv(index=False).encode("utf-8")
    mo.vstack([mo.md("### Leaderboard (test-dev AP is the primary metric; `d frontier` = distance to the baseline's resolution frontier at equal GFLOPs)"), mo.md(_md),
               mo.download(data=_csv, filename="all_runs_so_far.csv", mimetype="text/csv", label="Download the results so far (small CSV, a cheap backup)")])
    return


@app.cell
def _(AUTO, WORK, mo):
    # keeps the session busy until the worker has finished (so that Run all stays running), then offers the downloads
    while True:
        _w = jload(WORK / "worker.json", {})
        if not (_w and pid_alive(_w.get("pid", -1))):
            break
        time.sleep(30)
    _dz = WORK / "diagnostics.zip"
    mo.stop(not _dz.exists(), mo.md("## Download\n*The worker is not running and no diagnostics.zip exists (yet).*"))
    mo.vstack([mo.md("## Download — **do this before the session ends**\n`diagnostics.zip` holds results/leaderboard.csv, contrasts.csv, all_runs.csv, every run's result.json, logs."),
               mo.download(data=_dz.read_bytes(), filename="diagnostics.zip", mimetype="application/zip", label="Download diagnostics.zip")])
    return


@app.cell
def _(mo):
    diag_pred = mo.ui.checkbox(label="include saved predictions (hundreds of MB)")
    diag_btn = mo.ui.run_button(label="Build / rebuild diagnostics.zip now")
    mo.vstack([mo.md("### Manual diagnostics (if the run failed or stopped early)"), diag_pred, diag_btn])
    return diag_btn, diag_pred


@app.cell
def _(WORK, diag_btn, diag_pred, mo):
    mo.stop(not diag_btn.value, mo.md("*Not built.*"))
    write_leaderboard(WORK)
    _z = Path(zip_diagnostics(WORK, include_predictions=bool(diag_pred.value)))
    mo.vstack([mo.md(f"`{_z.name}`: {_z.stat().st_size / 1e6:.1f} MB"), mo.download(data=_z.read_bytes(), filename="diagnostics.zip", mimetype="application/zip", label="Download diagnostics.zip")])
    return


if __name__ == "__main__":
    app.run()
