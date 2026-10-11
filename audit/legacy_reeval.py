"""Re-evaluate the legacy LAF-YOLOv10 checkpoints with the legacy notebook's own model code,
but with (a) the legacy evaluator exactly as published (conf>0.25 before mAP) and
(b) a standard protocol (conf 0.001, 300 dets/img) scored by pycocotools.

Usage: python -I legacy_reeval.py <notebook.ipynb> <checkpoint_dir> <visdrone_yolo_root> <out_json> [n_images]
The legacy notebook code is executed only for its *definitions* (model + dataset + evaluator cells).
"""
import json, sys, os, time, io, contextlib, warnings
warnings.filterwarnings("ignore")
nb_path, ckpt_dir, data_root, out_json = sys.argv[1:5]
n_images = int(sys.argv[5]) if len(sys.argv) > 5 else 0

nb = json.load(open(nb_path, encoding="utf-8"))
# cells that only *define* things we need (indices verified against the notebook listing)
DEF_CELLS = [3, 6, 14, 20, 24, 28, 32, 36, 51]
src = []
for i in DEF_CELLS:
    code = "".join(nb["cells"][i]["source"])
    code = "\n".join(l for l in code.splitlines() if not l.lstrip().startswith(("!", "%")))
    src.append(f"# ---- legacy cell {i}\n{code}\n")
ns = {"__name__": "legacy_ns"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile("\n".join(src), "legacy_cells", "exec"), ns)

import numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
CFG, LAFConfig, LAFYOLOv10 = ns["CFG"], ns["LAFConfig"], ns["LAFYOLOv10"]
YoloDetDataset, collate_fn = ns["YoloDetDataset"], ns["collate_fn"]
postprocess_nms, DetMetrics, compute_precision_recall = ns["postprocess_nms"], ns["DetMetrics"], ns["compute_precision_recall"]
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
dev = torch.device("cpu")
ns["DEVICE"] = dev

ds = YoloDetDataset(data_root, "val", img_size=640, augment=False)
if n_images:
    ds = Subset(ds, list(range(0, len(ds), max(1, len(ds) // n_images)))[:n_images])
dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0)
print("val images:", len(ds), flush=True)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def coco_eval(all_dets, all_targets, max_dets=300):
    """all_dets: list of dict(boxes[xyxy], scores, labels) per image (letterboxed coords; IoU is scale/shift invariant)."""
    images, anns, dets = [], [], []
    aid = 1
    for i, (d, t) in enumerate(zip(all_dets, all_targets)):
        images.append({"id": i, "width": 640, "height": 640})
        for b, l in zip(t["boxes"].tolist(), t["labels"].tolist()):
            w, h = b[2] - b[0], b[3] - b[1]
            anns.append({"id": aid, "image_id": i, "category_id": int(l), "bbox": [b[0], b[1], w, h], "area": w * h, "iscrowd": 0}); aid += 1
        order = torch.argsort(d["scores"], descending=True)[:max_dets]
        for j in order.tolist():
            b = d["boxes"][j].tolist()
            dets.append({"image_id": i, "category_id": int(d["labels"][j]), "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": float(d["scores"][j])})
    gt = COCO(); gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": c} for c in range(10)]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt = gt.loadRes(dets) if dets else COCO()
        E = COCOeval(gt, dt, "bbox"); E.params.maxDets = [1, 10, max_dets]
        E.evaluate(); E.accumulate(); E.summarize()
    return {"AP50": float(E.stats[1]) * 100, "AP": float(E.stats[0]) * 100}

results = {}
for seed in (42, 123, 256):
    ck = torch.load(Path(ckpt_dir) / f"laf_yolov10_main_seed{seed}.pt", map_location="cpu", weights_only=False)
    cfg = LAFConfig(**ck["config"])
    model = LAFYOLOv10(cfg).to(dev)
    model.load_state_dict(ck["ema_state_dict"] if ck.get("ema_state_dict") is not None else ck["model_state_dict"])
    model.eval()
    use_nms = not cfg.use_dual_head
    raw_dets, targets = [], []
    t0 = time.time()
    with torch.no_grad():
        for imgs, tg in dl:
            cls_outs, reg_outs, feat_shapes = model.forward_for_inference(imgs)
            cls_flat, boxes, points, strides, counts = model.decode(cls_outs, reg_outs, feat_shapes, device=dev)
            raw_dets += postprocess_nms(cls_flat, boxes, conf_thres=0.001, iou_thres=0.6, max_det=300, use_nms=use_nms)
            targets += tg
    print(f"seed {seed}: inference {time.time()-t0:.0f}s, {sum(len(d['scores']) for d in raw_dets)} dets @conf>0.001", flush=True)

    def tm_map(dets, max_det_thr=None):
        m = DetMetrics(class_names=list(CFG.CLASS_NAMES))
        m.update(dets, targets)
        r = m.compute()
        return {"mAP50": r["mAP@0.5"], "mAP50_95": r["mAP@[.5,.95]"]}

    def subset(conf):
        out = []
        for d in raw_dets:
            k = d["scores"] > conf
            out.append({"boxes": d["boxes"][k], "scores": d["scores"][k], "labels": d["labels"][k]})
        return out

    d25 = subset(0.25)
    P, R = compute_precision_recall(d25, targets, 0.5)
    r = {
        "legacy_protocol(conf>0.25, torchmetrics)": tm_map(d25),
        "torchmetrics(conf>0.001)": tm_map(raw_dets),
        "pycocotools(conf>0.25, maxDets300)": coco_eval(d25, targets),
        "pycocotools(conf>0.001, maxDets300)": coco_eval(raw_dets, targets),
        "P@0.25": P, "R@0.25": R,
        "n_dets_per_img@0.25": float(np.mean([len(d["scores"]) for d in d25])),
        "n_gt_per_img": float(np.mean([len(t["labels"]) for t in targets])),
        "max_gt_per_img": int(max(len(t["labels"]) for t in targets)),
        "frac_images_with_>100_gt": float(np.mean([len(t["labels"]) > 100 for t in targets])),
        "recorded_final_val_in_ckpt": {k: v for k, v in ck["history"][-1].items() if k.startswith("val_")},
    }
    results[str(seed)] = r
    print(json.dumps(r, indent=1), flush=True)
    json.dump(results, open(out_json, "w"), indent=1)
print("done")
