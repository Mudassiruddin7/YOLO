"""Check the legacy paper's Table 7 (held-out internal split 23.5/11.9, test-dev 22.5/11.4) by re-creating the locked holdout
exactly as the notebook does (random.Random(777).sample over the sorted train file list) and scoring seed 42 with
(a) the legacy protocol (conf>0.25) and (b) a standard protocol (conf>0.001)."""
import json, sys, os, io, contextlib, random, time, warnings
warnings.filterwarnings("ignore")
nb_path, ckpt_dir, data_root, out_json = sys.argv[1:5]
nb = json.load(open(nb_path, encoding="utf-8"))
DEF_CELLS = [3, 6, 14, 20, 24, 28, 32, 36, 51]
src = ["\n".join(l for l in "".join(nb["cells"][i]["source"]).splitlines() if not l.lstrip().startswith(("!", "%"))) for i in DEF_CELLS]
ns = {"__name__": "legacy_ns"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile("\n".join(src), "legacy_cells", "exec"), ns)
import numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
ck = torch.load(Path(ckpt_dir) / "laf_yolov10_main_seed42.pt", map_location="cpu", weights_only=False)
cfg = ns["LAFConfig"](**ck["config"]); model = ns["LAFYOLOv10"](cfg); model.load_state_dict(ck["ema_state_dict"]); model.eval()

def score(ds, name):
    dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=ns["collate_fn"], num_workers=0)
    dets, targets = [], []
    with torch.no_grad():
        for imgs, tg in dl:
            co, ro, fs = model.forward_for_inference(imgs)
            cf, bx, pts, st, cn = model.decode(co, ro, fs, device=torch.device("cpu"))
            dets += ns["postprocess_nms"](cf, bx, conf_thres=0.001, iou_thres=0.6, max_det=300, use_nms=not cfg.use_dual_head)
            targets += tg
    def tm(conf):
        sub = [{"boxes": d["boxes"][d["scores"] > conf], "scores": d["scores"][d["scores"] > conf], "labels": d["labels"][d["scores"] > conf]} for d in dets]
        m = ns["DetMetrics"](class_names=list(ns["CFG"].CLASS_NAMES)); m.update(sub, targets); r = m.compute()
        return {"mAP50": r["mAP@0.5"], "mAP50_95": r["mAP@[.5,.95]"]}
    out = {"n_images": len(ds), "legacy_protocol(conf>0.25)": tm(0.25), "standard_protocol(conf>0.001)": tm(0.001)}
    print(name, json.dumps(out), flush=True)
    return out

res = {}
# 1) internal holdout, re-created exactly as make_holdout_split() does
train_ds = ns["YoloDetDataset"](data_root, "train", img_size=640, augment=False)
n_total = len(train_ds.img_files)
rng = random.Random(777)
n_hold = max(1, int(round(n_total * 0.10)))
hold_idx = sorted(rng.sample(range(n_total), n_hold))
print("train files:", n_total, "holdout:", len(hold_idx), flush=True)
res["internal_holdout_recreated"] = score(Subset(train_ds, hold_idx), "internal holdout")
json.dump(res, open(out_json, "w"), indent=1)
# 2) test-dev
test_ds = ns["YoloDetDataset"](data_root, "test", img_size=640, augment=False)
res["test_dev"] = score(test_ds, "test-dev")
json.dump(res, open(out_json, "w"), indent=1)
