"""Does the legacy TIDE analysis (Table 6) feed tidecv correctly-formatted boxes?  tidecv expects [x, y, w, h];
the legacy notebook passed xyxy.  Re-run seed 42 with (a) legacy call, (b) correct format at conf>0.25, (c) correct format at conf>0.001."""
import json, sys, os, io, contextlib, time, warnings
warnings.filterwarnings("ignore")
nb_path, ckpt_dir, data_root, out_json = sys.argv[1:5]
nb = json.load(open(nb_path, encoding="utf-8"))
DEF_CELLS = [3, 6, 14, 20, 24, 28, 32, 36, 51]
src = []
for i in DEF_CELLS:
    code = "\n".join(l for l in "".join(nb["cells"][i]["source"]).splitlines() if not l.lstrip().startswith(("!", "%")))
    src.append(code)
ns = {"__name__": "legacy_ns"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile("\n".join(src), "legacy_cells", "exec"), ns)
import numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader
from tidecv import TIDE, Data
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
ds = ns["YoloDetDataset"](data_root, "val", img_size=640, augment=False)
dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=ns["collate_fn"], num_workers=0)
ck = torch.load(Path(ckpt_dir) / "laf_yolov10_main_seed42.pt", map_location="cpu", weights_only=False)
cfg = ns["LAFConfig"](**ck["config"]); model = ns["LAFYOLOv10"](cfg); model.load_state_dict(ck["ema_state_dict"]); model.eval()
dets, targets = [], []
with torch.no_grad():
    for imgs, tg in dl:
        co, ro, fs = model.forward_for_inference(imgs)
        cf, bx, pts, st, cn = model.decode(co, ro, fs, device=torch.device("cpu"))
        dets += ns["postprocess_nms"](cf, bx, conf_thres=0.001, iou_thres=0.6, max_det=300, use_nms=not cfg.use_dual_head)
        targets += tg
print("inference done", flush=True)

def run_tide(dets, targets, conf, xywh):
    gt, pr = Data("gt", max_dets=300), Data("pred", max_dets=300)
    for i, (d, t) in enumerate(zip(dets, targets)):
        for b, l in zip(t["boxes"].tolist(), t["labels"].tolist()):
            gt.add_ground_truth(i, l, [b[0], b[1], b[2] - b[0], b[3] - b[1]] if xywh else b)
        keep = d["scores"] > conf
        for b, s, l in zip(d["boxes"][keep].tolist(), d["scores"][keep].tolist(), d["labels"][keep].tolist()):
            pr.add_detection(i, l, s, [b[0], b[1], b[2] - b[0], b[3] - b[1]] if xywh else b)
    t = TIDE(); t.evaluate(gt, pr, mode=TIDE.BOX, name="p")
    return {"AP50_tide": float(t.runs["p"].ap), **{k: float(v) for k, v in t.get_main_errors()["p"].items()}, **{k: float(v) for k, v in t.get_special_errors()["p"].items()}}

out = {}
for name, conf, xywh in (("legacy_call(xyxy_as_xywh, conf>0.25)", 0.25, False), ("correct_format(xywh, conf>0.25)", 0.25, True), ("correct_format(xywh, conf>0.001)", 0.001, True)):
    t0 = time.time(); out[name] = run_tide(dets, targets, conf, xywh); out[name]["seconds"] = time.time() - t0
    print(name, json.dumps(out[name]), flush=True)
json.dump(out, open(out_json, "w"), indent=1)
