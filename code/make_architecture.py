"""New architecture figure: feature pyramid drawn as bars, with real VisDrone crops and the class-score map of every prediction level of a trained P2-Graft model."""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Polygon, Circle
from PIL import Image

import os
REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(os.environ.get("ARCHIVE", REPO.parent / "archive"))   # raw downloads: data/VisDrone2019-DET-val, sessionB_hf/...
V = ARCHIVE / "data" / "VisDrone2019-DET-val"
W_PT = ARCHIVE / "sessionB_hf" / "sessionB-20261106-1639" / "runs" / "v10-graftd-ws-640-e150-s0" / "weights" / "last.pt"
BLUE, ORANGE, INK, GRAY = "#0072B2", "#D55E00", "#222222", "#9aa3ad"
imgs = sorted(p.name for p in (V / "images").glob("*.jpg"))
name = imgs[369]
im = Image.open(V / "images" / name).convert("RGB")
W, H = im.size
s640 = 640.0 / max(W, H)
rw, rh = int(round(W * s640)), int(round(H * s640))
small = im.resize((rw, rh), Image.LANCZOS)
# crop region (640-px input coordinates)
x0, y0, cw, ch = 320, 175, 200, 125

# ---- run the trained model and capture the score map of each level
from ultralytics import YOLO
model = YOLO(str(W_PT)).model.float().eval()
det = model.model[-1]
cap = {}
h = det.register_forward_pre_hook(lambda m, a: cap.__setitem__("x", [t.detach() for t in a[0]]))
H32 = int(np.ceil(rh / 32) * 32)
canvas = Image.new("RGB", (640, H32), (114, 114, 114)); canvas.paste(small, (0, 0))
x = torch.from_numpy(np.asarray(canvas).astype(np.float32) / 255.0).permute(2, 0, 1)[None]
with torch.no_grad():
    model(x)
h.remove()
maps = []
for i, feat in enumerate(cap["x"]):
    with torch.no_grad():
        lg = det.one2one_cv3[i](feat)
    sc = lg.sigmoid().max(1)[0][0].numpy()           # (H/s, W/s)
    stride = 640 / feat.shape[-1]
    a, b = int(x0 / stride), int(np.ceil((x0 + cw) / stride)); c, d = int(y0 / stride), int(np.ceil((y0 + ch) / stride))
    maps.append((stride, sc[c:d, a:b], (a * stride, b * stride, c * stride, d * stride)))
crop = small.crop((x0, y0, x0 + cw, y0 + ch))
order = [3, 0, 1, 2]                                  # P2, P3, P4, P5 (finest first)
names = {3: "P2", 0: "P3", 1: "P4", 2: "P5"}
print("score maps:", [(names[i], maps[i][1].shape, round(float(maps[i][1].max()), 2)) for i in order])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
fig = plt.figure(figsize=(6.3, 3.35))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 126); ax.set_ylim(0, 67); ax.axis("off")


def arrow(p, q, color=INK, lw=0.9, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=lw, color=color, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}", linestyle=ls))


# ---- input image (left, centred)
ai = fig.add_axes([0.0, 0.31, 0.165, 0.37]); ai.imshow(small); ai.add_patch(Rectangle((x0, y0), cw, ch, fill=False, ec=ORANGE, lw=1.0)); ai.axis("off")
ax.text(10.0, 47.0, "VisDrone image", ha="center", fontsize=6.6, color=INK)
ax.text(10.0, 18.0, "640 px input, crop in orange", ha="center", fontsize=5.6, color="#555555")
# ---- rows: y centres, bar lengths (resolution), labels
rows = {3: 55.5, 0: 40.5, 1: 25.5, 2: 10.5}
blen = {3: 24, 0: 17, 1: 11, 2: 7}
stride_lab = {3: "stride 4", 0: "stride 8", 1: "stride 16", 2: "stride 32"}
bx0, nx0 = 23, 55            # backbone feature bars start, neck bars start
for lvl in order:
    y = rows[lvl]; L = blen[lvl]
    new = lvl == 3
    col_b = "#b9c2cc"
    # backbone feature (bar)
    ax.add_patch(Rectangle((bx0, y - 3.2), L, 6.4, fc=col_b, ec="none"))
    ax.text(bx0 + L / 2, y - 5.6, stride_lab[lvl], ha="center", fontsize=5.6, color="#555555")
    # neck output (bar), new level in orange
    ax.add_patch(Rectangle((nx0, y - 3.2), L, 6.4, fc=(ORANGE if new else BLUE), ec="none", alpha=0.95))
    ax.text(nx0 + L + 2.0, y, names[lvl], fontsize=7, fontweight="bold", color=(ORANGE if new else BLUE), va="center")
    # lateral arrow backbone -> neck
    arrow((bx0 + L + 0.8, y), (nx0 - 0.8, y), color=(ORANGE if new else "#777777"), lw=(1.1 if new else 0.7))
    # head dot and heat map
    ax.add_patch(Circle((92, y), 1.9, fc=(ORANGE if new else BLUE), ec="white", lw=0.8))
    arrow((nx0 + L + 7.0, y), (89.6, y), color=(ORANGE if new else "#777777"), lw=0.7)
    stride, sc, ext = maps[lvl]
    ah = fig.add_axes([0.805, (y - 6.7) / 67, 0.17, 13.4 / 67])
    ah.imshow(crop, extent=(ext[0] - x0, ext[1] - x0, ext[3] - y0, ext[2] - y0), aspect="auto")
    ah.imshow(np.ma.masked_less(sc, 0.12), extent=(ext[0] - x0, ext[1] - x0, ext[3] - y0, ext[2] - y0), cmap="autumn_r", alpha=0.85, vmin=0.12, vmax=1.0, aspect="auto", interpolation="nearest")
    ah.set_xlim(0, cw); ah.set_ylim(ch, 0); ah.set_xticks([]); ah.set_yticks([])
    for sp in ah.spines.values(): sp.set_edgecolor(ORANGE if new else "#999999"); sp.set_linewidth(0.9 if new else 0.5)
# top-down path (neck): upsampling arrows between neck bars
for hi, lo in ((2, 1), (1, 0), (0, 3)):
    ax.add_patch(FancyArrowPatch((nx0 + 1.5, rows[hi] + 3.6), (nx0 + 1.5, rows[lo] - 3.6), arrowstyle="-|>", mutation_scale=5, lw=0.7, color=(ORANGE if lo == 3 else "#777777"), shrinkA=0, shrinkB=0))
ax.text(nx0 - 0.4, 33.0, "up", fontsize=5.4, color="#777777", ha="right"); ax.text(nx0 - 0.4, 48.0, "up", fontsize=5.4, color=ORANGE, ha="right"); ax.text(nx0 - 0.4, 18.0, "up", fontsize=5.4, color="#777777", ha="right")
# input -> backbone
for lvl in order:
    arrow((20.5, 33.0), (bx0 - 0.8, rows[lvl]), color="#9a9a9a", lw=0.5, rad=0.0)
# column headings
for xx, t in ((bx0 + 11, "pretrained backbone"), (nx0 + 12, "neck, new level in orange"), (92, "heads"), (115.0, "class-score map of each level")):
    ax.text(xx, 64.4, t, ha="center", fontsize=6.0, color=INK)
ax.text(115.0, -1.2, "yellow: confident object (crop of the input)", ha="center", fontsize=5.4, color="#555555")
fig.savefig(REPO / "paper" / "figures" / "fig_architecture.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(REPO / "paper" / "figures" / "fig_architecture.png", bbox_inches="tight", pad_inches=0.02, dpi=220)
print("saved")
