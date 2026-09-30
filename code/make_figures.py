"""Figures of the paper: real VisDrone validation images, saved predictions of the trained models, and the curated run table (results/final_runs.csv)."""
import io, json, sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from PIL import Image

import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]                                   # repository root
ARCHIVE = Path(os.environ.get("ARCHIVE", REPO.parent / "archive"))           # raw downloads: round*/diagnostics, session*_hf, data/VisDrone2019-DET-val
OUT = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
V = ARCHIVE / "data" / "VisDrone2019-DET-val"
PRED_ZIP = ARCHIVE / "round4" / "diagnostics_with_predictions.zip"
BLUE, ORANGE, GRAY, GREEN, RED, YELLOW, INK = "#0072B2", "#D55E00", "#8a8a8a", "#1a9850", "#d73027", "#f2b705", "#222222"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "axes.edgecolor": "#555555",
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.labelsize": 7, "savefig.dpi": 300})
imgs = sorted(p.name for p in (V / "images").glob("*.jpg"))
zf = zipfile.ZipFile(PRED_ZIP)


def preds(rid):
    d = np.load(io.BytesIO(zf.read(f"runs/{rid}/preds_val_640_o2o.npz")), allow_pickle=True)
    return {k: d[k] for k in d.files}


def gt(name):
    rows = []
    for l in (V / "annotations" / name.replace(".jpg", ".txt")).read_text().strip().splitlines():
        x, y, w, h, s, c, *_ = [int(float(v)) for v in l.split(",")[:8]]
        if 1 <= c <= 10 and w > 0 and h > 0:
            rows.append((x, y, x + w, y + h, c))
    return np.array(rows)


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def match(idx, p, g, thr=0.3):
    m = (p["img"] == idx) & (p["score"] > thr)
    order = np.argsort(-p["score"][m]); boxes, cls = p["xyxy"][m][order], p["cls"][m][order]
    gm = np.zeros(len(g), bool); tp = np.zeros(len(boxes), bool)
    if len(boxes) and len(g):
        I = iou(boxes, g[:, :4].astype(float))
        for i in range(len(boxes)):
            cand = [j for j in np.argsort(-I[i]) if not gm[j] and I[i, j] >= 0.5 and g[j, 4] - 1 == cls[i]]
            if cand:
                gm[cand[0]] = True; tp[i] = True
    return boxes, tp, gm


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=220)
    plt.close(fig)


# ===================================================================================================================== Fig. 1 architecture + real images
def fig_architecture():
    base, mod = preds("v10-base-f640e90-s1"), preds("v10-graftd-f640e90-s1")
    idx = 369
    name = imgs[idx]
    im = Image.open(V / "images" / name).convert("RGB")
    W, H = im.size
    s640 = 640.0 / max(W, H)  # the network input is the image resized to a 640 px long side
    g = gt(name)
    fig = plt.figure(figsize=(4.8, 3.2))
    ax = fig.add_axes([0, 0.40, 1, 0.60]); ax.set_xlim(0, 100); ax.set_ylim(0, 36); ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#444444", fs=6.2, tc=INK, lw=0.7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.15)

    def arrow(p, q, color="#444444", lw=0.8, style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=6, lw=lw, color=color, linestyle=ls, shrinkA=0, shrinkB=0))
    # input thumbnail
    cx0, cy0 = int(W * 0.52), int(H * 0.30)
    crop = im.crop((cx0, cy0, cx0 + 520, cy0 + 330))
    axi = fig.add_axes([0.0, 0.60, 0.2, 0.31]); axi.imshow(crop); axi.axis("off")
    for sp in axi.spines.values(): sp.set_visible(False)
    ax.text(10.0, 31.6, "VisDrone image", ha="center", fontsize=6.3, color=INK)
    # backbone, neck, heads (pretrained, grey)
    grey = "#e3e3e3"
    box(23, 12, 17, 18, "Backbone\nstem, C2f,\nSCDown, SPPF,\nPSA", grey)
    box(46, 18, 16, 12, "PAN neck\nP3 P4 P5", grey)
    for k, nm in enumerate(("P3", "P4", "P5")):
        box(68, 27.5 - k * 5.3, 9, 4.4, f"head {nm}", grey, fs=5.8)
    # new branch (orange)
    box(46, 3.5, 16, 9, "up(P3) + stride-4\nfeature, C2f x2", "#fbd9c3", ec=ORANGE, fs=5.9)
    box(68, 3.5, 9, 4.4, "head P2", "#fbd9c3", ec=ORANGE, fs=5.8)
    ax.text(54, 1.0, "new branch (copied from P3)", ha="center", fontsize=5.6, color=ORANGE)
    arrow((17.5, 21), (23, 21))
    arrow((40, 24), (46, 24))
    arrow((62, 24), (68, 29.7)); arrow((62, 24), (68, 24.4)); arrow((62, 24), (68, 19.1))
    arrow((40, 14.5), (46, 8.5), color=ORANGE, lw=0.9)
    arrow((54, 18), (54, 12.5), color=ORANGE, lw=0.9)
    arrow((62, 8), (68, 5.7), color=ORANGE, lw=0.9)
    ax.text(41.2, 9.9, "stride 4", fontsize=5.4, color=ORANGE, ha="right")
    arrow((77, 29.7), (83, 21)); arrow((77, 24.4), (83, 21)); arrow((77, 19.1), (83, 21)); arrow((77, 5.7), (83, 17.5), color=ORANGE, lw=0.9)
    # output thumbnail with the model's detections
    m = (mod["img"] == idx) & (mod["score"] > 0.3)
    axo = fig.add_axes([0.8, 0.60, 0.2, 0.31]); axo.imshow(crop); axo.axis("off")
    for b in mod["xyxy"][m]:
        x1, y1, x2, y2 = b[0] - cx0, b[1] - cy0, b[2] - cx0, b[3] - cy0
        if x2 > 0 and y2 > 0 and x1 < 520 and y1 < 330:
            axo.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=ORANGE, lw=0.6))
    ax.text(90.0, 31.6, "detections", ha="center", fontsize=6.3, color=INK)
    # bottom row: why stride 4 -- boxes that own no anchor centre (real annotations, network input scale)
    ax2 = fig.add_axes([0.0, 0.0, 1, 0.36]); ax2.axis("off")
    x0, y0, cw, ch = int(cx0 * s640) + 40, int(cy0 * s640) + 6, 150, 95
    small = im.resize((int(W * s640), int(H * s640)), Image.LANCZOS).crop((x0, y0, x0 + cw, y0 + ch))
    gg = g.astype(float) * s640
    for k, (stride, title) in enumerate(((8, "stride 8"), (4, "stride 4"))):
        a = fig.add_axes([0.005 + k * 0.30, 0.01, 0.285, 0.29]); a.imshow(small, extent=(0, cw, ch, 0)); a.set_xlim(0, cw); a.set_ylim(ch, 0)
        for t in np.arange(0, cw + 1, stride): a.axvline(t, color="white", lw=0.25, alpha=0.55)
        for t in np.arange(0, ch + 1, stride): a.axhline(t, color="white", lw=0.25, alpha=0.55)
        n_in = n_miss = 0
        for b in gg:
            bx1, by1, bx2, by2 = b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0
            if bx2 < 0 or by2 < 0 or bx1 > cw or by1 > ch: continue
            n_in += 1
            gx = np.arange(stride / 2, cw + ch + 800, stride)
            has = ((gx > bx1) & (gx < bx2)).any() and ((gx > by1) & (gx < by2)).any()
            n_miss += (not has)
            a.add_patch(Rectangle((bx1, by1), bx2 - bx1, by2 - by1, fill=False, ec=(GREEN if has else RED), lw=0.7))
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_linewidth(0.5)
        a.set_title(f"{title}: {n_miss} of {n_in} unmatched", fontsize=6.0, pad=1.5, color=INK)
    ax2.text(0.63, 0.80, "Red: ground-truth boxes with no grid-cell\ncentre inside, so they cannot become\npositives under centre-based assignment.\nCrop of the 640 px network input.", fontsize=6.0, color=INK, va="top", transform=ax2.transAxes, linespacing=1.35)
    save(fig, "fig_architecture")
    return n_miss


# ===================================================================================================================== Fig. 2 qualitative
def fig_qualitative():
    base, mod = preds("v10-base-f640e90-s1"), preds("v10-graftd-f640e90-s1")
    top = json.load(open(REPO / "code" / "qual_crops.json"))
    pick = [t for t in top if t[3] in (375, 181)][:2]
    fig, axes = plt.subplots(2, 3, figsize=(4.8, 2.0), gridspec_kw={"wspace": 0.03, "hspace": 0.1})
    for r, (gain, ng, nsmall, idx, x0, y0) in enumerate(pick):
        name = imgs[idx]; im = Image.open(V / "images" / name).convert("RGB"); g = gt(name)
        crop = im.crop((x0, y0, x0 + 240, y0 + 150))
        for c, (title, p) in enumerate((("ground truth", None), ("baseline", base), ("P2-Graft", mod))):
            ax = axes[r, c]; ax.imshow(crop); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_linewidth(0.4)
            if p is None:
                for b in g:
                    ax.add_patch(Rectangle((b[0] - x0, b[1] - y0), b[2] - b[0], b[3] - b[1], fill=False, ec=GREEN, lw=0.8))
                lab = ""
            else:
                boxes, tp, gm = match(idx, p, g)
                for b, ok in zip(boxes, tp):
                    ax.add_patch(Rectangle((b[0] - x0, b[1] - y0), b[2] - b[0], b[3] - b[1], fill=False, ec=(GREEN if ok else RED), lw=0.8))
                cxm = (g[:, 0] + g[:, 2]) / 2; cym = (g[:, 1] + g[:, 3]) / 2
                inw = (cxm >= x0) & (cxm < x0 + 240) & (cym >= y0) & (cym < y0 + 150)
                for b, hit in zip(g[inw], gm[inw]):
                    if not hit:
                        ax.add_patch(Rectangle((b[0] - x0, b[1] - y0), b[2] - b[0], b[3] - b[1], fill=False, ec=YELLOW, lw=0.8))
                lab = f"found {int(gm[inw].sum())} of {int(inw.sum())}"
            ax.set_xlim(0, 240); ax.set_ylim(150, 0)
            if r == 0: ax.set_title(title, fontsize=6.8, pad=2)
            if lab: ax.text(4, 146, lab, color="white", fontsize=6.2, fontweight="bold", va="bottom", bbox=dict(fc="black", ec="none", alpha=0.6, pad=1.2))
    save(fig, "fig_qualitative")


# ===================================================================================================================== Fig. 3 gain vs input size, accuracy vs compute
def fig_scaling():
    df = pd.read_csv(REPO / "results" / "final_runs.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.8, 2.1), gridspec_kw={"wspace": 0.42})
    pts = {}
    for cond, sz in (("n640e90", 640), ("n896e90", 896), ("n1280e90", 1280)):
        bb = df[(df.cond == cond) & (df.model == "base")].set_index("seed"); mm = df[(df.cond == cond) & (df.model == "M")].set_index("seed")
        seeds = sorted(set(bb.index) & set(mm.index))
        pts[sz] = [mm.loc[s, "test_AP"] - bb.loc[s, "test_AP"] for s in seeds]
    xs = {640: 0, 896: 1, 1280: 2}
    for sz, v in pts.items():
        a.scatter(np.full(len(v), xs[sz]) + np.linspace(-0.07, 0.07, len(v)) * (len(v) > 1), v, s=9, color=BLUE, zorder=3, lw=0)
        a.hlines(np.mean(v), xs[sz] - 0.22, xs[sz] + 0.22, color=INK, lw=1.0, zorder=2)
        a.text(xs[sz] + 0.3, np.mean(v), f"{np.mean(v):+.2f}".replace("-", "\u2212"), ha="left", va="center", fontsize=6.6, fontweight="bold")
    e150 = df[(df.cond == "n640e150") & (df.model == "M")].test_AP.iloc[0] - df[(df.cond == "n640e150") & (df.model == "base")].test_AP.iloc[0]
    a.scatter([0.42], [e150], marker="s", s=10, color=ORANGE, zorder=3); a.text(0.5, e150, "150 epochs", fontsize=5.8, color=ORANGE, va="center")
    bs = df[(df.cond == "s640e90") & (df.model == "base")].set_index("seed"); ms = df[(df.cond == "s640e90") & (df.model == "M")].set_index("seed")
    sv = [ms.loc[s, "test_AP"] - bs.loc[s, "test_AP"] for s in sorted(bs.index)]
    a.scatter(np.full(3, -0.62) + np.linspace(-0.05, 0.05, 3), sv, marker="^", s=9, color=GRAY, zorder=3); a.text(-0.62, 2.12, "YOLOv10s", fontsize=5.8, color="#555555", ha="center")
    a.axhline(0, color="#555555", lw=0.5)
    a.set_xticks([0, 1, 2]); a.set_xticklabels(["640", "896", "1280"]); a.set_xlim(-0.95, 2.75); a.set_ylim(-0.6, 2.4)
    a.set_xlabel("input size (px)"); a.set_ylabel("test-dev AP gain over baseline")
    a.set_title("(a) gain of the extra level", fontsize=7, loc="left", pad=3)
    # (b) accuracy vs compute
    bsl = {sz: df[(df.cond == c) & (df.model == "base")] for c, sz in (("n640e90", 640), ("n896e90", 896), ("n1280e90", 1280), ("n1536e90", 1536))}
    bx = [bsl[s].gflops.mean() for s in bsl]; by = [bsl[s].test_AP.mean() for s in bsl]
    b.plot(bx, by, "-o", color=BLUE, ms=3.2, lw=1.0, label="baseline, larger input")
    mx = [df[(df.cond == c) & (df.model == "M")].gflops.mean() for c in ("n640e90", "n896e90", "n1280e90")]
    my = [df[(df.cond == c) & (df.model == "M")].test_AP.mean() for c in ("n640e90", "n896e90", "n1280e90")]
    b.plot(mx, my, "-s", color=ORANGE, ms=3.2, lw=1.0, label="with extra level")
    for s_, x_, y_ in zip((640, 896, 1280, 1536), bx, by): b.text(x_ + (1.0 if s_ == 640 else -1.0), y_ + (0.3 if s_ == 640 else -0.5), str(s_), fontsize=5.6, color=BLUE, ha=("left" if s_ == 640 else "right"))
    for s_, x_, y_ in zip((640, 896, 1280), mx, my): b.text(x_ - 1.0, y_ + 0.45, str(s_), fontsize=5.6, color=ORANGE, ha="right")
    b.set_xlabel("deploy GFLOPs"); b.set_ylabel("test-dev AP"); b.set_xticks([10, 20, 30, 40, 50]); b.set_xlim(0, 64); b.set_ylim(13, 25)
    b.legend(frameon=False, fontsize=5.8, loc="lower right", handlelength=1.4)
    b.set_title("(b) accuracy against compute", fontsize=7, loc="left", pad=3)
    save(fig, "fig_scaling")


# ===================================================================================================================== Fig. 4 ablation and seeds
def fig_ablation():
    df = pd.read_csv(REPO / "results" / "final_runs.csv")
    ab = df[df.cond == "abl35"]
    base = ab[ab.model == "baseline"].set_index("seed").test_AP
    order = ["graft, random-init level", "graft, key-matched transfer", "graft, warm start", "deep graft, random-init level", "deep graft, warm start", "3x deep graft, warm start",
             "detail injection 8", "detail injection 16", "detail injection 32", "P2-assisted P3"]
    labels = ["P2 level, random init", "P2 level, key matching", "P2 level, warm start", "deep, random init", "deep, warm start", "3x deep, warm start", "injection, 8 ch", "injection, 16 ch", "injection, 32 ch", "P2-assisted P3"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.8, 2.15), gridspec_kw={"wspace": 0.55, "width_ratios": [1.45, 1]})
    for i, (m, lab) in enumerate(zip(order, labels)):
        d = ab[ab.model == m].set_index("seed").test_AP
        gains = [d[s] - base[s] for s in d.index if s in base.index]
        col = ORANGE if "warm" in m else (BLUE if "injection" in m or "assisted" in m else GRAY)
        a.barh(i, np.mean(gains), color=col, height=0.62, alpha=0.9)
        a.scatter(gains, np.full(len(gains), i), s=5, color=INK, zorder=3, lw=0)
        a.text(max(max(gains), np.mean(gains)) + 0.06, i, f"{np.mean(gains):+.2f}", va="center", fontsize=6.1)
    a.set_yticks(range(len(order))); a.set_yticklabels(labels, fontsize=6); a.invert_yaxis(); a.axvline(0, color="#555555", lw=0.5)
    a.set_xlabel("test-dev AP gain, 35 epochs"); a.set_xlim(-0.1, 1.55)
    a.set_title("(a) variants at 640 px", fontsize=7, loc="left", pad=3)
    bs = df[(df.cond == "s640e90") & (df.model == "base")].set_index("seed"); ms = df[(df.cond == "s640e90") & (df.model == "M")].set_index("seed")
    for s in sorted(bs.index):
        b.plot([0, 1], [bs.loc[s, "test_AP"], ms.loc[s, "test_AP"]], "-o", color=GRAY, ms=3, lw=0.9)
        ypos = {0: -0.07, 1: 0.07, 2: 0.0}[s]
        b.text(1.06, ms.loc[s, "test_AP"] + ypos, f"seed {s}", fontsize=5.8, va="center", color="#444444")
    b.plot([0, 1], [bs.test_AP.mean(), ms.test_AP.mean()], "-o", color=ORANGE, ms=3.6, lw=1.6)
    b.set_xticks([0, 1]); b.set_xticklabels(["baseline", "extra level"], fontsize=6.3); b.set_xlim(-0.25, 1.55); b.set_ylabel("test-dev AP")
    b.set_title("(b) YOLOv10s, 3 seeds", fontsize=7, loc="left", pad=3)
    save(fig, "fig_ablation")


if __name__ == "__main__":
    which = sys.argv[1:] or ["qual", "scaling", "ablation"]   # the architecture figure: code/make_architecture.py
    if "scaling" in which: fig_scaling()
    if "ablation" in which: fig_ablation()
    if "arch" in which: print("arch", fig_architecture())
    if "qual" in which: fig_qualitative()
