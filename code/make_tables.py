"""LaTeX tables of the paper, generated from results/final_runs.csv (and the bootstrap intervals of the sessions)."""
import numpy as np, pandas as pd
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(REPO / "results" / "final_runs.csv")
B = lambda s: r"\textbf{" + s + "}"
sg = lambda x: ("+" if x >= 0 else "$-$") + f"{abs(x):.2f}"
pm = lambda m, s: f"{m:.2f}" + (f"$\\pm${s:.2f}" if s == s else "")


def stat(sel, col):
    return sel[col].mean(), sel[col].std(ddof=1) if len(sel) > 1 else float("nan")


def gain(cond, model="M", base="base", col="test_AP"):
    b = df[(df.cond == cond) & (df.model == base)].set_index("seed")[col]
    m = df[(df.cond == cond) & (df.model == model)].set_index("seed")[col]
    return [m[s] - b[s] for s in sorted(set(b.index) & set(m.index))]


# ------------------------------------------------------------------ Table 1
rows = []
def row(name, cond, model, bold_gain=False, size=None, gains=None, extra=None, boldap=False):
    sel = df[(df.cond == cond) & (df.model == model)]
    t, ts = stat(sel, "test_AP"); v, vs = stat(sel, "val_AP"); gf = sel.gflops.mean()
    g = "--" if gains is None else (B(sg(np.mean(gains))) if bold_gain else sg(np.mean(gains)))
    ta = B(pm(t, ts)) if boldap else pm(t, ts)
    return f"{name} & {size} & {len(sel)} & {gf:.1f} & {ta} & {pm(v, vs)} & {g} \\\\"
L = [r"\begin{tabular}{llrrrrr}", r"\toprule", r"Model & Input & $n$ & GFLOPs & Test-dev AP & Val AP & Gain \\", r"\midrule"]
L += [r"\multicolumn{7}{l}{\textit{YOLOv10n, 90 epochs}} \\"]
L += [row("baseline", "n640e90", "base", size=640), row("P2-Graft", "n640e90", "M", True, 640, gain("n640e90"), boldap=True)]
L += [r"\addlinespace[1pt]", row("baseline", "n896e90", "base", size=896), row("P2-Graft (batch 16)", "n896e90", "M", True, 896, gain("n896e90"), boldap=True)]
b32 = df[(df.cond == "n896e90b32") & (df.model == "M")].iloc[0]; b1 = df[(df.cond == "n896e90") & (df.model == "base") & (df.seed == 1)].iloc[0]
L += [f"P2-Graft (batch 32) & 896 & 1 & {b32.gflops:.1f} & {b32.test_AP:.2f} & {b32.val_AP:.2f} & {sg(b32.test_AP - b1.test_AP)} \\\\"]
L += [r"\addlinespace[1pt]", row("baseline", "n1280e90", "base", size=1280), row("P2-Graft", "n1280e90", "M", True, 1280, gain("n1280e90"))]
L += [r"\addlinespace[1pt]", row("baseline", "n1536e90", "base", size=1536)]
L += [r"\midrule", r"\multicolumn{7}{l}{\textit{YOLOv10n, 150 epochs}} \\", row("baseline", "n640e150", "base", size=640), row("P2-Graft", "n640e150", "M", True, 640, gain("n640e150"), boldap=True)]
L += [r"\midrule", r"\multicolumn{7}{l}{\textit{YOLOv10s, 90 epochs}} \\", row("baseline", "s640e90", "base", size=640), row("P2-Graft", "s640e90", "M", True, 640, gain("s640e90"), boldap=True)]
L += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_main.tex").write_text("\n".join(L), encoding="utf-8")

# ------------------------------------------------------------------ Table 2 (ablation, 35 epochs, 640 px)
ab = df[df.cond == "abl35"]
base = ab[ab.model == "baseline"].set_index("seed")
order = [("baseline", "YOLOv10n baseline"), ("graft, random-init level", "P2 level, random init"), ("graft, key-matched transfer", "P2 level, key-matched weights"),
         ("graft, warm start", "P2 level, warm start"), ("deep graft, random-init level", "2$\\times$ deeper block, random init"), ("deep graft, warm start", "2$\\times$ deeper block, warm start"),
         ("3x deep graft, warm start", "3$\\times$ deeper block, warm start"), ("detail injection 16", "detail injection, 16 channels"),
         ("P2-assisted P3", "P2-assisted P3, 32 channels")]
L = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"Variant & $n$ & GFLOPs & Test-dev AP & Val AP & Gain \\", r"\midrule"]
for key, name in order:
    sel = ab[ab.model == key]
    t, ts = stat(sel, "test_AP"); v, vs = stat(sel, "val_AP")
    gs = [sel.set_index("seed").test_AP[s] - base.test_AP[s] for s in sel.seed if s in base.index]
    g = "--" if key == "baseline" else sg(np.mean(gs))
    hot = key == "deep graft, warm start"
    L.append(f"{B(name) if hot else name} & {len(sel)} & {sel.gflops.mean():.1f} & {B(pm(t, ts)) if hot else pm(t, ts)} & {pm(v, vs)} & {B(g) if hot else g} \\\\")
    if key in ("baseline", "graft, warm start", "3x deep graft, warm start", "detail injection 16"):
        L.append(r"\midrule")
L[-1] = L[-1] if L[-1] != r"\midrule" else L[-1]
L += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_ablation.tex").write_text("\n".join(L), encoding="utf-8")

# ------------------------------------------------------------------ Table 4 (published results against this work)
def ours(cond, model):
    sel = df[(df.cond == cond) & (df.model == model)]
    return sel.params_deploy_M.mean(), sel.gflops.mean(), sel.val_AP.mean(), sel.val_AP50.mean()
LIT = [  # (name, input, params M, GFLOPs, AP, AP50, source key) as reported by the authors
    ("YOLOv8-N", "640", "3.2", "8.7", "19.6", "33.3", r"\cite{xiao2025fbrt}"),
    ("FBRT-YOLO-N", "640", "0.9", "6.7", "20.2", "34.4", r"\cite{xiao2025fbrt}"),
    ("YOLOv10-S", "640", "7.2", "21.6", "23.8", "39.3", r"\cite{xiao2025fbrt}"),
    ("YOLOv8-S", "640", "11.2", "28.6", "23.6", "39.6", r"\cite{xiao2025fbrt}"),
    ("FBRT-YOLO-S", "640", "2.9", "22.9", "25.9", "42.4", r"\cite{xiao2025fbrt}"),
    ("FBRT-YOLO-M", "640", "7.2", "58.7", "28.4", "45.9", r"\cite{xiao2025fbrt}"),
    ("DroneScan-YOLO", "1280", "10.23", "--", "35.6", "55.3", r"\cite{bellec2026dronescan}"),
]
L = [r"\begin{tabular}{llrrrrl}", r"\toprule", r"Model & Input & Params (M) & GFLOPs & AP & AP$_{50}$ & Source \\", r"\midrule"]
for n, i, p, g, a, a5, c in LIT:
    L.append(f"{n} & {i} & {p} & {g} & {a} & {a5} & {c} \\\\")
L.append(r"\midrule")
mine = [("YOLOv10n baseline", "640", "n640e90", "base", False), ("P2-Graft, YOLOv10n", "640", "n640e90", "M", True),
        ("YOLOv10s baseline", "640", "s640e90", "base", False), ("P2-Graft, YOLOv10s", "640", "s640e90", "M", True),
        ("P2-Graft, YOLOv10n", "1280", "n1280e90", "M", True), ("YOLOv10n baseline", "1536", "n1536e90", "base", True)]
for name, size, cond, model, hot in mine:
    p, g, a, a5 = ours(cond, model)
    f = (lambda x: B(x)) if hot else (lambda x: x)
    L.append(f"{f(name)} & {f(size)} & {f(f'{p:.2f}')} & {f(f'{g:.1f}')} & {f(f'{a:.2f}')} & {f(f'{a5:.2f}')} & \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_lit.tex").write_text("\n".join(L), encoding="utf-8")
print("tab_lit written")
