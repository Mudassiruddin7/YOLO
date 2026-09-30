"""Curated final tables of the paper: every number comes from the result.json files of the runs (one entry per run id, no double counting)."""
import sys, json
import numpy as np, pandas as pd
import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]                                   # repository root
ARCHIVE = Path(os.environ.get("ARCHIVE", REPO.parent / "archive"))           # raw downloads: round*/diagnostics, session*_hf, data/VisDrone2019-DET-val
sys.path.insert(0, str(REPO / "code"))
import graft_p2_v2 as G

R = ARCHIVE
SRC = {"r2": R / "round2/diagnostics/runs", "r3": R / "round3/diagnostics/runs", "r4": R / "round4/diagnostics/runs", "r5": R / "round5/diagnostics/runs",
       "A": R / "sessionA6_hf/runs", "B": R / "sessionB_hf/sessionB-20261106-1639/runs", "C": R / "sessionC_hf/sessionC-20261107-0846/runs"}


def load(src, rid, cond, model):
    d = SRC[src] / rid / "result.json"
    r = json.load(open(d))
    row = G.flatten_result(r)
    row.update({"src": src, "cond": cond, "model": model, "run_id": rid})
    row["train_batch"] = r["spec"]["batch"]
    return row


rows = []
def add(src, ids, cond, model):
    for rid in ids:
        rows.append(load(src, rid, cond, model))

# ---- 90-epoch comparisons ------------------------------------------------------------------------------------------------
add("r3", ["v10-base-f640e90-s0"], "n640e90", "base"); add("r4", ["v10-base-f640e90-s1", "v10-base-f640e90-s2"], "n640e90", "base")
add("r3", ["v10-graftd-f640e90-s0"], "n640e90", "M"); add("r4", ["v10-graftd-f640e90-s1", "v10-graftd-f640e90-s2"], "n640e90", "M")
add("A", [f"v10-base-f16-896-e90-s{s}" for s in (0, 1, 2)], "n896e90", "base")
add("A", [f"v10-graftd-ws-896-e90-s{s}" for s in (0, 1, 2)], "n896e90", "M")
add("B", ["v10-graftd-ws-896-e90-b32-s1"], "n896e90b32", "M")
add("B", ["v10-base-f16-1280-e90-s0"], "n1280e90", "base"); add("B", ["v10-graftd-ws-1280-e90-s0"], "n1280e90", "M")
add("r5", ["v10-base-f1536e90-s0"], "n1536e90", "base")
add("B", ["v10-base-f16-640-e150-s0"], "n640e150", "base"); add("B", ["v10-graftd-ws-640-e150-s0"], "n640e150", "M")
add("C", [f"v10s-base-f16-640-e90-s{s}" for s in (0, 1, 2)], "s640e90", "base")
add("C", [f"v10s-graftd-ws-640-e90-s{s}" for s in (0, 1, 2)], "s640e90", "M")
# ---- 35-epoch ablation at 640 ----------------------------------------------------------------------------------------------
add("r2", [f"v10-base-f16-s{s}" for s in (0, 1, 2)], "abl35", "baseline")
add("r2", [f"v10-graft-f16-s{s}" for s in (0, 1, 2)], "abl35", "graft, random-init level")
add("r2", [f"v10-graft-key-s{s}" for s in (0, 1, 2)], "abl35", "graft, key-matched transfer")
add("r3", ["v10-graft-ws-s0"], "abl35", "graft, warm start"); add("r4", ["v10-graft-ws-s1", "v10-graft-ws-s2"], "abl35", "graft, warm start")
add("r4", ["v10-graftd-f16-s0"], "abl35", "deep graft, random-init level")
add("r3", [f"v10-graftd-ws-s{s}" for s in (0, 1, 2)], "abl35", "deep graft, warm start")
add("r4", ["v10-graftd3-ws-s0"], "abl35", "3x deep graft, warm start")
add("r3", ["v10-inj16-f16-s0"], "abl35", "detail injection 16"); add("r5", ["v10-inj16-f16-s1", "v10-inj16-f16-s2"], "abl35", "detail injection 16")
add("r5", ["v10-inj8-f16-s0"], "abl35", "detail injection 8"); add("r5", ["v10-inj32-f16-s0"], "abl35", "detail injection 32")
add("r3", ["v10-assist32-f16-s0"], "abl35", "P2-assisted P3")
# ---- detail injection at 90 epochs ------------------------------------------------------------------------------------------
add("r5", ["v10-inj16-f640e90-s0", "v10-inj16-f640e90-s1"], "inj90_640", "inj16"); add("r5", ["v10-inj16-f896e90-s0"], "inj90_896", "inj16")
add("r5", ["v10-inj16-f1280e90-s0"], "inj90_1280", "inj16")
df = pd.DataFrame(rows)
out = REPO / "results"
out.mkdir(parents=True, exist_ok=True)
keep = [c for c in df.columns if not c.endswith(("_perclass_AP50", "_perclass_AP"))]
df[keep].to_csv(out / "final_runs.csv", index=False)
print(len(df), "runs")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
g = df.groupby(["cond", "model"], sort=False)
tab = g.agg(n=("seed", "size"), gflops=("gflops", "mean"), params=("params_deploy_M", "mean"), test=("test_AP", "mean"), test_sd=("test_AP", "std"), val=("val_AP", "mean"), val_sd=("val_AP", "std"),
            test50=("test_AP50", "mean"), tests=("test_APs", "mean"), thr=("throughput_b16", "mean")).round(2)
print(tab.to_string())
print()
for cond in ["n640e90", "n896e90", "n1280e90", "n640e150", "s640e90"]:
    b = df[(df.cond == cond) & (df.model == "base")].set_index("seed"); m = df[(df.cond == cond) & (df.model == "M")].set_index("seed")
    seeds = sorted(set(b.index) & set(m.index))
    dt = [m.loc[s, "test_AP"] - b.loc[s, "test_AP"] for s in seeds]; dv = [m.loc[s, "val_AP"] - b.loc[s, "val_AP"] for s in seeds]
    ds = [m.loc[s, "test_APs"] - b.loc[s, "test_APs"] for s in seeds]
    print(cond, "seeds", seeds, "test gain", np.round(dt, 2), "mean", round(np.mean(dt), 2), "val gain", np.round(dv, 2), "mean", round(np.mean(dv), 2), "APs gain mean", round(np.mean(ds), 2))
