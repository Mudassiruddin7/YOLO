"""Headless fallback for the marimo notebook (same code, no UI):  python run_headless.py --work ~/graft_p2_work --budget-hours 9

It calls the same `autopilot()` as the notebook's step 0 (environment check/repair -> data -> static analyses -> smoke test -> calibration -> launch the
background worker), waits for the worker, and reports where the worker put the finished paper (it builds it by itself).
Use the notebook if you can (live monitor); use this on a machine where only a terminal is available.  --limit N runs everything on N images (a test).
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import graft_p2_study as G  # noqa: E402  (the notebook file is a plain importable module)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(G.default_work_dir()))
    ap.add_argument("--budget-hours", type=float, default=8.5)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tiers", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=0, help="0 = choose from the budget")
    ap.add_argument("--iso-size", type=int, default=896)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--template", choices=("lncs", "article"), default="lncs", help="LaTeX template of the generated paper (lncs = Springer LNCS, ICPR/ICPRv)")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--skip-calibration", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true", help="do not require a CUDA GPU (only for tests)")
    ap.add_argument("--limit", type=int, default=0, help="test mode: use only the first N images of every split, tiny epochs, no analyses/calibration/paper")
    ap.add_argument("--no-wait", action="store_true", help="launch the worker and return")
    a = ap.parse_args()

    work = Path(a.work).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    lp = work / "logs" / "headless.log"
    rep = G.autopilot(work, budget_h=a.budget_hours, seeds=tuple(int(s) for s in a.seeds.split(",")), tiers=tuple(int(t) for t in a.tiers.split(",")), epochs=a.epochs,
                      iso_size=a.iso_size, notebook_file=str(HERE / "graft_p2_study.py"), skip_smoke=a.skip_smoke, skip_calibration=a.skip_calibration,
                      need_cuda=not a.allow_cpu, limit=a.limit or None, template=a.template, bootstrap=a.bootstrap, log_path=lp)
    for s in rep["steps"]:
        print(("OK   " if s["ok"] else "FAIL ") + f"{s['step']} ({s['seconds']} s): {str(s['detail'])[:400]}")
    if not rep["ok"]:
        sys.exit(1)
    if a.no_wait or not rep["pid"]:
        return
    G.log(f"worker pid {rep['pid']}; status: {work / 'status.json'}", lp)
    while G.pid_alive(rep["pid"]):
        st = G.jload(work / "status.json", {})
        runs = st.get("runs", {})
        done = sum(1 for r in runs.values() if r.get("state") == "done")
        G.log(f"{done}/{len(runs)} runs done; time left in budget {G.fmt_hours(st.get('deadline', time.time()) - time.time())}", lp)
        time.sleep(300)
    bundle = work / "paper_bundle.zip"
    G.log(f"worker finished; paper bundle: {bundle if bundle.exists() else '(not built - see logs/worker.log and logs/analysis.log)'}", lp)


if __name__ == "__main__":
    main()
