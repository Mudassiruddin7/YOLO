"""Tidy the Hugging Face dataset repo that holds the runs and weights of the paper.

    set HF_TOKEN=<token with write access>      (PowerShell: $env:HF_TOKEN = "...")
    python tools/organize_hf.py                  # dry run: prints what would change
    python tools/organize_hf.py --apply          # does it

What it does (nothing else is touched):
  * deletes the two interrupted copies of session A (`sessionA-20261103-0052`, `sessionA-20261103-1538`); the finished session is `sessionA-20261104-0048`;
  * deletes the runs of session A that were started by mistake inside the session C folder, and the START.txt marker files;
  * uploads the dataset card (hf_card/README.md), its images (assets/*.png), the paper PDF and the results tables (results/*.csv of this repository).
"""
import argparse, os, sys
from pathlib import Path

REPO_ID = "mudassiruddin/yolo"
DEAD_FOLDERS = ["sessionA-20261103-0052", "sessionA-20261103-1538"]
STRAY_IN_C = ["sessionC-20261107-0846/runs/v10-graftd-ws-896-e90-s0", "sessionC-20261107-0846/runs/v10-graftd-ws-896-e90-s1", "sessionC-20261107-0846/runs/v10-base-f16-896-e90-s0",
              "sessionC-20261107-0846/runs/v10-base-f16-896-e90-s1", "sessionC-20261107-0846/runs/v10-graftd-ws-896-e90-s2", "sessionC-20261107-0846/runs/v10-base-f16-896-e90-s2"]
ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("set HF_TOKEN first")
    api = HfApi(token=token)
    files = [f.path for f in api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True) if hasattr(f, "size")]
    ops = []
    for f in files:
        top = f.split("/")[0]
        if top in DEAD_FOLDERS or f.endswith("/START.txt") or any(f.startswith(s + "/") for s in STRAY_IN_C):
            ops.append(CommitOperationDelete(path_in_repo=f))
    card = ROOT / "hf_card" / "README.md"
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card)))
    for csv in sorted((ROOT / "results").glob("*.csv")):
        ops.append(CommitOperationAdd(path_in_repo=f"results/{csv.name}", path_or_fileobj=str(csv)))
    for png in sorted((ROOT / "assets").glob("*.png")):          # images shown in the dataset card
        ops.append(CommitOperationAdd(path_in_repo=f"assets/{png.name}", path_or_fileobj=str(png)))
    ops.append(CommitOperationAdd(path_in_repo="paper/main.pdf", path_or_fileobj=str(ROOT / "paper" / "main.pdf")))
    n_del = sum(isinstance(o, CommitOperationDelete) for o in ops)
    print(f"{n_del} files to delete, {len(ops) - n_del} files to add")
    for o in ops[:12]:
        print(" ", type(o).__name__.replace("CommitOperation", ""), o.path_in_repo)
    if not a.apply:
        print("dry run: nothing changed (use --apply)")
        return
    api.create_commit(REPO_ID, repo_type="dataset", operations=ops, commit_message="Tidy: drop interrupted sessions and stray runs, add dataset card and results tables")
    print("done")


if __name__ == "__main__":
    main()
