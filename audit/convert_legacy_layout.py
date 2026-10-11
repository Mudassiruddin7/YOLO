"""Unzip a VisDrone2019-DET zip and write the *legacy notebook's* on-disk layout
(images/<split>/*.jpg, labels/<split>/*.txt in normalised YOLO format, rows with score==0 dropped),
plus a raw-annotation census used by the audit (categories, ignore flags, size bins)."""
import sys, zipfile, json, shutil, collections
from pathlib import Path
from PIL import Image
import numpy as np

zip_path, out_root, split = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
tmp = out_root / "_unz" / split
tmp.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path) as z:
    z.extractall(tmp)
src = next(p for p in tmp.iterdir() if p.is_dir())
img_dir, lbl_dir = out_root / "images" / split, out_root / "labels" / split
img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)
for im in (src / "images").glob("*.jpg"):
    shutil.move(str(im), img_dir / im.name)

cat_counts = collections.Counter(); kept_by_flag = collections.Counter()
areas = []; n_img = 0; per_img = []
for f in sorted((src / "annotations").glob("*.txt")):
    W, H = Image.open(img_dir / f.with_suffix(".jpg").name).size
    lines = []; n_kept = 0
    for row in [x.split(",") for x in f.read_text(encoding="utf-8").strip().splitlines()]:
        x, y, w, h, score, cat = map(int, row[:6])
        cat_counts[(cat, score)] += 1
        if score != 0:
            lines.append(f"{cat-1} {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}\n")
            n_kept += 1
            areas.append((w, h, cat, W, H))
    (lbl_dir / f.name).write_text("".join(lines), encoding="utf-8")
    per_img.append(n_kept); n_img += 1
shutil.rmtree(out_root / "_unz", ignore_errors=True)
a = np.array(areas, dtype=float)
side = np.sqrt(a[:, 0] * a[:, 1])
census = {
    "split": split, "images": n_img, "kept_boxes": int(len(a)),
    "category_x_score_counts": {f"cat{c}_score{s}": n for (c, s), n in sorted(cat_counts.items())},
    "labels_with_class_index_ge_10": int((a[:, 2] - 1 >= 10).sum()),
    "frac_area_lt_8sq": float((a[:, 0] * a[:, 1] < 64).mean()), "frac_area_lt_16sq": float((a[:, 0] * a[:, 1] < 256).mean()),
    "frac_area_lt_32sq": float((a[:, 0] * a[:, 1] < 1024).mean()), "frac_area_gt_256sq": float((a[:, 0] * a[:, 1] > 256 * 256).mean()),
    "frac_sqrtarea_lt_8": float((side < 8).mean()), "frac_sqrtarea_lt_16": float((side < 16).mean()), "frac_sqrtarea_lt_32": float((side < 32).mean()),
    "boxes_per_image_mean": float(np.mean(per_img)), "boxes_per_image_max": int(np.max(per_img)),
}
print(json.dumps(census, indent=1))
(out_root / f"census_{split}.json").write_text(json.dumps(census, indent=1))
