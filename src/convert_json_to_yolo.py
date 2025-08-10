import os, json, random, shutil
import cv2
from pathlib import Path

# config 
ROOT = Path(r"C:\Users\sawar\OneDrive\SkinAI")
IMAGES_DIR = ROOT / "data" / "images"
JSON_PATH  = ROOT / "data" / "Acne04-v2_annotations.json"
OUT        = ROOT / "dataset"
SPLIT = 0.8  # train split
random.seed(42)

# class mapping (if severity present)
CLASS_MAP = {"mild":0, "moderate":1, "severe":2}

def yolo_line(cx, cy, r, w, h):
    # bbox centered at (cx,cy) with width=height=2r
    bw, bh = 2*r, 2*r
    # normalize
    return (cx/w, cy/h, bw/w, bh/h)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ["images/train","images/val","labels/train","labels/val"]:
        (OUT/sub).mkdir(parents=True, exist_ok=True)

    data = json.load(open(JSON_PATH, "r", encoding="utf-8"))
    # Expecting COCO-like:
    # data["images"] = [{id, file_name, ...}]
    # data["annotations"] = [{image_id, coordinates:[x,y], radius, class?}, ...]
    images = {img["id"]: img for img in data["images"]}
    anns_by_img = {}
    for ann in data["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    img_ids = list(images.keys())
    random.shuffle(img_ids)
    cut = int(len(img_ids)*SPLIT)
    train_ids, val_ids = set(img_ids[:cut]), set(img_ids[cut:])

    def split_name(img_id):
        return "train" if img_id in train_ids else "val"

    classes_seen = set()

    for img_id, img_meta in images.items():
        # paths
        rel = img_meta["file_name"].lstrip("/\\")
        src_img = IMAGES_DIR / rel
        split = split_name(img_id)
        dst_img = OUT / f"images/{split}" / rel
        dst_lbl = OUT / f"labels/{split}" / (Path(rel).with_suffix(".txt").name)

        # ensure subdirs exist (for nested rel)
        dst_img.parent.mkdir(parents=True, exist_ok=True)

        # read to get size (for normalization)
        img = cv2.imread(str(src_img))
        if img is None:
            print(f"[WARN] missing image: {src_img}")
            # still create empty label to keep pairs aligned
            open(dst_lbl, "w").close()
            continue
        h, w = img.shape[:2]

        # write label lines
        lines = []
        for ann in anns_by_img.get(img_id, []):
            coords = ann.get("coordinates", [None,None])
            if coords is None or len(coords)!=2: continue
            x, y = float(coords[0]), float(coords[1])
            r = float(ann.get("radius", 0))
            if r <= 0: continue

            # class id
            cls = 0
            ann_class = ann.get("class")
            if isinstance(ann_class, str) and ann_class.lower() in CLASS_MAP:
                cls = CLASS_MAP[ann_class.lower()]
                classes_seen.add(cls)
            else:
                classes_seen.add(0)

            xc, yc, bw, bh = yolo_line(x, y, r, w, h)
            # clamp to [0,1]
            xc = min(max(xc,0.0),1.0); yc = min(max(yc,0.0),1.0)
            bw = min(max(bw,0.0),1.0); bh = min(max(bh,0.0),1.0)
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        # save image & labels
        shutil.copy2(src_img, dst_img)
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    print(f"[DONE] Wrote YOLO data to {OUT}")
    print(f"[INFO] classes detected: {sorted(classes_seen)} (count={len(classes_seen)})")
    print(f"[HINT] train images: {(OUT/'images/train').glob('**/*')}")
    print(f"[HINT] val images:   {(OUT/'images/val').glob('**/*')}")

if __name__ == "__main__":
    main()
