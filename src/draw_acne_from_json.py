import os
import json
import cv2
import matplotlib.pyplot as plt

images_path = r"C:\Users\sawar\OneDrive\SkinAI\data\images"
path_labels = r"C:\Users\sawar\OneDrive\SkinAI\data\Acne04-v2_annotations.json"
out_dir = r"C:\Users\sawar\OneDrive\SkinAI\examples"

os.makedirs(out_dir, exist_ok=True)

with open(path_labels, "r", encoding="utf-8") as f:
    labels_dict = json.load(f)

for img_dict in labels_dict["images"]:
    img_id = img_dict["id"]

    # robust path join
    filename = img_dict["file_name"].lstrip("/\\")
    img_path = os.path.join(images_path, filename)

    # read image safely
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Could not read image: {img_path}. Skipping.")
        continue

    # all annotations for this image
    annotations = [
        a for a in labels_dict["annotations"] if a.get("image_id") == img_id
    ]

    # draw circles (guard against weird coord types)
    for ann in annotations:
        coords = ann.get("coordinates")
        radius = int(ann.get("radius", 0))

        # coords expected like (x, y) or [x, y]; convert to tuple
        if isinstance(coords, (list, tuple)) and len(coords) == 2:
            x, y = int(coords[0]), int(coords[1])
            thickness = max(1, int(max(img.shape[:2]) / 1000))
            cv2.circle(img, (x, y), max(1, radius), (255, 0, 0), thickness)
        else:
            print(f"[WARN] Bad coordinates for {filename}: {coords}. Skipping this ann.")

    # preview 
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# save to examples
rel = filename.lstrip("/\\")
out_path = os.path.join(out_dir, rel)
os.makedirs(os.path.dirname(out_path), exist_ok=True)

ok = cv2.imwrite(out_path, img)
if not ok:
    print(f"[WARN] Could not write: {out_path}")
else:
    print(f"[OK] Saved: {out_path}")