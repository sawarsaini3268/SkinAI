import json
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# paths
DATA_DIR = "../data"
IMG_DIR = os.path.join(DATA_DIR, "images")
ANNOTATION_FILE = os.path.join(DATA_DIR, "Acne04-v2_annotations.json")

# load annotations
with open(ANNOTATION_FILE, 'r') as f:
    annotations = json.load(f)

print("Type of annotations:", type(annotations))
print("Keys in annotations:", annotations.keys())

# visualization function
def visualize(image_name, bboxes):
    img_path = os.path.join(IMG_DIR, image_name)
    
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        return
    
    image = cv2.imread(img_path)
    if image is None:
        print(f"Unable to load image (possibly corrupted): {img_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box in bboxes:
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

    plt.title(image_name)
    plt.axis('off')
    plt.show()

# map image_id to filename 
image_id_to_filename = {img["id"]: img["file_name"] for img in annotations["images"]}

# group bboxes by image_id 
image_to_bboxes = {}

for item in annotations["annotations"]:
    image_id = item["image_id"]
    x, y = item["coordinates"]
    r = item["radius"]
    bbox = [x - r, y - r, 2 * r, 2 * r]  # convert circle to square bbox

    if image_id not in image_to_bboxes:
        image_to_bboxes[image_id] = []

    image_to_bboxes[image_id].append(bbox)

# visualize first few images with annotations 
count = 0
for image_id, bboxes in image_to_bboxes.items():
    image_name = image_id_to_filename.get(image_id)

    if not image_name:
        print(f"Image ID {image_id} not found in image list.")
        continue

    visualize(image_name, bboxes)

    count += 1
    if count >= 5:
        break
