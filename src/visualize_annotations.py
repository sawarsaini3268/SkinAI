import json
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# paths
DATA_DIR = "../data"
IMG_DIR = os.path.join(DATA_DIR, "images")
ANNOTATION_FILE = os.path.join(DATA_DIR, "Acne04-v2_annotations.json")

# load annotations from json file
with open(ANNOTATION_FILE, 'r') as f:
    annotations = json.load(f)

with open(ANNOTATION_FILE, 'r') as f:
    annotations = json.load(f)

print("Type of annotations:", type(annotations))
print("Keys in annotations:", annotations.keys())

# display an image with bounding boxes 
def visualize(image_name, bboxes):
    img_path = os.path.join(IMG_DIR, image_name)
    image = cv2.imread(img_path)
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

image_id_to_filename = {img["id"]: img["file_name"] for img in annotations["images"]}

# Step 1: Create a dictionary to group boxes per image
image_to_bboxes = {}

for item in annotations["annotations"]:
    image_id = item["image_id"]
    x, y = item["coordinates"]
    r = item["radius"]
    bbox = [x - r, y - r, 2 * r, 2 * r]

    if image_id not in image_to_bboxes:
        image_to_bboxes[image_id] = []

    image_to_bboxes[image_id].append(bbox)

# Step 2: Visualize first few examples
count = 0
for image_id, bboxes in image_to_bboxes.items():
    image_name = image_id_to_filename[image_id]
    visualize(image_name, bboxes)

    count += 1
    if count >= 5:
        break
