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

# visualizng some of the first few examples
count = 0
for item in annotations:
    image_name = item['name']
    bboxes = [anno['bbox'] for anno in item['annotations']]
    visualize(image_name, bboxes)
    
    count += 1
    if count >= 5:
        break
