import os
import json

# loading paths
annotations_path = r"C:\Users\sawar\OneDrive\SkinAI\data\Acne04-v2_annotations.json"
images_folder = r"C:\Users\sawar\OneDrive\SkinAI\data\images"

# load json file
with open(annotations_path, 'r') as f:
    annotations = json.load(f)

annotated_files = set(img['file_name'] for img in annotations['images'])

# check existing files in the images folder
image_files_on_disk = set(os.listdir(images_folder))

# keep only relevant image files
image_files_on_disk = {f for f in image_files_on_disk if f.endswith('.jpg')}

# compare annotated files with existing files
existing = annotated_files & image_files_on_disk
missing = annotated_files - image_files_on_disk

# results
print(f"Total annotated images in JSON: {len(annotated_files)}")
print(f"Images found on disk: {len(existing)} ✅")
print(f"Images missing: {len(missing)} ❌")

# show missing files
if missing:
    print("\nMissing image files:")
    for fname in sorted(missing):
        print(f" - {fname}")