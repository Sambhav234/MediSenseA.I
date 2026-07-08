import os
import random
import shutil

SOURCE_DIR = "D:/my_projects/dataset_xray/chest_xray/train"
OUTPUT_DIR = "D:/my_projects/dataset_xray/balanced_dataset"

VAL_SIZE = 200

normal = os.listdir(os.path.join(SOURCE_DIR, "NORMAL"))
pneumonia = os.listdir(os.path.join(SOURCE_DIR, "PNEUMONIA"))

random.shuffle(normal)
random.shuffle(pneumonia)

min_count = min(len(normal), len(pneumonia))

normal = normal[:min_count]
pneumonia = pneumonia[:min_count]

def split(lst):
    return lst[VAL_SIZE:], lst[:VAL_SIZE]

n_train, n_val = split(normal)
p_train, p_val = split(pneumonia)

for split_type in ["train", "val"]:
    for cls in ["NORMAL", "PNEUMONIA"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split_type, cls), exist_ok=True)

def copy(files, src_folder, dst_folder):
    for f in files:
        shutil.copy(
            os.path.join(src_folder, f),
            os.path.join(dst_folder, f)
        )

copy(n_train, os.path.join(SOURCE_DIR, "NORMAL"),
     os.path.join(OUTPUT_DIR, "train", "NORMAL"))

copy(n_val, os.path.join(SOURCE_DIR, "NORMAL"),
     os.path.join(OUTPUT_DIR, "val", "NORMAL"))

copy(p_train, os.path.join(SOURCE_DIR, "PNEUMONIA"),
     os.path.join(OUTPUT_DIR, "train", "PNEUMONIA"))

copy(p_val, os.path.join(SOURCE_DIR, "PNEUMONIA"),
     os.path.join(OUTPUT_DIR, "val", "PNEUMONIA"))

print("DONE")