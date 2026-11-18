"""Kaggle Retina Blood Vessel veri setini train / val / test olarak ayırır.

Beklenen girdi (Kaggle'dan indirilen arşivin 'Data' klasörü):
    KAGGLE_dataset/train/image, KAGGLE_dataset/train/mask   (80 görüntü)
    KAGGLE_dataset/test/image,  KAGGLE_dataset/test/mask    (20 görüntü)

Çıktı:
    data/kaggle_split/train  -> 64 görüntü (%80)
    data/kaggle_split/val    -> 16 görüntü (%20)
    data/kaggle_split/test   -> 20 görüntü (Kaggle'ın ayrı test seti)
"""
import os
import shutil
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE_DIR, "KAGGLE_dataset")
DST = os.path.join(BASE_DIR, "data", "kaggle_split")

train_image_dir = os.path.join(SRC, "train", "image")
train_mask_dir = os.path.join(SRC, "train", "mask")
test_image_dir = os.path.join(SRC, "test", "image")
test_mask_dir = os.path.join(SRC, "test", "mask")

for sub in ["train/images", "train/masks", "val/images", "val/masks", "test/images", "test/masks"]:
    os.makedirs(os.path.join(DST, sub), exist_ok=True)

# Karıştır ve %80 train / %20 validation olarak böl
all_images = sorted(os.listdir(train_image_dir))
random.seed(42)
random.shuffle(all_images)

split_idx = int(0.8 * len(all_images))
train_imgs = all_images[:split_idx]
val_imgs = all_images[split_idx:]


def copy_images(image_list, image_dir, mask_dir, target):
    for img in image_list:
        shutil.copy(os.path.join(image_dir, img), os.path.join(DST, target, "images", img))
        shutil.copy(os.path.join(mask_dir, img), os.path.join(DST, target, "masks", img))


copy_images(train_imgs, train_image_dir, train_mask_dir, "train")
copy_images(val_imgs, train_image_dir, train_mask_dir, "val")
copy_images(sorted(os.listdir(test_image_dir)), test_image_dir, test_mask_dir, "test")

print(f"train: {len(train_imgs)}, val: {len(val_imgs)}, test: {len(os.listdir(test_image_dir))}")
