"""Test görüntüsü / gerçek maske / tahmin edilen maske karşılaştırma görselleri üretir.

test.py, test görüntülerini dosya adlarının metin sırasına göre işler
(0.png, 1.png, 10.png, 11.png, ...) ve tahminleri sample_000.png, sample_001.png ...
olarak kaydeder. Bu yüzden eşleştirme aynı sıralamayla yapılır.
"""
import os
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "data", "kaggle_split", "test", "images")
true_mask_dir = os.path.join(BASE_DIR, "data", "kaggle_split", "test", "masks")
pred_mask_dir = os.path.join(BASE_DIR, "predictions", "kaggle_test_split_100ep")
output_dir = os.path.join(BASE_DIR, "images")

N = 5  # Kaç örnek görselleştirilecek

if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    # Dataset ile aynı sıralama (sorted) -> sample_i, test_images[i]'ye karşılık gelir
    test_images = sorted(f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg")))

    for i in range(N):
        name = test_images[i]
        img = Image.open(os.path.join(image_dir, name)).convert("RGB")
        true_mask = Image.open(os.path.join(true_mask_dir, name)).convert("L")
        pred_mask = Image.open(os.path.join(pred_mask_dir, f"sample_{i:03}.png")).convert("L")

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(img)
        axs[0].set_title(f"Original Image ({name})")
        axs[1].imshow(true_mask, cmap="gray")
        axs[1].set_title("Ground Truth Mask")
        axs[2].imshow(pred_mask, cmap="gray")
        axs[2].set_title("Predicted Mask")
        for ax in axs:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparison_sample_{i:03}.png"))
        plt.close()
        print(f"comparison_sample_{i:03}.png <- {name}")
