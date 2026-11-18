import os
import csv
import torch
from torchvision.utils import save_image


def save_predictions_and_dice(model, loader, device, output_dir):
    """Her test görüntüsü için tahmin maskesini sample_XXX.png olarak kaydeder
    ve Dice skorlarını dice_scores.csv dosyasına yazar.

    Not: sample_XXX sırası, veri setindeki dosya adlarının metin sıralamasını izler
    (0.png, 1.png, 10.png, 11.png, ...). Eşleşme dice_scores.csv'deki
    'source_image' sütununda yazılıdır.
    """
    model.eval()
    dice_scores = []
    os.makedirs(output_dir, exist_ok=True)
    source_names = getattr(loader.dataset, "images", None)

    with torch.no_grad():
        for idx, (images, masks) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            # Dice hesapla
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            dice = (2. * intersection + 1e-8) / (union + 1e-8)
            source = source_names[idx] if source_names is not None else ""
            dice_scores.append((f"sample_{idx:03d}.png", source, dice.item()))

            # Görsel kaydet
            save_image(preds, os.path.join(output_dir, f"sample_{idx:03d}.png"))

    # Dice skorlarını CSV'ye yaz
    with open(os.path.join(output_dir, "dice_scores.csv"), mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "source_image", "dice_score"])
        writer.writerows(dice_scores)

    scores = [d[2] for d in dice_scores]
    print(f"Ortalama Dice: {sum(scores) / len(scores):.4f} (min {min(scores):.4f}, max {max(scores):.4f})")
