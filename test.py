"""Eğitilmiş modeli Kaggle test seti (20 görüntü) üzerinde değerlendirir."""
import os
import torch
from torch.utils.data import DataLoader
from train import UNet, KaggleRetinaDataset
from utils import save_predictions_and_dice

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_images_dir = os.path.join(BASE_DIR, "data", "kaggle_split", "test", "images")
test_masks_dir = os.path.join(BASE_DIR, "data", "kaggle_split", "test", "masks")
model_path = os.path.join(BASE_DIR, "best_unet_kaggle_split_100ep.pth")
output_dir = os.path.join(BASE_DIR, "predictions", "kaggle_test_split_100ep")

if __name__ == "__main__":
    # Modeli yükle
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    test_dataset = KaggleRetinaDataset(test_images_dir, test_masks_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    save_predictions_and_dice(model, test_loader, DEVICE, output_dir)
    print("Tahminler ve Dice skorları kaydedildi:", output_dir)
