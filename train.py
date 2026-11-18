import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Basit UNet modeli ---
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        self.enc1 = conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        return out

# --- Dataset sınıfı ---
class KaggleRetinaDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.masks = sorted([f for f in os.listdir(masks_dir) if f.endswith(('.png'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        else:
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)
        return image, mask

# --- Dice Loss ---
def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

# --- Combined Loss ---
def focal_dice_loss(pred, target):
    fl = FocalLoss()(pred, target)
    dl = dice_loss(pred, target)
    return fl + dl

# --- Eğitim fonksiyonu ---
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(dataloader, desc="Training"):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = focal_dice_loss(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# --- Validasyon fonksiyonu ---
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, masks in tqdm(dataloader, desc="Validation"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = focal_dice_loss(preds, masks)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# --- Dice skor hesaplama ---
def dice_score(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

# --- Ana Eğitim Scripti (python train.py) ---
if __name__ == "__main__":

    BATCH_SIZE = 4
    EPOCHS = 100
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Kaggle veri yolları ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SPLIT_DIR = os.path.join(BASE_DIR, "data", "kaggle_split")
    train_images_dir = os.path.join(SPLIT_DIR, "train", "images")
    train_masks_dir = os.path.join(SPLIT_DIR, "train", "masks")
    val_images_dir = os.path.join(SPLIT_DIR, "val", "images")
    val_masks_dir = os.path.join(SPLIT_DIR, "val", "masks")
    model_path = os.path.join(BASE_DIR, "best_unet_kaggle_split_100ep.pth")

    # --- Veri seti ---
    train_dataset = KaggleRetinaDataset(train_images_dir, train_masks_dir)
    val_dataset = KaggleRetinaDataset(val_images_dir, val_masks_dir)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- Model, optim ---
    model = UNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # --- Eğitim ve Validasyon ---
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    trigger_times = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_loss = validate(model, val_loader, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"✅ Best model saved with val loss: {best_val_loss:.4f}")
            trigger_times = 0
        else:
            trigger_times += 1
            print(f"No improvement: Trigger count: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # --- Kaybı görselleştir ---
    epochs_run = range(1, len(train_losses) + 1)
    plt.plot(epochs_run, train_losses, label="Train Loss")
    plt.plot(epochs_run, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    os.makedirs(os.path.join(BASE_DIR, "images"), exist_ok=True)
    plt.savefig(os.path.join(BASE_DIR, "images", "loss_curve.png"))
    plt.show()


