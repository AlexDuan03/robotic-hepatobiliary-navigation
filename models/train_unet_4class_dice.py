import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from unet_model import UNet


# =========================
# 配置
# =========================
TRAIN_IMG_DIR = "data/train/images"
TRAIN_MASK_DIR = "data/train/masks"
VAL_IMG_DIR = "data/val/images"
VAL_MASK_DIR = "data/val/masks"

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 20
LR = 3e-5
NUM_CLASSES = 4

SAVE_PATH = "best_model_unet_4class_dice.pth"
CURVE_PATH = "unet_4class_dice_curves.png"

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)


# =========================
# RGB -> 4类映射
# 0 background
# 1 liver
# 2 gallbladder
# 3 instrument
# =========================
RGB_TO_CLASS = {
    (127, 127, 127): 0,
    (255, 255, 255): 0,
    (231, 70, 156): 0,
    (186, 183, 75): 0,   # fat -> background
    (210, 140, 140): 0,
    (111, 74, 0): 0,
    (255, 0, 0): 0,
    (0, 50, 128): 0,

    (255, 114, 114): 1,  # liver

    (255, 160, 165): 2,  # gallbladder
    (255, 85, 0): 2,     # fascia on gallbladder -> gallbladder

    (169, 255, 184): 3,  # instrument
    (170, 255, 0): 3
}


# =========================
# 数据集
# =========================
class Cholec4ClassDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=224):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.img_size = img_size

        assert len(self.img_paths) == len(self.mask_paths), "图像和mask数量不一致"

    def __len__(self):
        return len(self.img_paths)

    def rgb_mask_to_class_mask(self, mask_rgb):
        h, w, _ = mask_rgb.shape
        class_mask = np.zeros((h, w), dtype=np.uint8)

        for rgb, cls in RGB_TO_CLASS.items():
            match = np.all(mask_rgb == np.array(rgb, dtype=np.uint8), axis=-1)
            class_mask[match] = cls

        return class_mask

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = self.rgb_mask_to_class_mask(mask)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)


# =========================
# Dice Loss
# =========================
class DiceLoss(nn.Module):
    def __init__(self, num_classes=4, smooth=1e-6, ignore_background=True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

        start_cls = 1 if self.ignore_background else 0
        dice_losses = []

        for cls in range(start_cls, self.num_classes):
            prob_cls = probs[:, cls]
            target_cls = targets_onehot[:, cls]

            intersection = (prob_cls * target_cls).sum(dim=(1, 2))
            union = prob_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2))

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_losses.append(1.0 - dice.mean())

        return torch.stack(dice_losses).mean()


# =========================
# Dice metric
# =========================
def dice_per_class(logits, targets, num_classes=4):
    preds = torch.argmax(logits, dim=1)

    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()

    dices = {}

    for cls in range(num_classes):
        pred_cls = (preds == cls).astype(np.uint8)
        target_cls = (targets == cls).astype(np.uint8)

        inter = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()

        if union == 0:
            dices[cls] = np.nan
        else:
            dices[cls] = (2.0 * inter) / (union + 1e-8)

    return dices


def mean_dice_no_bg(dice_dict):
    vals = []
    for cls in [1, 2, 3]:
        if not np.isnan(dice_dict[cls]):
            vals.append(dice_dict[cls])
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))


# =========================
# 训练 / 验证
# =========================

def train_one_epoch(model, loader, optimizer, dice_loss, ce_loss):
    model.train()
    epoch_loss = 0.0
    epoch_dice = []
    liver_dice = []
    gallbladder_dice = []

    for imgs, masks in loader:
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = 0.5 * dice_loss(logits, masks) + 0.5 * ce_loss(logits, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        d = dice_per_class(logits, masks, NUM_CLASSES)
        epoch_dice.append(mean_dice_no_bg(d))
        if not np.isnan(d[1]):
            liver_dice.append(d[1])
        if not np.isnan(d[2]):
            gallbladder_dice.append(d[2])

    return (
        epoch_loss / len(loader),
        float(np.mean(epoch_dice)),
        float(np.mean(liver_dice)) if len(liver_dice) > 0 else 0.0,
        float(np.mean(gallbladder_dice)) if len(gallbladder_dice) > 0 else 0.0
    )


@torch.no_grad()
def val_one_epoch(model, loader, dice_loss, ce_loss):
    model.eval()
    epoch_loss = 0.0
    epoch_dice = []

    liver_dice = []
    gallbladder_dice = []

    for imgs, masks in loader:
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        logits = model(imgs)
        loss = 0.5 * dice_loss(logits, masks) + 0.5 * ce_loss(logits, masks)

        epoch_loss += loss.item()

        d = dice_per_class(logits, masks, NUM_CLASSES)
        epoch_dice.append(mean_dice_no_bg(d))
        if not np.isnan(d[1]):
            liver_dice.append(d[1])
        if not np.isnan(d[2]):
            gallbladder_dice.append(d[2])

    return (
        epoch_loss / len(loader),
        float(np.mean(epoch_dice)),
        float(np.mean(liver_dice)) if len(liver_dice) > 0 else 0.0,
        float(np.mean(gallbladder_dice)) if len(gallbladder_dice) > 0 else 0.0
    )


# =========================
# 画曲线
# =========================
def plot_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.title("Dice Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, history["train_dice"], label="train")
    plt.plot(epochs, history["val_dice"], label="val")
    plt.title("Mean Dice (no background)")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, history["train_liver_dice"], label="train")
    plt.plot(epochs, history["val_liver_dice"], label="val")
    plt.title("Liver Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, history["train_gallbladder_dice"], label="train")
    plt.plot(epochs, history["val_gallbladder_dice"], label="val")
    plt.title("Gallbladder Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# =========================
# 主函数
# =========================
def main():
    print("device:", DEVICE)

    train_set = Cholec4ClassDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, IMG_SIZE)
    val_set = Cholec4ClassDataset(VAL_IMG_DIR, VAL_MASK_DIR, IMG_SIZE)

    print("train:", len(train_set))
    print("val:", len(val_set))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = UNet(num_classes=NUM_CLASSES).to(DEVICE)
    dice_loss = DiceLoss(num_classes=NUM_CLASSES, ignore_background=True)

    class_weights = torch.tensor([0.2, 1.0, 2.5, 1.2], dtype=torch.float32).to(DEVICE)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',        # 因为我们监控 val dice
    factor=0.5,        # 学习率减半
    patience=2,        # 2个epoch不提升就降lr
    
)

    best_val_dice = -1.0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "train_liver_dice": [],
        "val_liver_dice": [],
        "train_gallbladder_dice": [],
        "val_gallbladder_dice": []
    }

    for epoch in range(EPOCHS):
        train_loss, train_dice, train_liver, train_gall = train_one_epoch(
            model, train_loader, optimizer, dice_loss, ce_loss
        )
        val_loss, val_dice, val_liver, val_gall = val_one_epoch(
            model, val_loader, dice_loss, ce_loss
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        history["train_liver_dice"].append(train_liver)
        history["val_liver_dice"].append(val_liver)
        history["train_gallbladder_dice"].append(train_gall)
        history["val_gallbladder_dice"].append(val_gall)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
            f"train dice {train_dice:.4f} | val dice {val_dice:.4f} | "
            f"train liver {train_liver:.4f} | val liver {val_liver:.4f} | "
            f"train gall {train_gall:.4f} | val gall {val_gall:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"saved best model -> {SAVE_PATH}")

        scheduler.step(val_dice)

    plot_curves(history, CURVE_PATH)
    print(f"saved curves -> {CURVE_PATH}")


if __name__ == "__main__":
    main()