import os
from pathlib import Path
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT


# =========================
# Device
# =========================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


# =========================
# Config
# =========================
DATA_ROOT = "dataset_4class_v2"
IMG_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 20
LR = 5e-5
NUM_WORKERS = 0
NUM_CLASSES = 4   # 0 background, 1 liver, 2 gallbladder, 3 instrument

BEST_MODEL_PATH = "./best_model_transunet_4class_dice.pth"
CURVE_PATH = "./transunet_4class_dice_curves.png"


# =========================
# Dataset
# =========================
class Cholec4ClassDataset(Dataset):
    def __init__(self, split: str):
        self.img_dir = Path(DATA_ROOT) / split / "images"
        self.mask_dir = Path(DATA_ROOT) / split / "masks"

        self.img_paths = sorted(self.img_dir.glob("*.png"))
        self.mask_paths = [self.mask_dir / p.name for p in self.img_paths]

        valid = []
        for img_p, mask_p in zip(self.img_paths, self.mask_paths):
            if mask_p.exists():
                valid.append((img_p, mask_p))
        self.samples = valid

        print(f"{split}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # C,H,W

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.int64)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)


# =========================
# Dice Loss
# =========================
class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6, ignore_background=True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)  # [B,C,H,W]
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes)  # [B,H,W,C]
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()       # [B,C,H,W]

        start_class = 1 if self.ignore_background else 0
        dices = []

        for c in range(start_class, self.num_classes):
            p = probs[:, c]
            t = targets_onehot[:, c]

            inter = (p * t).sum(dim=(1, 2))
            denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))

            dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
            dices.append(dice)

        dice_mean = torch.stack(dices, dim=0).mean()
        return 1.0 - dice_mean


# =========================
# Metrics
# =========================
def dice_for_class(pred, target, class_idx, eps=1e-7):
    pred_bin = (pred == class_idx).float()
    target_bin = (target == class_idx).float()

    inter = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()

    if denom.item() == 0:
        return float("nan")

    return ((2.0 * inter + eps) / (denom + eps)).item()


def evaluate(model, loader, ce_loss_fn, dice_loss_fn):
    model.eval()
    running_loss = 0.0

    liver_dices = []
    gall_dices = []
    inst_dices = []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(imgs)
            ce = ce_loss_fn(logits, masks)
            dice_loss = dice_loss_fn(logits, masks)
            loss = ce + dice_loss
            running_loss += loss.item() * imgs.size(0)

            pred = torch.argmax(logits, dim=1)

            d1 = dice_for_class(pred, masks, 1)
            d2 = dice_for_class(pred, masks, 2)
            d3 = dice_for_class(pred, masks, 3)

            if not np.isnan(d1):
                liver_dices.append(d1)
            if not np.isnan(d2):
                gall_dices.append(d2)
            if not np.isnan(d3):
                inst_dices.append(d3)

    epoch_loss = running_loss / len(loader.dataset)

    liver_mean = float(np.mean(liver_dices)) if liver_dices else float("nan")
    gall_mean = float(np.mean(gall_dices)) if gall_dices else float("nan")
    inst_mean = float(np.mean(inst_dices)) if inst_dices else float("nan")

    valid_means = [x for x in [liver_mean, gall_mean, inst_mean] if not np.isnan(x)]
    mean_dice = float(np.mean(valid_means)) if valid_means else float("nan")

    return epoch_loss, liver_mean, gall_mean, inst_mean, mean_dice


# =========================
# Build Model
# =========================
def build_model():
    config_vit = CONFIGS_ViT["R50-ViT-B_16"]
    config_vit.n_classes = NUM_CLASSES
    config_vit.n_skip = 3
    config_vit.patches.grid = (IMG_SIZE // 16, IMG_SIZE // 16)

    model = ViT_seg(config_vit, img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    return model


# =========================
# Main
# =========================
def main():
    print("DATA_ROOT:", DATA_ROOT)

    train_ds = Cholec4ClassDataset("train")
    val_ds = Cholec4ClassDataset("val")
    test_ds = Cholec4ClassDataset("test")

    # 检查整个训练集是否包含 0,1,2,3
    vals = set()
    for _, mask_path in train_ds.samples[:100]:
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        vals.update(np.unique(m).tolist())
    print("sample unique labels in train subset:", sorted(vals))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = build_model().to(DEVICE)

    class_weights = torch.tensor([0.2, 1.0, 1.5, 2.0], device=DEVICE)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss_fn = DiceLoss(NUM_CLASSES, ignore_background=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    train_losses = []
    val_losses = []
    val_liver = []
    val_gall = []
    val_inst = []
    val_mean = []

    best_mean = -1.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for step, (imgs, masks) in enumerate(train_loader):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            logits = model(imgs)

            ce = ce_loss_fn(logits, masks)
            dice_loss = dice_loss_fn(logits, masks)
            loss = ce + dice_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

            if step % 50 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | step {step}/{len(train_loader)} | batch_loss={loss.item():.4f}")

        train_loss = running_loss / len(train_loader.dataset)

        val_loss, liver_dice, gall_dice, inst_dice, mean_dice = evaluate(
            model, val_loader, ce_loss_fn, dice_loss_fn
        )

        scheduler.step(mean_dice if not np.isnan(mean_dice) else 0.0)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_liver.append(liver_dice)
        val_gall.append(gall_dice)
        val_inst.append(inst_dice)
        val_mean.append(mean_dice)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"liver_dice={liver_dice:.4f} | "
            f"gall_dice={gall_dice:.4f} | "
            f"inst_dice={inst_dice:.4f} | "
            f"mean_dice={mean_dice:.4f}"
        )

        if not np.isnan(mean_dice) and mean_dice > best_mean:
            best_mean = mean_dice
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, BEST_MODEL_PATH)

    print("\nBest val mean dice:", best_mean)
    print("saved:", BEST_MODEL_PATH)

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_liver, test_gall, test_inst, test_mean = evaluate(
        model, test_loader, ce_loss_fn, dice_loss_fn
    )

    print("\n=== Test Result ===")
    print(f"test_loss={test_loss:.4f}")
    print(f"test_liver_dice={test_liver:.4f}")
    print(f"test_gall_dice={test_gall:.4f}")
    print(f"test_inst_dice={test_inst:.4f}")
    print(f"test_mean_dice={test_mean:.4f}")

    # Plot
    epochs = range(1, EPOCHS + 1)
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, val_losses, marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(epochs, val_mean, marker="o", label="Val Mean Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Mean Dice")
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(epochs, val_liver, marker="o", label="Val Liver Dice")
    plt.plot(epochs, val_gall, marker="o", label="Val Gall Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Liver & Gall")
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(epochs, val_inst, marker="o", label="Val Instrument Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Instrument Dice")
    plt.legend()

    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=300)
    print("saved:", CURVE_PATH)


if __name__ == "__main__":
    main()