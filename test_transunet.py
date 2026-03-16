import glob
import math
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm


IMG_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 0
DATA_ROOT = "data_811"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def build_color_map(mask_paths, max_samples=200):
    color_set = set()
    sample_paths = mask_paths[:max_samples]

    for p in tqdm(sample_paths, desc="Scanning train mask colors"):
        mask = cv2.imread(p)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        colors = np.unique(mask.reshape(-1, 3), axis=0)
        for c in colors:
            color_set.add(tuple(c.tolist()))

    color_list = sorted(list(color_set))
    color2idx = {c: i for i, c in enumerate(color_list)}
    return color2idx


class CholecDataset(Dataset):
    def __init__(self, mode="test", color2idx=None):
        self.img_paths = sorted(glob.glob(f"{DATA_ROOT}/{mode}/images/*.png"))
        self.mask_paths = sorted(glob.glob(f"{DATA_ROOT}/{mode}/masks/*.png"))

        if len(self.img_paths) == 0:
            raise ValueError(f"No images found in {DATA_ROOT}/{mode}/images")
        if len(self.mask_paths) == 0:
            raise ValueError(f"No masks found in {DATA_ROOT}/{mode}/masks")
        if len(self.img_paths) != len(self.mask_paths):
            raise ValueError("image count != mask count")

        self.color2idx = color2idx

    def __len__(self):
        return len(self.img_paths)

    def rgb_mask_to_label(self, mask):
        h, w, _ = mask.shape
        flat = mask.reshape(-1, 3)
        label_flat = np.zeros((flat.shape[0],), dtype=np.int64)

        for color, idx in self.color2idx.items():
            match = np.all(flat == np.array(color), axis=1)
            label_flat[match] = idx

        return label_flat.reshape(h, w)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        mask = cv2.imread(self.mask_paths[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = self.rgb_mask_to_label(mask)
        mask = torch.from_numpy(mask).long()

        return img, mask


class SimpleTransUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            num_classes=0
        )

        embed_dim = self.encoder.num_features

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, 2, 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 2, 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        x = self.encoder.forward_features(x)

        if x.dim() == 3:
            x = x[:, 1:, :]
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            x = x.permute(0, 2, 1).reshape(B, C, H, W)

        x = self.decoder(x)
        return x


def multiclass_dice(logits, target, num_classes, eps=1e-6):
    pred = torch.argmax(logits, dim=1)
    dice_list = []

    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        if target_c.sum() == 0:
            continue

        inter = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2.0 * inter + eps) / (union + eps)
        dice_list.append(dice)

    if len(dice_list) == 0:
        return 0.0

    return torch.mean(torch.stack(dice_list)).item()


@torch.no_grad()
def evaluate(model, loader, criterion, num_classes):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    pbar = tqdm(loader, desc="Test")
    for imgs, masks in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, masks)
        dice = multiclass_dice(outputs, masks, num_classes)

        total_loss += loss.item()
        total_dice += dice

        pbar.set_postfix({
            "test_loss": f"{loss.item():.4f}",
            "test_dice": f"{dice:.4f}"
        })

    return total_loss / len(loader), total_dice / len(loader)


def main():
    train_mask_paths = sorted(glob.glob(f"{DATA_ROOT}/train/masks/*.png"))
    color2idx = build_color_map(train_mask_paths, max_samples=200)
    num_classes = len(color2idx)

    test_ds = CholecDataset("test", color2idx=color2idx)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Test images: {len(test_ds)}")
    print(f"Num classes: {num_classes}")

    model = SimpleTransUNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("best_transunet.pth", map_location=device))

    criterion = nn.CrossEntropyLoss()

    test_loss, test_dice = evaluate(model, test_loader, criterion, num_classes)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Dice: {test_dice:.4f}")


if __name__ == "__main__":
    main()