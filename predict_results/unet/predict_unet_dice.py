import os
import argparse
import random
import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

from unet_model import UNet


# 4-class
# 0 background
# 1 liver
# 2 gallbladder
# 3 instrument
CLASS_COLORS = {
    0: (0, 0, 0),
    1: (255, 114, 114),   # liver
    2: (255, 160, 165),   # gallbladder
    3: (170, 255, 0),     # instrument
}


def build_transform(img_size: int):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor()
    ])


def colorize_mask(mask_idx: np.ndarray) -> np.ndarray:
    h, w = mask_idx.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in CLASS_COLORS.items():
        color_mask[mask_idx == cls_idx] = color
    return color_mask


def overlay_mask(image_rgb: np.ndarray, color_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(image_rgb, 1 - alpha, color_mask, alpha, 0)


def load_model(checkpoint_path: str, device: torch.device):
    model = UNet(num_classes=4)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def pick_random_images(test_dir: str, num_images: int = 5):
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"test_dir 不存在: {test_dir}")

    files = [
        f for f in os.listdir(test_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if len(files) == 0:
        raise RuntimeError(f"{test_dir} 里没有图片")

    selected = random.sample(files, min(num_images, len(files)))
    return [os.path.join(test_dir, f) for f in selected]


@torch.no_grad()
def predict_one(model, image_path: str, output_dir: str, img_size: int, device: torch.device):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")

    os.makedirs(output_dir, exist_ok=True)

    image_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image_pil.size
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    transform = build_transform(img_size)
    x = transform(image_pil).unsqueeze(0).to(device)

    logits = model(x)
    pred_idx = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    pred_idx = cv2.resize(pred_idx, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    print(f"{image_name} unique pred:", np.unique(pred_idx))

    pred_mask = pred_idx.astype(np.uint8)
    pred_color = colorize_mask(pred_idx)

    image_rgb = np.array(image_pil)
    pred_overlay = overlay_mask(image_rgb, pred_color, alpha=0.45)

    mask_path = os.path.join(output_dir, f"{image_name}_mask.png")
    color_path = os.path.join(output_dir, f"{image_name}_color.png")
    overlay_path = os.path.join(output_dir, f"{image_name}_overlay.png")

    Image.fromarray(pred_mask).save(mask_path)
    Image.fromarray(pred_color).save(color_path)
    Image.fromarray(pred_overlay).save(overlay_path)

    print("saved:", mask_path)
    print("saved:", color_path)
    print("saved:", overlay_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best_model_unet_4class_dice.pth", help="模型权重路径")
    parser.add_argument("--image", type=str, default=None, help="单张图片路径；不填就随机选")
    parser.add_argument("--test_dir", type=str, default="dataset_4class/test/images", help="测试集图片目录")
    parser.add_argument("--output_dir", type=str, default="predict_results/unet_dice", help="输出目录")
    parser.add_argument("--img_size", type=int, default=224, help="输入尺寸，要和训练一致")
    parser.add_argument("--num_images", type=int, default=5, help="随机预测图片数量")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("device:", device)

    model = load_model(args.checkpoint, device)
    print("model loaded")

    if args.image is not None:
        image_paths = [args.image]
    else:
        image_paths = pick_random_images(args.test_dir, args.num_images)

    print("要预测的图片:")
    for p in image_paths:
        print(p)

    for image_path in image_paths:
        predict_one(
            model=model,
            image_path=image_path,
            output_dir=args.output_dir,
            img_size=args.img_size,
            device=device
        )


if __name__ == "__main__":
    main()