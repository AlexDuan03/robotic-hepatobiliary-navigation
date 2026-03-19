# Robotic Hepatobiliary Navigation

Semantic scene understanding for robotic hepatobiliary surgery using deep learning-based segmentation.

---

## 🧠 Overview

This project explores surgical scene segmentation to support robotic intraoperative navigation and path planning. We compare three representative models:

- **UNet** (baseline CNN)
- **TransUNet** (CNN + Transformer hybrid)
- **nnUNet** (self-configuring state-of-the-art framework)

The system identifies key anatomical structures:

- Liver
- Gallbladder
- Surgical instruments

---

## 📊 Model Comparison

We visualize segmentation results using overlay maps for intuitive comparison.

### Example Results

| Original | UNet | TransUNet | nnUNet |
|---------|------|-----------|--------|
| ![](results/compare/153_compare.png) |

> Additional examples are provided in `results/compare/`

---

## 🏗️ Project Structure
