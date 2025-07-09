# image-sharpening-kd

# Image Sharpening via Knowledge Distillation using the Restormer model

A lightweight deep learning-based image sharpening system trained via **knowledge distillation**, where a powerful transformer-based **Restormer** model serves as the teacher and a compact CNN learns to enhance blurred images efficiently.

---

## 📌 Introduction

This project:
- Uses a **Restormer** as a high-capacity **teacher** model.
- Trains a lightweight **CNN student** to mimic the teacher using **pseudo-labels**.
- Enables real-time image enhancement on **Full HD (1920×1080)** resolution.

---

## 🎯 Objective

- ✅ The student model should achieve near teacher-level SSIM with reduced inference time.

---

## 📂 Dataset & Preprocessing

- **Dataset**: [GoPro Deblurring Dataset](https://seungjunnah.github.io/Datasets/gopro)
- **Image Resolution**: 1920×1080

**Preprocessing Steps**:
- Pass blurred images through Restormer to generate sharp pseudo-labels.
- Train on randomly cropped **192×192 patches** with:
  - Horizontal & vertical flips
  - Random brightness adjustment

---

## Model Architecture

### 🧑‍🏫 Teacher Model — Restormer

- [Restormer: Efficient Transformer for High-Resolution Image Restoration](https://github.com/swz30/Restormer)
- A transformer-based model pre-trained on motion deblurring tasks.
- Employs **patch-based inference** with **Gaussian blending** for seamless full-resolution reconstruction.

### 👨‍🎓 Student Model — Lightweight CNN

- Designed for low-resource inference.
- Trained on smaller patches using pseudo-labels from the teacher.

**Architecture Overview**:
- Encoder:
  - SeparableConv2D(32) → SeparableConv2D(64) → MaxPooling
- Bottleneck:
  - SeparableConv2D(128) × 2
- Decoder:
  - UpSampling → SeparableConv2D(64) → SeparableConv2D(32) → Conv2D(3 filters)

---

## 📊 Quantitative Results

| Model               | SSIM  |
|---------------------|-------|
| Restormer (Teacher) | 0.875 |
| Student       | 0.803 |

---

## 🖼️ Visual Quality

- Blur was significantly reduced.
- Fine details and sharp edges were preserved.
- Performance drops on images with **heavy motion blur**.

---
