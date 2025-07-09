import numpy as np
import cv2
from PIL import Image
from glob import glob
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt


IMG_SIZE = 192  # Same as the size on which the model was trained


# Function for calculating combined loss
def combined_loss(y_true, y_pred):
    # Mean Absolute Error
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))

    # SSIM loss
    ssim_loss = tf.reduce_mean(1 - tf.image.ssim(y_true, y_pred, max_val=1.0))

    return 0.8 * l1 + 0.2 * ssim_loss

# Gaussian mask for blending
def get_gaussian_mask(patch_size):
    center = np.zeros((patch_size, patch_size), dtype=np.float32)
    center[patch_size // 2, patch_size // 2] = 1
    gauss = cv2.GaussianBlur(center, (0, 0), sigmaX=patch_size//6, sigmaY=patch_size//6)
    gauss /= gauss.max()
    return np.expand_dims(gauss, axis=2)

# Patch based inference
def patch_inference(model, img_rgb, patch_size=IMG_SIZE, stride=IMG_SIZE//2):
    h, w = img_rgb.shape[:2]
    output_img = np.zeros_like(img_rgb, dtype=np.float32)
    count_map = np.zeros((h, w, 3), dtype=np.float32)

    gaussian = get_gaussian_mask(patch_size)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            patch = img_rgb[y:y_end, x:x_end, :]
            orig_h, orig_w = patch.shape[:2]

            pad_h = max(0, patch_size - orig_h)
            pad_w = max(0, patch_size - orig_w)
            patch_padded = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

            patch_input = patch_padded.astype(np.float32) / 255.0
            patch_input = np.expand_dims(patch_input, axis=0)

            pred = model.predict(patch_input, verbose=0)[0]
            pred = pred[:orig_h, :orig_w, :]

            weight = gaussian[:orig_h, :orig_w, :]
            output_img[y:y_end, x:x_end, :] += pred * weight
            count_map[y:y_end, x:x_end, :] += weight

    output_img = output_img / np.maximum(count_map, 1e-8)
    output_img = np.clip(output_img, 0, 1)
    output_img = (output_img * 255).astype(np.uint8)
    return output_img


# Loading trained student model
student = load_model("best_student_model.keras", custom_objects={'combined_loss': combined_loss})

# Loading GoPro dataset paths for testing
blur_paths = sorted(glob("./GoPro/test/Blur/*.png"))
gt_paths = sorted(glob("./GoPro/test/Sharp/*.png"))

# SSIM score calculation
ssim_scores = []
for i in range(600,len(blur_paths)):
    blur_img = Image.open(blur_paths[i]).convert("RGB")
    blur_img = np.array(blur_img)

    gt_img = np.array(Image.open(gt_paths[i]).convert("RGB"))

    # Ensuring same shape
    if blur_img.shape != gt_img.shape:
        gt_img = cv2.resize(gt_img, (blur_img.shape[1], blur_img.shape[0]))

    pred_img = patch_inference(student, blur_img)

    score = ssim(pred_img, gt_img, channel_axis=2, data_range=255)
    ssim_scores.append(score)

    # For image visualisation
    """
    if True:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(blur_img)
        plt.title("Blurred")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(pred_img)
        plt.title("Student Output")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(gt_img)
        plt.title("Ground Truth")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
    """

avg_ssim = sum(ssim_scores) / len(ssim_scores)
print(f"\nAverage SSIM between Student output and Ground Truth: {avg_ssim:.4f}")