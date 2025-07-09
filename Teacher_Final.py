# Add Restormer to path for importing
import sys
sys.path.append('./Restormer/basicsr/models/archs')

import os, cv2, torch
import numpy as np
from PIL import Image
from glob import glob
from torchvision.transforms.functional import to_tensor
from restormer_arch import Restormer
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt


# Gaussian mask for blending
def get_gaussian_mask(patch_size):
    center = np.zeros((patch_size, patch_size), dtype=np.float32)
    center[patch_size // 2, patch_size // 2] = 1
    gauss = gaussian_filter(center, sigma=patch_size // 6)
    gauss /= gauss.max()
    return np.expand_dims(gauss, axis=2)

# Patch-based inference with Gaussian blending
def patch_inference(model, img_rgb, patch_size=192, stride=96):
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

            input_tensor = to_tensor(patch_padded).unsqueeze(0).cuda()
            with torch.no_grad():
                out = model(input_tensor)[0].squeeze().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            torch.cuda.empty_cache()

            out = out[:orig_h, :orig_w, :]
            weight = gaussian[:orig_h, :orig_w, :]
            output_img[y:y_end, x:x_end, :] += out * weight
            count_map[y:y_end, x:x_end, :] += weight

    output_img = output_img / np.maximum(count_map, 1e-8)
    output_img = np.clip(output_img, 0, 1)
    output_img = (output_img * 255).astype(np.uint8)
    return Image.fromarray(output_img)


# Creating output folders
os.makedirs("blur", exist_ok=True)
os.makedirs("sharp", exist_ok=True)

# Loading pretrained model
teacher = Restormer()
ckpt = torch.load('./Restormer/motion_deblurring.pth')
teacher.load_state_dict(ckpt['params'])
teacher.eval().cuda()

# Loading GoPro dataset paths
gopro_blur_paths = sorted(glob("./GoPro/test/Blur/*.png"))
gopro_sharp_paths = sorted(glob("./GoPro/test/Sharp/*.png"))

# Processing images using teacher model and storing the output
for blur_path in gopro_blur_paths:
    fname = os.path.basename(blur_path)

    img = cv2.imread(blur_path)
    img = cv2.resize(img, (1920, 1080))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output_img = patch_inference(teacher, img_rgb)

    cv2.imwrite(f"blur/{fname}", img)
    output_img.save(f"sharp/{fname}")

torch.cuda.empty_cache()

# SSIM score calculation
ssim_scores = []
for sharp_path, blur_path in zip(gopro_sharp_paths, gopro_blur_paths):
    fname = os.path.basename(blur_path)

    gt_img = cv2.imread(sharp_path)
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

    sharp_img = np.array(Image.open(f"sharp/{fname}"))

    # Ensuring same shape
    if gt_img.shape != sharp_img.shape:
        sharp_img = cv2.resize(sharp_img, (gt_img.shape[1], gt_img.shape[0]))

    score = ssim(gt_img, sharp_img, channel_axis=2, data_range=255)
    ssim_scores.append(score)

    # For image visualisation
    """
    if True:
        img_blur = cv2.imread(blur_path)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB))
        plt.title("Blurred")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(sharp_img)
        plt.title("Teacher Output")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(gt_img)
        plt.title("Ground Truth")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
    """

avg_ssim = sum(ssim_scores) / len(ssim_scores)
print(f"Average SSIM between Ground Truth and Teacher output: {avg_ssim:.4f}")