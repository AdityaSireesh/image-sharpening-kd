import tensorflow as tf
from tensorflow.image import stateless_random_flip_left_right, stateless_random_flip_up_down, stateless_random_brightness
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
import os, cv2
import numpy as np
from PIL import Image
from glob import glob

# Different parameters
# For easy modification of training program
IMG_SIZE = 192
PATCHES = 10
BATCH_SIZE = 2
EPOCHS = 20


# Dynamic memory allocation for efficient GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Function for calculating combined loss
def combined_loss(y_true, y_pred):
    # Mean Absolute Error
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))

    # SSIM loss
    ssim_loss = tf.reduce_mean(1 - tf.image.ssim(y_true, y_pred, max_val=1.0))

    return 0.8 * l1 + 0.2 * ssim_loss


# Generator to yield cropped patches for training dataset
def patch_generator():
    for blur_path, sharp_path in zip(blur_paths, sharp_paths):
        blur_img = cv2.imread(blur_path)
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
        sharp_img = np.array(Image.open(sharp_path).convert("RGB"))

        # Ensuring same shape
        if blur_img.shape != sharp_img.shape:
            sharp_img = cv2.resize(sharp_img, (blur_img.shape[1], blur_img.shape[0]))

        h, w = blur_img.shape[:2]
        for i in range(PATCHES):
            y = np.random.randint(0, h - IMG_SIZE)
            x = np.random.randint(0, w - IMG_SIZE)
            blur_patch = blur_img[y:y + IMG_SIZE, x:x + IMG_SIZE, :]
            sharp_patch = sharp_img[y:y + IMG_SIZE, x:x + IMG_SIZE, :]

            blur_patch = tf.convert_to_tensor(blur_patch, dtype=tf.float32) / 255.0
            sharp_patch = tf.convert_to_tensor(sharp_patch, dtype=tf.float32) / 255.0

            # Randomised modifications for the training data
            seed = np.random.randint(0, 10000)

            blur_patch = stateless_random_flip_left_right(blur_patch, seed=[seed, 0])
            sharp_patch = stateless_random_flip_left_right(sharp_patch, seed=[seed, 0])

            blur_patch = stateless_random_flip_up_down(blur_patch, seed=[seed, 0])
            sharp_patch = stateless_random_flip_up_down(sharp_patch, seed=[seed, 0])

            blur_patch = stateless_random_brightness(blur_patch, 0.2, seed=[seed, 0])
            sharp_patch = stateless_random_brightness(sharp_patch, 0.2, seed=[seed, 0])

            yield blur_patch, sharp_patch

# Student model creation
def build_student_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Encoder layers
    x = layers.SeparableConv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.SeparableConv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)

    # Bottleneck
    x = layers.SeparableConv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.SeparableConv2D(128, 3, activation='relu', padding='same')(x)

    # Decoder layers
    x = layers.UpSampling2D()(x)
    x = layers.SeparableConv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.SeparableConv2D(32, 3, activation='relu', padding='same')(x)

    outputs = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)
    return tf.keras.models.Model(inputs, outputs)


# Checking for existing student model and creating new one if not found
if os.path.exists("best_student_model.keras"):
    student = load_model("best_student_model.keras", custom_objects={'combined_loss': combined_loss})
else:
    student = build_student_model()
    student.compile(optimizer='adam', loss=combined_loss, metrics=['mae'])

# Loading image paths for training the model
blur_paths = sorted(glob("blur/*.png"))
sharp_paths = sorted(glob("sharp/*.png"))

# Dataset creation
ds = tf.data.Dataset.from_generator(
    patch_generator,
    output_signature=(
        tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
    )
).repeat().shuffle(500).batch(2).prefetch(tf.data.AUTOTUNE)

# Splitting into training and validation datasets
total = len(blur_paths) * PATCHES
val_count = int(total * 0.2)
train_ds = ds.skip(val_count)
val_ds = ds.take(val_count)
steps_per_epoch = int((len(blur_paths) * PATCHES * 0.8) // BATCH_SIZE)
validation_steps = int((len(blur_paths) * PATCHES * 0.2) // BATCH_SIZE)

checkpoint = ModelCheckpoint(
    filepath='best_student_model.keras',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)

# Training the student model
history = student.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    shuffle=True,
    callbacks=[
        checkpoint,
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1)
    ]
)

student.save("student_model.keras")