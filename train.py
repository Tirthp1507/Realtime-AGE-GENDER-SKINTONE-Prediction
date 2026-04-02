import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import pickle

# Get the folder where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset path
DATASET_PATH = os.path.join(BASE_DIR, "part1\dataset")

# Image size for training
IMG_SIZE = 128

print("[INFO] Loading dataset...")

images = []
labels = []

# Loop through files in dataset folder
for file in os.listdir(DATASET_PATH):
    file_path = os.path.join(DATASET_PATH, file)

    # Only process images
    if not (file.lower().endswith(".jpg") or file.lower().endswith(".png")):
        continue

    try:
        # Extract age from filename (first number before "_")
        age = int(file.split("_")[0])
    except:
        print(f"[WARNING] Skipping {file}, cannot extract age")
        continue

    img = cv2.imread(file_path)
    if img is None:
        print(f"[WARNING] Skipping invalid image: {file_path}")
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    images.append(img)
    labels.append(age)

# Convert to numpy arrays
images = np.array(images, dtype="float32") / 255.0
labels = np.array(labels)

print(f"[INFO] Dataset loaded: {len(images)} images.")

# ---------------------------
# Convert to regression task
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Build CNN model (regression for age)
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="linear")   # regression output
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

print("[INFO] Training model...")
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=15, batch_size=32)

# Save model
model_path = os.path.join(BASE_DIR, "age_model.h5")
model.save(model_path)

print(f"[INFO] Model training complete and saved as:\n - {model_path}")
