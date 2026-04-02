import numpy as np
import pandas as pd
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Paths
data_dir = "part1\dataset"   # update with your images folder
csv_file = "labels.csv"  # CSV with filename,skin_tone,...

df = pd.read_csv(csv_file)

X = []
y = []

for _, row in df.iterrows():
    img_path = os.path.join(data_dir, row["filename"])
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (128, 128)) / 255.0
        X.append(img)
        y.append(row["skin_tone"])  # skin_tone column

X = np.array(X)
y = np.array(y)

# Encode skin_tone classes
classes = np.unique(y)
np.save("skin_classes.npy", classes)

y = np.array([np.where(classes == label)[0][0] for label in y])
y = to_categorical(y, num_classes=len(classes))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(classes), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

model.save("skin_model.h5")
print("✅ Skin model saved as skin_model.h5")
