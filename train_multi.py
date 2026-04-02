import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Paths
DATASET_CSV = "labels.csv"   # make sure this CSV exists
IMAGE_DIR = "part1\dataset"          # folder containing images

# Load dataset
df = pd.read_csv(DATASET_CSV)

# Encode categorical labels
le_gender = LabelEncoder()
df["gender"] = le_gender.fit_transform(df["gender"])  # "Man" -> 1, "Woman" -> 0, etc.

le_skin = LabelEncoder()
df["skin_tone"] = le_skin.fit_transform(df["skin_tone"])  # e.g., "Fair", "Medium", "Dark"

# Lists for data
images = []
ages = []
genders = []
skin_tones = []

# Load images
for i, row in df.iterrows():
    img_path = os.path.join(IMAGE_DIR, row["filename"])
    if os.path.exists(img_path):
        try:
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            ages.append(row["age"])
            genders.append(row["gender"])
            skin_tones.append(row["skin_tone"])
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    else:
        print(f"Missing: {img_path}")

# Convert to numpy arrays
images = np.array(images)
ages = np.array(ages)
genders = to_categorical(np.array(genders), num_classes=len(le_gender.classes_))
skin_tones = to_categorical(np.array(skin_tones), num_classes=len(le_skin.classes_))

# Model
input_layer = Input(shape=(128, 128, 3))

x = Conv2D(32, (3, 3), activation="relu")(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation="relu")(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)

# Outputs
age_output = Dense(1, activation="linear", name="age_output")(x)
gender_output = Dense(len(le_gender.classes_), activation="softmax", name="gender_output")(x)
skin_output = Dense(len(le_skin.classes_), activation="softmax", name="skin_output")(x)

model = Model(inputs=input_layer, outputs=[age_output, gender_output, skin_output])

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        "age_output": "mse",
        "gender_output": "categorical_crossentropy",
        "skin_output": "categorical_crossentropy",
    },
    metrics={
        "age_output": "mae",
        "gender_output": "accuracy",
        "skin_output": "accuracy",
    },
)

# Train
model.fit(
    images,
    {"age_output": ages, "gender_output": genders, "skin_output": skin_tones},
    validation_split=0.2,
    epochs=20,
    batch_size=32,
)

# Save model & encoders
model.save("multi_task_model.h5")
np.save("gender_classes.npy", le_gender.classes_)
np.save("skin_classes.npy", le_skin.classes_)

print("✅ Training complete. Model & label encoders saved.")
