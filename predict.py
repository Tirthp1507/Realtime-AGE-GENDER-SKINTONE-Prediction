import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load all models
age_model = load_model("age_model.h5")
gender_model = load_model("gender_model.h5")
skin_model = load_model("skin_model.h5")

# Load classes
gender_classes = np.load("gender_classes.npy", allow_pickle=True)
skin_classes = np.load("skin_classes.npy", allow_pickle=True)

# Preprocessing function
def preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_all(img_path):
    # Preprocess image
    img_array = preprocess_image(img_path)

    # Predictions
    age_pred = age_model.predict(img_array)
    gender_pred = gender_model.predict(img_array)
    skin_pred = skin_model.predict(img_array)

    # Decode predictions
    age = int(age_pred[0][0])   # Regression output
    gender = gender_classes[np.argmax(gender_pred)]
    skin = skin_classes[np.argmax(skin_pred)]

    return {
        "Age": age,
        "Gender": gender,
        "Skin Tone": skin
    }

if __name__ == "__main__":
    test_image = "test.jpg"   # Change with your image path
    result = predict_all(test_image)
    print("Prediction Results:")
    print(f"Age: {result['Age']}")
    print(f"Gender: {result['Gender']}")
    print(f"Skin Tone: {result['Skin Tone']}")
