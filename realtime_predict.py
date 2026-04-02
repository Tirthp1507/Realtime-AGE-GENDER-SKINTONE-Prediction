import cv2
import numpy as np
import tensorflow as tf

# Load trained models
age_model = tf.keras.models.load_model("age_model.h5")
gender_model = tf.keras.models.load_model("gender_model.h5")
skin_model = tf.keras.models.load_model("skin_model.h5")

# Labels
gender_labels = ["Male", "Female"]
skin_labels = np.load("skin_classes.npy")  # Example: ["Fair", "Medium", "Dark"]

# Preprocessing function
def preprocess_face(face_img):
    face_resized = cv2.resize(face_img, (128, 128))  # match model input
    face_normalized = face_resized / 255.0
    face_expanded = np.expand_dims(face_normalized, axis=0)
    return face_expanded

# OpenCV setup
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Dictionary to store final predictions
locked_predictions = {}
face_id_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_center = (x + w // 2, y + h // 2)
        matched_id = None

        # Match existing faces (within 50px distance)
        for fid, data in locked_predictions.items():
            (px, py) = data["center"]
            if abs(face_center[0] - px) < 50 and abs(face_center[1] - py) < 50:
                matched_id = fid
                break

        if matched_id is None:
            face_id_counter += 1
            matched_id = face_id_counter

            # Run predictions only ONCE for this new face
            face = frame[y:y+h, x:x+w]
            processed_face = preprocess_face(face)

            age_pred = age_model.predict(processed_face, verbose=0)
            gender_pred = gender_model.predict(processed_face, verbose=0)
            skin_pred = skin_model.predict(processed_face, verbose=0)

            age = int(age_pred[0][0])  # regression output
            gender = gender_labels[np.argmax(gender_pred)]
            skin = skin_labels[np.argmax(skin_pred)]

            # Save locked predictions
            locked_predictions[matched_id] = {
                "center": face_center,
                "age": age,
                "gender": gender,
                "skin": skin
            }
        else:
            # Just update center (no re-prediction!)
            locked_predictions[matched_id]["center"] = face_center

        # Draw results (always from saved values, never recomputed)
        pred = locked_predictions[matched_id]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Age: {pred['age']}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Gender: {pred['gender']}", (x, y+h+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Skin: {pred['skin']}", (x, y+h+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Fixed Age, Gender & Skin Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
