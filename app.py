import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

st.set_page_config(
    page_title="Real-Time Age, Gender & Skin Prediction",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}
h1, h2, h3, h4, h5, h6, p, label, div, span {
    color: white !important;
}
.stButton>button {
    background-color: #1f77ff;
    color: white;
    border-radius: 8px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    age_model = tf.keras.models.load_model("age_model.h5")
    gender_model = tf.keras.models.load_model("gender_model.h5")
    skin_model = tf.keras.models.load_model("skin_model.h5")
    return age_model, gender_model, skin_model

@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

age_model, gender_model, skin_model = load_models()
face_cascade = load_face_cascade()

gender_labels = ["Male", "Female"]
skin_labels = np.load("skin_classes.npy", allow_pickle=True)

def preprocess_face(face_img):
    face_resized = cv2.resize(face_img, (128, 128))
    face_normalized = face_resized.astype("float32") / 255.0
    face_expanded = np.expand_dims(face_normalized, axis=0)
    return face_expanded

if "locked_predictions" not in st.session_state:
    st.session_state.locked_predictions = {}
if "face_id_counter" not in st.session_state:
    st.session_state.face_id_counter = 0

st.title("🧠 Real-Time Age, Gender & Skin Prediction")
st.write("📷 Click below to start camera")

img_file = st.camera_input("Take a picture")

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    locked_predictions = st.session_state.locked_predictions

    for (x, y, w, h) in faces:
        face_center = (x + w // 2, y + h // 2)
        matched_id = None

        for fid, data in locked_predictions.items():
            px, py = data["center"]
            if abs(face_center[0] - px) < 50 and abs(face_center[1] - py) < 50:
                matched_id = fid
                break

        if matched_id is None:
            st.session_state.face_id_counter += 1
            matched_id = st.session_state.face_id_counter

            face = frame[y:y+h, x:x+w]

            if face.size == 0:
                continue

            face = cv2.GaussianBlur(face, (5, 5), 0)
            processed_face = preprocess_face(face)

            age_pred = age_model.predict(processed_face, verbose=0)
            gender_pred = gender_model.predict(processed_face, verbose=0)
            skin_pred = skin_model.predict(processed_face, verbose=0)

            age = max(0, int(age_pred[0][0]))

            gender_idx = np.argmax(gender_pred)
            skin_idx = np.argmax(skin_pred)

            gender_name = gender_labels[gender_idx]
            skin_name = str(skin_labels[skin_idx]).title()

            gender_conf = gender_pred[0][gender_idx] * 100
            skin_conf = skin_pred[0][skin_idx] * 100

            locked_predictions[matched_id] = {
                "center": face_center,
                "age": age,
                "gender": f"{gender_name} ({gender_conf:.1f}%)",
                "skin": f"{skin_name} ({skin_conf:.1f}%)"
            }
        else:
            locked_predictions[matched_id]["center"] = face_center

        pred = locked_predictions[matched_id]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        age_y = max(y - 10, 20)
        cv2.putText(
            frame,
            f"Age: {pred['age']}",
            (x, age_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Gender: {pred['gender']}",
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Skin Tone: {pred['skin']}",
            (x, y + h + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

    result_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(result_rgb, caption="Prediction", use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Reset Faces"):
        st.session_state.locked_predictions = {}
        st.session_state.face_id_counter = 0
        st.success("Face predictions reset successfully.")

with col2:
    st.write(f"Tracked Faces: {len(st.session_state.locked_predictions)}")