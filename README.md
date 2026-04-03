# 🚀 Real-Time Age, Gender & Skin Tone Prediction

An AI-powered Computer Vision project that predicts **Age, Gender, and Skin Tone** in real-time using **Deep Learning and OpenCV**, with an interactive **Streamlit web interface**.

---

## 🔥 Features

- 🎯 Real-time face detection using OpenCV
- 🧠 Age prediction (CNN Regression Model)
- 👤 Gender classification with confidence scores
- 🎨 Skin tone classification with confidence scores
- ⚡ Smart face tracking (prevents repeated predictions)
- 🌐 Streamlit-based interactive UI

---

## 🛠 Tech Stack

- Python  
- OpenCV  
- TensorFlow / Keras  
- NumPy  
- Streamlit  

---

## 📂 Project Structure
├── app.py # Streamlit App
├── train.py # Age model training
├── train_gender.py # Gender model
├── train_skin.py # Skin tone model
├── realtime_predict.py # OpenCV real-time version
├── predict.py # Image prediction
├── model.py # Model architecture
├── labels.csv # Dataset labels
├── test.jpg # Sample image
├── .gitignore
└── README.md


---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
