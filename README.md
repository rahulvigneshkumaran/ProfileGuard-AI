# 🕵️ ProfileGuard AI

ProfileGuard AI is a **multimodal fake social media profile detection system** built using Deep Learning.

---

## 🚀 Features

- 🧠 Profile image analysis using CNN
- ✍️ Bio text analysis using LSTM
- 📊 Account activity analysis (followers, engagement, etc.)
- 🎯 Fake / Real prediction with confidence score
- 🔍 Explainable AI (reasons for prediction)
- 🎨 Premium Streamlit UI (glassmorphism design)

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pandas
- OpenCV
- Scikit-learn
- Pillow

---

## 📥 Inputs

- Profile Image
- Bio Text
- Followers
- Following
- Posts
- Account Age
- Engagement Rate

---

## 📤 Output

- Fake / Real Prediction
- Confidence Score
- Fake Probability
- Reason Explanation

---

## 🧠 How It Works

This project uses a **multimodal deep learning approach**:

- CNN → analyzes profile images  
- LSTM → processes bio text  
- Dense layers → analyze activity patterns  

All features are combined to make a final prediction.

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python train.py
streamlit run streamlit_app.py
