import os
import pickle
import numpy as np
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

IMG_SIZE = 128
MAX_LEN = 30
MODEL_DIR = "models"

MODEL_PATH = os.path.join(MODEL_DIR, "fake_profile_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Load saved files once
model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

def preprocess_image_from_array(image_array):
    """
    image_array should be a NumPy array in RGB format
    """
    img = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([str(text)])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    return padded

def preprocess_numeric(followers, following, posts, account_age, engagement_rate):
    arr = np.array([[followers, following, posts, account_age, engagement_rate]], dtype="float32")
    arr = scaler.transform(arr)
    return arr

def generate_reason(bio_text, followers, following, posts, account_age, engagement_rate, prediction_score):
    reasons = []

    bio_lower = str(bio_text).lower()

    spam_words = ["earn", "money", "click", "dm", "offer", "free", "promo", "promotion", "cash"]
    if any(word in bio_lower for word in spam_words):
        reasons.append("Suspicious words found in bio")

    if following > 0 and followers / following < 0.1:
        reasons.append("Very low follower-to-following ratio")

    if posts == 0:
        reasons.append("No posts on profile")

    if account_age < 30:
        reasons.append("Very new account")

    if engagement_rate < 0.01:
        reasons.append("Low engagement rate")

    if prediction_score >= 0.5 and not reasons:
        reasons.append("Combined profile patterns look suspicious")

    if prediction_score < 0.5 and not reasons:
        reasons.append("Profile patterns appear more natural")

    return reasons

def predict_profile(image_array, bio_text, followers, following, posts, account_age, engagement_rate):
    img = preprocess_image_from_array(image_array)
    text = preprocess_text(bio_text)
    num = preprocess_numeric(followers, following, posts, account_age, engagement_rate)

    prediction = model.predict([img, text, num], verbose=0)[0][0]

    label = "Fake" if prediction >= 0.5 else "Real"
    confidence = float(prediction if prediction >= 0.5 else 1 - prediction) * 100

    reasons = generate_reason(
        bio_text=bio_text,
        followers=followers,
        following=following,
        posts=posts,
        account_age=account_age,
        engagement_rate=engagement_rate,
        prediction_score=prediction
    )

    return label, confidence, float(prediction), reasons