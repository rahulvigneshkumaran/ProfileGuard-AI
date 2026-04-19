import os
import pickle
import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    Embedding,
    LSTM,
    concatenate
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------
# CONFIG
# ----------------------------
IMG_SIZE = 128
MAX_WORDS = 5000
MAX_LEN = 30
MODEL_DIR = "models"
DATASET_CSV = "dataset/profiles.csv"

os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------
# LOAD DATASET
# ----------------------------
df = pd.read_csv(DATASET_CSV)

# Fill missing values safely
df["bio_text"] = df["bio_text"].fillna("").astype(str)

required_columns = [
    "image_path",
    "bio_text",
    "followers",
    "following",
    "posts",
    "account_age",
    "engagement_rate",
    "label"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing column in CSV: {col}")

# ----------------------------
# IMAGE PREPROCESSING
# ----------------------------
def load_image(img_path):
    full_path = os.path.join("dataset", img_path) if not img_path.startswith("dataset") else img_path
    img = cv2.imread(full_path)

    if img is None:
        print(f"Warning: Could not load image: {full_path}. Using blank image.")
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = img.astype("float32") / 255.0
    return img

image_data = np.array([load_image(path) for path in df["image_path"]])

# ----------------------------
# TEXT PREPROCESSING
# ----------------------------
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df["bio_text"])

text_sequences = tokenizer.texts_to_sequences(df["bio_text"])
text_data = pad_sequences(text_sequences, maxlen=MAX_LEN, padding="post", truncating="post")

# ----------------------------
# NUMERIC PREPROCESSING
# ----------------------------
numeric_columns = ["followers", "following", "posts", "account_age", "engagement_rate"]
numeric_data = df[numeric_columns].astype("float32").values

scaler = StandardScaler()
numeric_data = scaler.fit_transform(numeric_data)

# ----------------------------
# LABELS
# ----------------------------
labels = df["label"].astype("float32").values

# ----------------------------
# TRAIN / TEST SPLIT
# ----------------------------
(
    X_img_train, X_img_test,
    X_text_train, X_text_test,
    X_num_train, X_num_test,
    y_train, y_test
) = train_test_split(
    image_data,
    text_data,
    numeric_data,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels if len(np.unique(labels)) > 1 else None
)

# ----------------------------
# IMAGE BRANCH
# ----------------------------
image_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image_input")

x1 = Conv2D(32, (3, 3), activation="relu")(image_input)
x1 = MaxPooling2D((2, 2))(x1)

x1 = Conv2D(64, (3, 3), activation="relu")(x1)
x1 = MaxPooling2D((2, 2))(x1)

x1 = Flatten()(x1)
x1 = Dense(64, activation="relu")(x1)
x1 = Dropout(0.3)(x1)

# ----------------------------
# TEXT BRANCH
# ----------------------------
text_input = Input(shape=(MAX_LEN,), name="text_input")

x2 = Embedding(input_dim=MAX_WORDS, output_dim=64)(text_input)
x2 = LSTM(64)(x2)
x2 = Dense(32, activation="relu")(x2)
x2 = Dropout(0.3)(x2)

# ----------------------------
# NUMERIC BRANCH
# ----------------------------
num_input = Input(shape=(len(numeric_columns),), name="num_input")

x3 = Dense(32, activation="relu")(num_input)
x3 = Dropout(0.3)(x3)
x3 = Dense(16, activation="relu")(x3)

# ----------------------------
# COMBINE
# ----------------------------
combined = concatenate([x1, x2, x3])

z = Dense(64, activation="relu")(combined)
z = Dropout(0.3)(z)
z = Dense(32, activation="relu")(z)
output = Dense(1, activation="sigmoid")(z)

model = Model(inputs=[image_input, text_input, num_input], outputs=output)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------
# TRAIN
# ----------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    [X_img_train, X_text_train, X_num_train],
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# ----------------------------
# EVALUATE
# ----------------------------
loss, accuracy = model.evaluate([X_img_test, X_text_test, X_num_test], y_test, verbose=1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# ----------------------------
# SAVE MODEL + TOKENIZER + SCALER
# ----------------------------
model.save(os.path.join(MODEL_DIR, "fake_profile_model.keras"))

with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print("Training complete. Model files saved in 'models/' folder.")