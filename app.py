# ============================================================
# APP.PY - HYBRID SYSTEM
# IOU + SSIM + CNN EMBEDDING (Dense 256)
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64

from skimage.metrics import structural_similarity as ssim
from skimage.morphology import skeletonize
from tensorflow.keras import layers, models
from keras.models import load_model, Model
from numpy.linalg import norm

# ============================================================
# INIT APP
# ============================================================

app = Flask(__name__)
CORS(app)

# ============================================================
# LOAD CNN MODEL (Dense 256 as Embedding)
# ============================================================

# CNN_MODEL_PATH = "model/cnn_embedding_aksara_sasak.keras"
# cnn_model = load_model(CNN_MODEL_PATH)

# # Layer sebelum softmax = Dense(256)
# embedding_model = Model(
#     inputs=cnn_model.input,
#     outputs=cnn_model.layers[-2].output
# )

# print("✅ CNN Embedding Model Loaded")

# Di app.py

def build_model_local(num_classes=18):
    input_layer = layers.Input(shape=(128, 128, 1))
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_layer)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    embedding = layers.Dense(256, activation='relu', name="embedding_layer")(x)
    x = layers.Dropout(0.5)(embedding)
    output = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=input_layer, outputs=output)

# Bangun dan load bobot
cnn_model = build_model_local(num_classes=18) # Sesuaikan jumlah kelas
cnn_model.load_weights("model/cnn_embedding_aksara_sasak.h5")

# Ambil model embeddingnya
embedding_model = models.Model(
    inputs=cnn_model.input,
    outputs=cnn_model.get_layer("embedding_layer").output
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def apply_thinning(img):
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    binary_bool = binary > 127
    skeleton = skeletonize(binary_bool)
    skeleton_img = (skeleton * 255).astype(np.uint8)
    return skeleton_img


def dilasi(img, ukuran_kernel):
    kernel = np.ones((ukuran_kernel, ukuran_kernel), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


def calculate_iou(img1, img2):
    mask1 = img1 > 127
    mask2 = img2 > 127

    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    sum_union = np.sum(union)
    if sum_union == 0:
        return 0.0

    return (np.sum(intersection) / sum_union) * 100

# ============================================================
# PREPROCESS FOR RULE-BASED (256x256)
# ============================================================


def preprocess_rule_based(img, use_thinning=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    cropped = thresh[y:y+h, x:x+w]

    resized = cv2.resize(cropped, (256, 256))

    if use_thinning:
        resized = apply_thinning(resized)

    return resized

# ============================================================
# PREPROCESS FOR CNN (128x128 - SAME AS TRAINING)
# ============================================================


def preprocess_for_cnn(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    cropped = thresh[y:y+h, x:x+w]

    resized = cv2.resize(cropped, (128, 128))

    # thinning + dilasi 3x3 (SAMA DENGAN TRAINING)
    thinned = apply_thinning(resized)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thinned, kernel, iterations=1)

    normalized = dilated.astype("float32") / 255.0
    normalized = normalized.reshape(1, 128, 128, 1)

    return normalized

# ============================================================
# ROUTE 1 - IOU + SSIM
# ============================================================

@app.route("/cek-akurasi", methods=["POST"])
def cek_akurasi():

    data = request.json
    image_data = data.get("image")
    huruf = data.get("huruf")

    if not image_data or not huruf:
        return jsonify({"error": "Data tidak lengkap"}), 400

    image_data = image_data.split(",")[1]
    img_bytes = np.frombuffer(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    processed_user = preprocess_rule_based(img, use_thinning=True)
    if processed_user is None:
        return jsonify({"score": 0})

    template_path = f"assets/perbandingan/indikator2{huruf}.png"
    if not os.path.exists(template_path):
        return jsonify({"error": "Template tidak ditemukan"}), 404

    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.resize(template, (256, 256))
    template = apply_thinning(template)

    processed_user = dilasi(processed_user, 7)
    template = dilasi(template, 7)

    score_ssim, _ = ssim(processed_user, template, full=True)
    score_ssim *= 100

    score_iou = calculate_iou(processed_user, template)
    score_iou *= 3

    final_score = (0.3 * score_ssim) + (0.7 * score_iou)

    return jsonify({
        "score": round(final_score, 2),
        "method": "rule_based"
    })

# ============================================================
# ROUTE 2 - CNN EMBEDDING + COSINE SIMILARITY
# ============================================================

@app.route("/cek-akurasi-cnn", methods=["POST"])
def cek_akurasi_cnn():

    data = request.json
    image_data = data.get("image")
    huruf = data.get("huruf")

    if not image_data or not huruf:
        return jsonify({"error": "Data tidak lengkap"}), 400

    image_data = image_data.split(",")[1]
    img_bytes = np.frombuffer(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    user_processed = preprocess_for_cnn(img)
    if user_processed is None:
        return jsonify({"score": 0})

    user_embedding = embedding_model.predict(user_processed)[0]

    template_path = f"assets/perbandingan/indikator2{huruf}.png"
    if not os.path.exists(template_path):
        return jsonify({"error": "Template tidak ditemukan"}), 404

    template_img = cv2.imread(template_path)
    template_processed = preprocess_for_cnn(template_img)

    if template_processed is None:
        return jsonify({"score": 0})

    template_embedding = embedding_model.predict(template_processed)[0]

    similarity = cosine_similarity(user_embedding, template_embedding)

    score = max(0, min(similarity * 100, 100))

    return jsonify({
        "score": round(score, 2),
        "method": "cnn_embedding"
    })

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run()
