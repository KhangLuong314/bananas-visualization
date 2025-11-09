from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
from joblib import load as load_joblib
from PIL import Image
import numpy as np
import pandas as pd
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Load models once at startup
try:
    model0 = load_model("/model/banana_ripeness (1).h5")
    model1 = load_joblib("/model/bananas-visualization/model/banana_regression_uncertainty.joblib")
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    model0 = None
    model1 = None

def extract_features_bgr(img_bgr):
    """Extract color and texture features from BGR image"""
    h0, w0 = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(float)   # 0..179
    S = hsv[:, :, 1].astype(float)   # 0..255
    V = hsv[:, :, 2].astype(float)   # 0..255

    # Rough banana mask: moderate brightness & saturation
    mask = (V > 30) & (S > 20)

    # If mask too small, fallback to center crop mask
    if mask.sum() < 50:
        cy, cx = h0 // 2, w0 // 2
        r = min(h0, w0) // 3
        yy, xx = np.ogrid[:h0, :w0]
        mask = (xx - cx)**2 + (yy - cy)**2 <= r*r

    # Compute bounding box of mask and crop to focus on fruit
    ys, xs = np.where(mask)
    if len(xs) > 0 and len(ys) > 0:
        x1, x2 = max(0, np.min(xs)), min(w0 - 1, np.max(xs))
        y1, y2 = max(0, np.min(ys)), min(h0 - 1, np.max(ys))
        # Small padding
        pad = 4
        x1, x2 = max(0, x1 - pad), min(w0 - 1, x2 + pad)
        y1, y2 = max(0, y1 - pad), min(h0 - 1, y2 + pad)
        img_crop = img_bgr[y1:y2 + 1, x1:x2 + 1]
        hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
        H = hsv[:, :, 0].astype(float)
        S = hsv[:, :, 1].astype(float)
        V = hsv[:, :, 2].astype(float)
        mask = (V > 30) & (S > 20)
        if mask.sum() == 0:
            mask = np.ones_like(V, dtype=bool)

    # Compute statistics safely
    inds = mask
    if np.sum(inds) == 0:
        inds = np.ones_like(H, dtype=bool)

    H_mean = float(np.mean(H[inds]))
    S_mean = float(np.mean(S[inds]))
    V_mean = float(np.mean(V[inds]))
    texture_std = float(np.std(V[inds]))

    # Per-pixel color masks (OpenCV H ranges 0..179)
    brown_mask = (H >= 5) & (H <= 25) & (V < 200) & (S > 30)
    green_mask = (H >= 30) & (H <= 85) & (S > 40)
    yellow_mask = (H >= 20) & (H <= 40) & (S > 30)
    
    denom = max(1.0, np.sum(inds))
    brown_frac = float(np.sum(brown_mask & inds) / denom)
    green_frac = float(np.sum(green_mask & inds) / denom)
    yellow_frac = float(np.sum(yellow_mask & inds) / denom)

    return [H_mean, S_mean, V_mean, texture_std, brown_frac, green_frac, yellow_frac]

def interpret_prediction(pred_mean, pred_std):
    """Interpret the regression model prediction"""
    if pred_mean <= 6:
        status = "Unripe 🍏"
        message = "Your banana needs more time to ripen. Wait a few more days."
    elif pred_mean <= 10:
        status = "Almost ripe 🍌"
        message = "Almost there! Your banana will be perfect in 1-2 days."
    elif pred_mean <= 15:
        status = "Perfectly ripe 🍯"
        message = "Perfectly ripe — eat now for the best flavor!"
    elif pred_mean <= 18:
        status = "Slightly overripe 🍞"
        message = "Still good to eat, or perfect for banana bread!"
    else:
        status = "Overripe ♻️"
        message = "Very ripe — best used for smoothies or baking."
    
    return {
        "status": status,
        "message": message,
        "days_estimate": float(pred_mean),
        "uncertainty": float(pred_std)
    }

@app.route("/predict", methods=["POST"])
def predict():
    """Main prediction endpoint"""
    try:
        # Check if models are loaded
        if model0 is None or model1 is None:
            return jsonify({"error": "Models not loaded"}), 500

        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # Read image from file
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Step 1: Use classification model (model0) to get initial prediction
        img_resized = tf.image.resize(img, (256, 256))
        img_normalized = np.expand_dims(img_resized / 255.0, 0)
        yhat = model0.predict(img_normalized, verbose=0)
        
        # Get the class with highest probability
        class_idx = np.argmax(yhat[0])
        confidence = float(yhat[0][class_idx])
        
        # Map class index to category
        class_names = ["Overripe", "Ripe", "Unripe"]
        result_class = class_names[class_idx]
        
        # Step 2: If perfectly ripe, return immediately
        if result_class == "Ripe":
            return jsonify({
                "classification": result_class,
                "confidence": confidence,
                "status": "Perfectly ripe 🍯",
                "message": "Perfectly ripe — eat now for the best flavor!",
                "model_used": "classification_only"
            })
        
        # Step 3: For Overripe or Unripe, use regression model for detailed analysis
        img_resized_300 = cv2.resize(img, (300, 300))
        feats = extract_features_bgr(img_resized_300)
        X_new = pd.DataFrame([feats], columns=[
            "H_mean", "S_mean", "V_mean", "texture_std",
            "brown_frac", "green_frac", "yellow_frac"
        ])

        # Predict across trees for uncertainty
        all_tree_preds = np.stack([t.predict(X_new.values) for t in model1.estimators_], axis=1)
        pred_mean = np.mean(all_tree_preds, axis=1)[0]
        pred_std = np.std(all_tree_preds, axis=1)[0]

        # Interpret regression result
        interpretation = interpret_prediction(pred_mean, pred_std)
        
        return jsonify({
            "classification": result_class,
            "confidence": confidence,
            "status": interpretation["status"],
            "message": interpretation["message"],
            "days_estimate": interpretation["days_estimate"],
            "uncertainty": interpretation["uncertainty"],
            "model_used": "classification_and_regression"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": model0 is not None and model1 is not None
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)