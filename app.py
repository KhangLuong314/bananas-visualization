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
import os
import base64
import random
from huggingface_hub import hf_hub_download

# Get port from environment variable
PORT = int(os.environ.get('PORT', 5001))

app = Flask(__name__)
CORS(app)

# Load models from Hugging Face
try:
    print("Downloading models from Hugging Face...")
    
    # YOUR HUGGING FACE REPO - REPLACE WITH YOUR USERNAME
    HF_REPO = "khangluong314/banana-ripeness-models"
    
    # Download classification model
    print("Downloading classification model...")
    classification_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="banana_ripeness.h5",
        cache_dir="./model_cache"
    )
    
    # Download regression model
    print("Downloading regression model...")
    regression_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="banana_regression_uncertainty.joblib",
        cache_dir="./model_cache"
    )
    
    # Load models
    print("Loading models into memory...")
    model0 = load_model(classification_path)
    model1 = load_joblib(regression_path)
    
    print("Models loaded successfully!")
    print(f"\nClassification Model Summary:")
    print(f"  - Input shape: {model0.input_shape}")
    print(f"  - Output shape: {model0.output_shape}")
    print(f"  - Total parameters: {model0.count_params():,}")
    print(f"\nRegression Model Summary:")
    print(f"  - Model type: {type(model1).__name__}")
    print(f"  - Number of estimators: {model1.n_estimators}")
    print(f"  - Number of features: {model1.n_features_in_}")
    
except Exception as e:
    print(f"Error loading models: {e}")
    import traceback
    traceback.print_exc()
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

def calculate_color_percentages(img_bgr):
    """Calculate color percentages for frontend display"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    
    # Create masks for banana region
    banana_mask = (V > 30) & (S > 20)
    total_pixels = max(1, np.sum(banana_mask))
    
    # Color classification
    yellow_mask = (H >= 15) & (H <= 35) & (S > 30) & (V > 80) & banana_mask
    brown_mask = (H >= 5) & (H <= 25) & (V < 150) & (S > 20) & banana_mask
    black_mask = (V < 80) & (S > 10) & banana_mask
    green_mask = (H >= 35) & (H <= 85) & (S > 40) & (V > 50) & banana_mask  # Green bananas
    
    yellow_pct = (np.sum(yellow_mask) / total_pixels) * 100
    brown_pct = (np.sum(brown_mask) / total_pixels) * 100
    black_pct = (np.sum(black_mask) / total_pixels) * 100
    green_pct = (np.sum(green_mask) / total_pixels) * 100
    
    # Normalize to 100%
    total = yellow_pct + brown_pct + black_pct + green_pct
    if total > 0:
        yellow_pct = (yellow_pct / total) * 100
        brown_pct = (brown_pct / total) * 100
        black_pct = (black_pct / total) * 100
        green_pct = (green_pct / total) * 100
    
    return {
        "yellow": round(yellow_pct, 1),
        "brown": round(brown_pct, 1),
        "black": round(black_pct, 1),
        "green": round(green_pct, 1)
    }

def create_visualization(img_bgr, percentages):
    """Create a visualization of the analyzed banana"""
    # Create a copy for drawing
    vis_img = img_bgr.copy()
    
    # Draw text overlay
    h, w = vis_img.shape[:2]
    
    # Add semi-transparent overlay at bottom
    overlay = vis_img.copy()
    cv2.rectangle(overlay, (0, h-100), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, vis_img, 0.5, 0, vis_img)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_pos = h - 70
    cv2.putText(vis_img, f"Yellow: {percentages['yellow']}%", (10, y_pos), 
                font, 0.6, (255, 255, 255), 2)
    cv2.putText(vis_img, f"Brown: {percentages['brown']}%", (10, y_pos + 25), 
                font, 0.6, (255, 255, 255), 2)
    
    # Convert to base64
    _, buffer = cv2.imencode('.png', vis_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return img_base64

def generate_banana_comment(percentages, classification, days_estimate):
    """Generate a fun banana comment based on analysis"""
    yellow = percentages['yellow']
    brown = percentages['brown']
    black = percentages['black']
    green = percentages['green']
    
    # Fun comments based on ripeness
    if classification == "Unripe" or green > 40:
        comments = [
            f"This banana is still getting its glow-up! With {green:.0f}% green and {yellow:.0f}% yellow, it's playing hard to get. Give it a few days to reach peak deliciousness! 🌱",
            f"Patience, young banana padawan! At {green:.0f}% green, it needs more time to achieve banana perfection. ⏰",
            f"This green machine isn't ready for prime time yet! Let it mature for a couple more days. Your future self will thank you! 💚"
        ]
    elif classification == "Ripe":
        comments = [
            f"*Chef's kiss* 👨‍🍳 This banana is absolutely PERFECT! {yellow:.0f}% yellow, {brown:.0f}% brown - it's the golden ratio of banana excellence!",
            f"WOW! This is textbook banana perfection! The {yellow:.0f}% yellow and {brown:.0f}% brown balance is *exactly* what we look for. EAT THIS NOW! 🎯",
            f"BANANA JACKPOT! 🎰 This beauty is in its prime with gorgeous {yellow:.0f}% yellow coloring. Don't wait - this is the sweet spot! ✨"
        ]
    elif classification == "Overripe":
        if black > 30:
            comments = [
                f"Uh oh! With {black:.0f}% black spots, this banana has seen better days. But hey - PERFECT for banana bread! 🍞",
                f"This banana is living its best life... in the past tense. At {black:.0f}% black, it's smoothie material now! 🥤",
                f"Time traveler alert! ⏰ This banana has {black:.0f}% black spots. Quick - make it into something delicious before it's too late!"
            ]
        else:
            comments = [
                f"Still totally edible! With {brown:.0f}% brown, it's super sweet and perfect for baking. Banana bread time? 🍌➡️🍞",
                f"This banana is on the ripe side with {brown:.0f}% brown spots, but that just means extra sweetness! Use it today! 💛",
                f"Getting spotty with {brown:.0f}% brown, but don't you dare throw it away! Perfect for smoothies or baking! ♻️"
            ]
    else:
        comments = [
            f"Interesting specimen! {yellow:.0f}% yellow, {brown:.0f}% brown, {green:.0f}% green. Every banana tells a story! 📖🍌",
            f"This banana is on a journey! Currently at {yellow:.0f}% yellow and {green:.0f}% green. Monitor its progress! 🔬"
        ]
    
    return random.choice(comments)

def interpret_prediction(pred_mean, pred_std):
    """Interpret the regression model prediction"""
    if pred_mean <= 6:
        status = "Unripe 🟢"
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

def generate_banana_comment(percentages, classification, days_estimate):
    """Generate a fun banana comment based on analysis"""
    yellow = percentages['yellow']
    brown = percentages['brown']
    black = percentages['black']
    
    # Fun comments based on ripeness
    if classification == "Unripe":
        comments = [
            f"This banana is still getting its glow-up! With {yellow:.0f}% yellow, it's playing hard to get. Give it a few days to reach peak deliciousness! 🌱",
            f"Patience, young banana padawan! At {yellow:.0f}% yellow and {brown:.0f}% brown, it needs more time to achieve banana perfection. ⏰",
            f"This green machine isn't ready for prime time yet! Let it mature for a couple more days. Your future self will thank you! 💚"
        ]
    elif classification == "Ripe":
        comments = [
            f"*Chef's kiss* 👨‍🍳 This banana is absolutely PERFECT! {yellow:.0f}% yellow, {brown:.0f}% brown - it's the golden ratio of banana excellence!",
            f"WOW! This is textbook banana perfection! The {yellow:.0f}% yellow and {brown:.0f}% brown balance is *exactly* what we look for. EAT THIS NOW! 🎯",
            f"BANANA JACKPOT! 🎰 This beauty is in its prime with gorgeous {yellow:.0f}% yellow coloring. Don't wait - this is the sweet spot! ✨"
        ]
    elif classification == "Overripe":
        if black > 30:
            comments = [
                f"Uh oh! With {black:.0f}% black spots, this banana has seen better days. But hey - PERFECT for banana bread! 🍞",
                f"This banana is living its best life... in the past tense. At {black:.0f}% black, it's smoothie material now! 🥤",
                f"Time traveler alert! ⏰ This banana has {black:.0f}% black spots. Quick - make it into something delicious before it's too late!"
            ]
        else:
            comments = [
                f"Still totally edible! With {brown:.0f}% brown, it's super sweet and perfect for baking. Banana bread time? 🍌➡️🍞",
                f"This banana is on the ripe side with {brown:.0f}% brown spots, but that just means extra sweetness! Use it today! 💛",
                f"Getting spotty with {brown:.0f}% brown, but don't you dare throw it away! Perfect for smoothies or baking! ♻️"
            ]
    else:
        comments = [
            f"Interesting specimen! {yellow:.0f}% yellow, {brown:.0f}% brown. Every banana tells a story! 📖🍌",
            f"This banana is on a journey! Currently at {yellow:.0f}% yellow. Monitor its progress! 🔬"
        ]
    
    return random.choice(comments)

@app.route("/api/process-image", methods=["POST"])
def process_image():
    """Main prediction endpoint matching frontend expectations"""
    try:
        # Debug logging
        print(f"Received request with files: {request.files}")
        print(f"Request form: {request.form}")
        print(f"Request headers: {dict(request.headers)}")
        
        # Check if models are loaded
        if model0 is None or model1 is None:
            print("ERROR: Models not loaded")
            return jsonify({"error": "Models not loaded"}), 500

        # Check if image file is present
        if 'image' not in request.files:
            print(f"ERROR: No 'image' in request.files. Available keys: {list(request.files.keys())}")
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            print("ERROR: Empty filename")
            return jsonify({"error": "Empty filename"}), 400
        
        print(f"Processing image: {file.filename}")

        # Read image from file
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Calculate color percentages for frontend
        percentages = calculate_color_percentages(img)
        
        # Create visualization
        processed_image = create_visualization(img, percentages)

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
        
        # Step 2: Use regression model for detailed analysis
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
        
        # Generate fun banana comment
        banana_comment = generate_banana_comment(percentages, result_class, pred_mean)
        
        # Return response in format expected by frontend
        return jsonify({
            "percentages": percentages,
            "processed_image": processed_image,
            "classification": result_class,
            "confidence": confidence,
            "status": interpretation["status"],
            "message": interpretation["message"],
            "days_estimate": interpretation["days_estimate"],
            "uncertainty": interpretation["uncertainty"],
            "banana_comment": banana_comment
        })

    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": model0 is not None and model1 is not None
    })

@app.route("/model-info", methods=["GET"])
def model_info():
    """Get information about loaded models"""
    info = {
        "classification_model": None,
        "regression_model": None
    }
    
    if model0 is not None:
        info["classification_model"] = {
            "type": str(type(model0)),
            "input_shape": str(model0.input_shape) if hasattr(model0, 'input_shape') else "N/A",
            "output_shape": str(model0.output_shape) if hasattr(model0, 'output_shape') else "N/A",
            "layers": len(model0.layers) if hasattr(model0, 'layers') else "N/A",
            "trainable_params": int(model0.count_params()) if hasattr(model0, 'count_params') else "N/A"
        }
    
    if model1 is not None:
        info["regression_model"] = {
            "type": str(type(model1)),
            "n_estimators": model1.n_estimators if hasattr(model1, 'n_estimators') else "N/A",
            "max_depth": model1.max_depth if hasattr(model1, 'max_depth') else "N/A",
            "n_features": model1.n_features_in_ if hasattr(model1, 'n_features_in_') else "N/A"
        }
    
    # Check model file paths and sizes
    import os
    model_files = {}
    if os.path.exists('model/banana_ripeness.h5'):
        size = os.path.getsize('model/banana_ripeness.h5')
        model_files['classification_model_file'] = {
            "path": "model/banana_ripeness.h5",
            "size_mb": round(size / (1024 * 1024), 2),
            "exists": True
        }
    
    if os.path.exists('model/banana_regression_uncertainty.joblib'):
        size = os.path.getsize('model/banana_regression_uncertainty.joblib')
        model_files['regression_model_file'] = {
            "path": "model/banana_regression_uncertainty.joblib",
            "size_mb": round(size / (1024 * 1024), 2),
            "exists": True
        }
    
    return jsonify({
        "models": info,
        "files": model_files,
        "python_version": os.sys.version,
        "tensorflow_version": tf.__version__,
        "sklearn_available": model1 is not None
    })

if __name__ == "__main__":
    print("Starting Banana Eats backend server...")
    print(f"Server running on port {PORT}")
    # debug=False for production, host 0.0.0.0 to accept external connections
    app.run(debug=False, host="0.0.0.0", port=PORT)