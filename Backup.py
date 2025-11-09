from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

#/Users/macpro/Desktop/backend/bananas-visualization/app.py

app = Flask(__name__)

# Load model once at startup
model = load_model("/Users/macpro/Desktop/backend/bananas-visualization/model/banana_ripeness (1).h5")

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img = preprocess_image(img)

    prediction = model.predict(img)[0][0]
    result = {
        "predicted_days": float(prediction),
        "interpretation": interpret_prediction(prediction)
    }
    return jsonify(result)

def interpret_prediction(days):
    if days < 3:
        return "Unripe — wait a few days 🍏"
    elif 3 <= days < 7:
        return "Ripening — almost ready 🍌"
    else:
        return "Perfectly ripe — eat now! 🍯"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
