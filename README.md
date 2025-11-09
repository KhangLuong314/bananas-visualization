🍌 Banana Ripeness Prediction App
🌟 Overview

The Banana Ripeness Prediction App uses deep learning and color analysis to determine how ripe a banana is based on an uploaded image.
This project combines TensorFlow, Flask, and scikit-learn in the backend, and a simple React or HTML/JS frontend, providing users with an intuitive and educational fruit visualization tool.

🚀 Features

📸 Upload an image of a banana for prediction

🧠 Dual-model system:

Model 1 (.h5) – CNN classification model that predicts Unripe, Ripe, or Overripe

Model 2 (.joblib) – Regression model estimating days to ripeness and uncertainty

🌈 Real-time response via Flask API

🌍 Deployed with:

Backend → Render

Frontend → GitHub Pages

🧩 Tech Stack
Component	Technology
Frontend	React / HTML / CSS / JavaScript
Backend	Python (Flask)
Machine Learning	TensorFlow / Keras / scikit-learn
Deployment	Render (Backend), GitHub Pages (Frontend)
Image Processing	OpenCV, Pillow
Data Analysis	NumPy, Pandas
🗂️ Project Structure
bananas-visualization/
│
├── app.py                     # Flask backend
├── model/
│   ├── banana_ripeness.h5     # CNN classification model
│   └── banana_regression_uncertainty.joblib  # Regression model
├── static/                    # Frontend assets (optional)
├── venv/                      # Virtual environment
└── requirements.txt           # Dependencies

⚙️ Setup and Run Locally
1️⃣ Clone the repository
git clone https://github.com/<your-username>/bananas-visualization.git
cd bananas-visualization

2️⃣ Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the backend server
python app.py


Once running, Flask will start on:
👉 http://127.0.0.1:5050

📤 API Endpoints
POST /predict

Upload a banana image for prediction.

Example using curl:

curl -X POST -F "image=@/path/to/banana.jpg" http://127.0.0.1:5050/predict


Response:

{
  "classification": "Ripe",
  "confidence": 0.94,
  "status": "Perfectly ripe 🍯",
  "message": "Perfectly ripe — eat now for the best flavor!",
  "days_estimate": 10.5,
  "uncertainty": 1.2,
  "model_used": "classification_and_regression"
}

GET /health

Returns model loading status and server health.

☁️ Deployment
Backend on Render

Push your code to GitHub

Create a new Web Service on Render

Connect your repo

Set the start command:

gunicorn app:app


Set environment:

PYTHON_VERSION = 3.12

Frontend on GitHub Pages

Build your React app (npm run build)

Push to the gh-pages branch

Enable GitHub Pages in repo settings → Branch → gh-pages

🧠 Model Information

banana_ripeness.h5 – CNN image classifier trained on labeled banana images

banana_regression_uncertainty.joblib – Random Forest model trained on HSV color and texture features to estimate ripeness duration

🧑‍💻 Contributors

Backend: Daniel Tran

Frontend: Mary Tran

ML Models: Khang Luong and Cat Dinh
