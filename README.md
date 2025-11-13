# 🍌 Banana Eats - Setup Guide

AI-Powered Banana Ripeness Analyzer

## 📁 Project Structure

```
bananas-visualization/
├── bananaeats.html          # Frontend HTML
├── styles.css               # Frontend styles
├── script.js                # Frontend JavaScript
├── app.py                   # Backend Flask server
├── rotating-banana-banana.gif  # Upload animation
├── daniel.png               # Developer image
├── khang.png                # Developer image
├── mary.png                 # Developer image
├── cat.png                  # Developer image
├── rotating-banana-banana.gif
├── requirements.txt
└── model/                   # Model directory (auto-created)
    ├── banana_ripeness.h5
    └── banana_regression_uncertainty.joblib
```

## 🔧 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Modern web browser (Chrome, Firefox, Safari, Edge)

## 📦 Installation

### 1. Install Python Dependencies

Create a virtual environment (recommended):

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

Install required packages:

```bash
pip install flask flask-cors tensorflow opencv-python joblib pandas pillow numpy gdown
```

**Package versions (recommended):**
- flask>=2.3.0
- flask-cors>=4.0.0
- tensorflow>=2.13.0
- opencv-python>=4.8.0
- joblib>=1.3.0
- pandas>=2.0.0
- pillow>=10.0.0
- numpy>=1.24.0
- gdown>=4.7.0

### 2. Download Models

The models will be automatically downloaded when you first run the server. Alternatively, you can manually download them:

**Classification Model:**
- URL: https://drive.google.com/file/d/1W0c3OWKpOaMxtuEofTCuAqyR5tM6b1wU/view?usp=sharing
- Save to: `model/banana_ripeness.h5`

**Regression Model:**
- URL: https://drive.google.com/file/d/1ESymScvSBZw1X7EWwRgefa3u46tyFgMK/view?usp=drive_link
- Save to: `model/banana_regression_uncertainty.joblib`

### 3. Add Developer Images

Place these images in the project root:
- `daniel.png`
- `khang.png`
- `mary.png`
- `cat.png`
- `rotating-banana-banana.gif`

## 🚀 Running the Application

### Step 1: Start the Backend Server

```bash
python app.py
```

You should see:
```
Starting Banana Eats backend server...
Models loaded successfully!
Server running on http://127.0.0.1:5000
 * Running on http://0.0.0.0:5000
```

**Note:** Keep this terminal window open while using the app.

### Step 2: Open the Frontend

Simply open `bananaeats.html` in your web browser:

**Option A: Double-click the file**
- Navigate to the project folder
- Double-click `bananaeats.html`

**Option B: Use a local server (recommended for development)**
```bash
# Using Python's built-in server
python -m http.server 8000
```
Then open: http://localhost:8000/bananaeats.html

## 🎯 Using the Application

1. **Upload a Banana Image**
   - Click or drag-and-drop an image into the upload area
   - Or click "Open Camera" to take a photo (mobile/desktop with camera)

2. **Analyze**
   - Click the "Analyze Banana" button
   - Wait for the AI to process the image

3. **View Results**
   - See the ripeness status (Unripe, Ripe, Overripe, etc.)
   - View color analysis breakdown (Yellow, Brown, Black, White percentages)
   - Read the Banana Bot's comment about your banana
   - See the processed image visualization

4. **Meet the Developers**
   - Click the "Meet the Developers!" button in the top-left
   - Watch the bouncing developer faces!

## 🐛 Troubleshooting

### Backend Issues

**Problem: "Models not loaded" error**
- Solution: Ensure models are downloaded to the `model/` directory
- Try running `python app.py` again to trigger auto-download

**Problem: Port already in use**
- Solution: Change the port in `app.py`:
  ```python
  app.run(debug=True, host="0.0.0.0", port=5001)  # Use different port
  ```
- Update frontend `script.js`:
  ```javascript
  const backendURL = "http://127.0.0.1:5001/api/process-image";
  ```

**Problem: CORS errors**
- Solution: Make sure `flask-cors` is installed: `pip install flask-cors`

### Frontend Issues

**Problem: "Make sure the backend server is running" error**
- Check if backend is running on http://127.0.0.1:5000
- Verify no firewall is blocking the connection
- Check browser console (F12) for detailed errors

**Problem: Images don't load**
- Ensure all image files are in the correct directory
- Check file names match exactly (case-sensitive)

**Problem: Camera button doesn't work**
- Grant camera permissions in your browser
- Use HTTPS or localhost (required for camera access)

**Problem: "Meet the Developers" button doesn't work**
- Ensure all developer images exist in the project directory
- Check browser console for errors

## 🔌 API Endpoints

### POST /api/process-image
Analyzes banana image and returns ripeness data.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: image file

**Response:**
```json
{
  "percentages": {
    "yellow": 65.2,
    "brown": 23.4,
    "black": 5.1,
    "white": 6.3
  },
  "processed_image": "base64_encoded_image...",
  "classification": "Ripe",
  "confidence": 0.95,
  "status": "Perfectly ripe 🍯",
  "message": "Perfectly ripe — eat now for the best flavor!",
  "days_estimate": 12.5,
  "uncertainty": 2.1
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

## 🎨 Features

- **AI-Powered Analysis**: Uses deep learning models for accurate ripeness detection
- **Color Breakdown**: Shows detailed color percentages
- **Banana Bot**: AI-generated personalized comments about your banana
- **Interactive UI**: Smooth animations and responsive design
- **Camera Support**: Take photos directly from your device
- **Developer Easter Egg**: Bouncing physics-based developer faces

## 📝 Development Notes

### Modifying the Backend

The backend uses two models:
1. **Classification Model** (`model0`): Categorizes banana as Overripe/Ripe/Unripe
2. **Regression Model** (`model1`): Estimates ripeness on a continuous scale

To modify analysis logic, edit the `process_image()` function in `app.py`.

### Modifying the Frontend

- **HTML**: Edit `bananaeats.html` for structure changes
- **CSS**: Edit `styles.css` for styling changes
- **JavaScript**: Edit `script.js` for functionality changes

## 🤝 Team

- Cat Dinh, Khang Luong - ML Engineers
- Mary Tran, Daniel Tran - Frontend Developers

## 📄 License

Made with 💛 for banana lovers everywhere

---

**Need Help?** Check the console logs (F12 in browser) for detailed error messages.