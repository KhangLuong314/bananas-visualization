#!/bin/bash

echo "🍌 Starting Banana Eats..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Create model directory
mkdir -p model

# Start the backend
echo ""
echo "✅ Starting backend server..."
echo "Backend will run on: http://127.0.0.1:5001"
echo ""
echo "To use the app:"
echo "1. Keep this terminal open"
echo "2. Open bananaeats.html in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py