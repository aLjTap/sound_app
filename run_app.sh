#!/bin/bash
# Script to run the Pro Equalizer application

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run: python3.12 -m venv venv"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if requirements are installed
if ! python -c "import librosa, customtkinter, pyaudio" 2>/dev/null; then
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

# Run the application
echo "Starting Pro Equalizer..."
python main.py
