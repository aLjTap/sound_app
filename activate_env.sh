#!/bin/bash
# Virtual environment activation script

# Activate virtual environment
source venv/bin/activate

# Check Python version
echo "Python version: $(python --version)"
echo "Virtual environment activated successfully!"
echo "Location: $(which python)"

# List installed packages
echo ""
echo "Installed packages:"
pip list --format=columns
