#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Create necessary directories if they don't exist
mkdir -p data/focus-groups
mkdir -p data/raw
mkdir -p data/processed
mkdir -p results/visualizations

echo "Setup complete! To activate the virtual environment, run:"
echo "source venv/bin/activate" 