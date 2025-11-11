#!/bin/bash

# Project Initialization Script
# Smoking Cessation ML Project
# Run this script to set up your development environment

echo "=========================================="
echo "Smoking Cessation ML Project Setup"
echo "=========================================="
echo ""

# Navigate to project directory
cd "$(dirname "$0")"
PROJECT_DIR=$(pwd)
echo "✓ Project directory: $PROJECT_DIR"
echo ""

# Check if Python is installed
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Python found: $PYTHON_VERSION"
else
    echo "✗ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install requirements
echo "Installing Python packages..."
echo "This may take a few minutes..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
echo "✓ All packages installed"
echo ""

# Check for PATH Study data
echo "Checking for PATH Study data..."
if [ -z "$(ls -A data/raw/*.dta 2>/dev/null)" ] && [ -z "$(ls -A data/raw/*.sav 2>/dev/null)" ]; then
    echo "⚠️  WARNING: No data files found in data/raw/"
    echo ""
    echo "ACTION REQUIRED:"
    echo "1. Register at ICPSR: https://www.icpsr.umich.edu/"
    echo "2. Download PATH Study Waves 1-5 (STATA .dta or SPSS .sav format)"
    echo "3. Download ADULT files only (NOT Youth or Parent files)"
    echo "4. Place data files in: $PROJECT_DIR/data/raw/"
    echo ""
    echo "See PATH_DATA_GUIDE.md for detailed instructions"
    echo ""
    echo "Continue setup? (y/n)"
    read -r response
    if [[ "$response" != "y" ]]; then
        echo "Setup paused. Run this script again after downloading data."
        exit 0
    fi
else
    echo "✓ PATH Study data files found in data/raw/"
    ls data/raw/*.dta 2>/dev/null || ls data/raw/*.sav 2>/dev/null
fi
echo ""

# Initialize Git if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial commit: Project structure and implementation plan"
    echo "✓ Git repository initialized"
else
    echo "✓ Git repository already initialized"
fi
echo ""

# Create .env file for any environment variables
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOL
# Environment variables for Smoking Cessation ML Project
PROJECT_DIR=$PROJECT_DIR
RANDOM_SEED=42
EOL
    echo "✓ .env file created"
else
    echo "✓ .env file already exists"
fi
echo ""

# Test imports
echo "Testing Python imports..."
python3 << EOTEST
import sys
sys.path.append('src')
try:
    import pandas as pd
    import numpy as np
    import sklearn
    import xgboost
    import shap
    import streamlit
    print("✓ All critical packages imported successfully")
    print("")
    print("Package versions:")
    print(f"  - pandas: {pd.__version__}")
    print(f"  - numpy: {np.__version__}")
    print(f"  - scikit-learn: {sklearn.__version__}")
    print(f"  - xgboost: {xgboost.__version__}")
    print(f"  - shap: {shap.__version__}")
    print(f"  - streamlit: {streamlit.__version__}")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
EOTEST

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ SETUP COMPLETE!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Activate environment: source venv/bin/activate"
    echo "2. Start Jupyter: jupyter notebook"
    echo "3. Read ACTION_GUIDE.md for detailed instructions"
    echo ""
    echo "If you haven't already:"
    echo "- Register at ICPSR: https://www.icpsr.umich.edu/"
    echo "- Download PATH Study Waves 1-5 (STATA .dta or SPSS .sav format)"
    echo "- Download ADULT files only (NOT Youth or Parent files)"
    echo "- Place data files in: $PROJECT_DIR/data/raw/"
    echo "- See PATH_DATA_GUIDE.md for detailed download instructions"
    echo ""
    echo "Project timeline: 16 days"
    echo "Target performance: ROC-AUC > 0.70"
    echo ""
    echo "Good luck! 🚀"
else
    echo ""
    echo "✗ Setup encountered errors. Please check output above."
    exit 1
fi
