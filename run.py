#!/usr/bin/env python3
"""
Main run script for Airline Satisfaction Prediction Project
Manages all services and provides a unified interface
"""

import os
import sys
import subprocess
import time
import argparse
import webbrowser
from pathlib import Path

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(message):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{message.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(message):
    """Print success message"""
    print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")

def print_warning(message):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠️  {message}{Colors.ENDC}")

def print_error(message):
    """Print error message"""
    print(f"{Colors.FAIL}❌ {message}{Colors.ENDC}")

def print_info(message):
    """Print info message"""
    print(f"{Colors.OKBLUE}ℹ️  {message}{Colors.ENDC}")

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'models/xgboost_final.pkl',
        'models/scaler.pkl',
        'models/label_encoders.pkl',
        'models/model_info.json',
        'models/fill_values.json',
        'requirements.txt',
        'app.py',
        'api/main.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print_error("Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    return True

def install_requirements():
    """Install Python requirements"""
    print_info("Installing requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print_success("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install requirements: {e}")
        return False

def run_streamlit():
    """Run Streamlit application"""
    print_info("Starting Streamlit dashboard...")
    try:
        process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", 
                                   "--server.port=8501", "--server.address=0.0.0.0"])
        time.sleep(3)
        print_success("Streamlit running at http://localhost:8501")
        return process
    except Exception as e:
        print_error(f"Failed to start Streamlit: {e}")
        return None

def run_fastapi():
    """Run FastAPI application"""
    print_info("Starting FastAPI server...")
    try:
        os.chdir('api')
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", 
                                   "--host", "0.0.0.0", "--port", "8000", "--reload"])
        os.chdir('..')
        time.sleep(3)
        print_success("FastAPI running at http://localhost:8000")
        print_info("API docs available at http://localhost:8000/docs")
        return process
    except Exception as e:
        print_error(f"Failed to start FastAPI: {e}")
        return None

def run_tests():
    """Run all tests"""
    print_header("Running Tests")