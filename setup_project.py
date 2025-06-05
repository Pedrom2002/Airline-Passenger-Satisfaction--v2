#!/usr/bin/env python3
"""
Setup script for Airline Satisfaction Project
Prepares the entire project for execution
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# ANSI colors
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(msg):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{msg}{Colors.ENDC}")

def print_success(msg):
    print(f"{Colors.OKGREEN}✅ {msg}{Colors.ENDC}")

def print_error(msg):
    print(f"{Colors.FAIL}❌ {msg}{Colors.ENDC}")

def print_warning(msg):
    print(f"{Colors.WARNING}⚠️  {msg}{Colors.ENDC}")

def print_info(msg):
    print(f"{Colors.OKBLUE}ℹ️  {msg}{Colors.ENDC}")

def create_directories():
    """Create necessary directories"""
    print_header("Creating directory structure...")
    
    directories = [
        'models',
        'data/raw',
        'data/processed',
        'api',
        'tests',
        'logs',
        'monitoring',
        'results',
        'plots'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print_success(f"Created {directory}/")

def create_api_init():
    """Create __init__.py for api module"""
    init_content = '"""API module for Airline Satisfaction Prediction"""'
    
    with open('api/__init__.py', 'w') as f:
        f.write(init_content)
    
    print_success("Created api/__init__.py")

def check_model_files():
    """Check if model files exist"""
    print_header("Checking model files...")
    
    required_files = [
        'models/xgboost_final.pkl',
        'models/scaler.pkl',
        'models/label_encoders.pkl',
        'models/model_info.json',
        'models/fill_values.json'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print_success(f"Found {file}")
        else:
            missing_files.append(file)
            print_error(f"Missing {file}")
    
    if missing_files:
        print_warning("\nSome model files are missing!")
        print_info("Please run the modeling notebook to generate these files.")
        print_info("Or download them from the repository.")
        return False
    
    return True

def create_sample_env():
    """Create .env file from example"""
    if not os.path.exists('.env') and os.path.exists('.env.example'):
        shutil.copy('.env.example', '.env')
        print_success("Created .env file from .env.example")
        print_info("Please update .env with your configuration")
    elif os.path.exists('.env'):
        print_info(".env file already exists")

def install_requirements():
    """Install Python requirements"""
    print_header("Installing Python dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        print_success("Updated pip")
        
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print_success("Installed all requirements")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install requirements: {e}")
        return False

def create_test_init():
    """Create __init__.py for tests"""
    with open('tests/__init__.py', 'w') as f:
        f.write('"""Test module"""')
    print_success("Created tests/__init__.py")

def download_sample_data():
    """Create sample data for testing"""
    print_header("Creating sample data...")
    
    sample_csv = """Gender,Customer Type,Age,Type of Travel,Class,Flight Distance,Inflight wifi service,Departure/Arrival time convenient,Ease of Online booking,Gate location,Food and drink,Online boarding,Seat comfort,Inflight entertainment,On-board service,Leg room service,Baggage handling,Checkin service,Inflight service,Cleanliness,Departure Delay in Minutes,Arrival Delay in Minutes
Male,Loyal Customer,35,Business travel,Business,1500,4,4,5,3,4,5,4,4,5,4,5,4,5,5,0,0
Female,Disloyal Customer,28,Personal Travel,Eco,800,2,3,3,2,2,3,2,2,3,2,3,3,3,3,15,20
Male,Loyal Customer,42,Business travel,Eco Plus,2000,3,4,4,3,3,4,3,3,4,3,4,4,4,4,5,5
"""
    
    with open('data/sample_passengers.csv', 'w') as f:
        f.write(sample_csv)
    
    print_success("Created data/sample_passengers.csv")

def verify_installation():
    """Verify that everything is properly installed"""
    print_header("Verifying installation...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print_success(f"Python {python_version.major}.{python_version.minor} ✓")
    else:
        print_error(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")
    
    # Check key packages
    packages = ['pandas', 'numpy', 'scikit-learn', 'xgboost', 'streamlit', 'fastapi']
    
    for package in packages:
        try:
            __import__(package)
            print_success(f"{package} ✓")
        except ImportError:
            print_error(f"{package} ✗")

def main():
    """Main setup function"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'Airline Satisfaction Project Setup'.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")
    
    # Create directories
    create_directories()
    
    # Create necessary files
    create_api_init()
    create_test_init()
    create_sample_env()
    
    # Check model files
    model_files_ok = check_model_files()
    
    # Install requirements
    if input("\nInstall Python requirements? (y/n): ").lower() == 'y':
        install_requirements()
    
    # Create sample data
    download_sample_data()
    
    # Verify installation
    verify_installation()
    
    # Final message
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}{Colors.BOLD}Setup Complete!{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")
    
    if model_files_ok:
        print_success("Project is ready to run!")
        print_info("\nNext steps:")
        print("1. Run the complete system: python run.py")
        print("2. Or run individual components:")
        print("   - Streamlit: streamlit run app.py")
        print("   - API: cd api && uvicorn main:app --reload")
        print("3. Run tests: pytest tests/")
    else:
        print_warning("Project setup is incomplete!")
        print_info("Please generate model files by running the modeling notebook")
        print_info("Or download them from the repository")
    
    print(f"\n{Colors.OKBLUE}Happy coding! 🚀{Colors.ENDC}\n")

if __name__ == "__main__":
    main()