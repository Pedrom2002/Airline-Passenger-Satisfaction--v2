#!/usr/bin/env python3
"""
Setup script for Airline Passenger Satisfaction project
Automates the initial configuration and setup process
"""

import os
import sys
import subprocess
import platform
import json
import shutil
from pathlib import Path
from datetime import datetime

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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

def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error(f"Python 3.8+ required. Current version: {version.major}.{version.minor}")
        sys.exit(1)
    print_success(f"Python {version.major}.{version.minor} detected")

def check_system_requirements():
    """Check system requirements"""
    print_header("Checking System Requirements")
    
    # Check Python
    check_python_version()
    
    # Check OS
    os_name = platform.system()
    print_info(f"Operating System: {os_name}")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print_info(f"Available memory: {memory.available / (1024**3):.2f} GB")
        if memory.available < 4 * (1024**3):  # Less than 4GB
            print_warning("Low memory detected. Model training might be slow.")
    except ImportError:
        print_warning("psutil not installed. Cannot check memory.")
    
    # Check for required commands
    commands = {
        'git': 'Git version control',
        'docker': 'Docker containerization (optional)',
        'make': 'Make build tool (optional)'
    }
    
    for cmd, description in commands.items():
        if shutil.which(cmd):
            print_success(f"{description} found")
        else:
            print_warning(f"{description} not found")

def create_directory_structure():
    """Create project directory structure"""
    print_header("Creating Directory Structure")
    
    directories = [
        'data/raw',
        'data/processed',
        'data/external',
        'models/experiments',
        'logs',
        'monitoring',
        'results',
        'plots',
        'tests/unit',
        'tests/integration',
        'docs',
        'scripts',
        'src/data',
        'src/features',
        'src/models',
        'src/visualization',
        'src/utils',
        'api/endpoints',
        'api/schemas',
        '.github/workflows'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print_success(f"Created {directory}/")

def create_environment_files():
    """Create environment configuration files"""
    print_header("Creating Configuration Files")
    
    # .env file
    env_content = """# Environment Variables
MODEL_PATH=models/final_model.pkl
SCALER_PATH=models/scaler.pkl
LOG_LEVEL=INFO
API_PORT=8000
STREAMLIT_PORT=8501

# Database
DATABASE_URL=sqlite:///monitoring/predictions.db

# MLflow
MLFLOW_TRACKING_URI=sqlite:///monitoring/mlflow.db
MLFLOW_EXPERIMENT_NAME=airline_satisfaction

# Security
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here

# Cloud Storage (optional)
# AWS_ACCESS_KEY_ID=
# AWS_SECRET_ACCESS_KEY=
# S3_BUCKET_NAME=

# Monitoring
GRAFANA_API_KEY=
PROMETHEUS_PORT=9090
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    print_success("Created .env file")
    
    # .gitignore file
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv
pip-log.txt
pip-delete-this-directory.txt
.pytest_cache/
.coverage
htmlcov/
.tox/
.mypy_cache/
.ruff_cache/

# Jupyter
.ipynb_checkpoints
*.ipynb_checkpoints

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep
*.csv
*.xlsx
*.parquet

# Models
models/*.pkl
models/*.joblib
models/*.h5
models/experiments/*

# Logs
logs/
*.log

# Environment
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Docker
.dockerignore

# MLflow
mlruns/
monitoring/mlflow.db

# Database
*.db
*.sqlite
*.sqlite3

# Temporary files
tmp/
temp/
*.tmp

# Distribution
dist/
build/
*.egg-info/

# Documentation
docs/_build/
site/

# Security
*.pem
*.key
secrets/
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print_success("Created .gitignore file")
    
    # Create .gitkeep files
    gitkeep_dirs = ['data/raw', 'data/processed', 'models/experiments']
    for directory in gitkeep_dirs:
        Path(f"{directory}/.gitkeep").touch()

def install_dependencies():
    """Install Python dependencies"""
    print_header("Installing Dependencies")
    
    try:
        # Upgrade pip
        print_info("Upgrading pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        print_success("Pip upgraded")
        
        # Install requirements
        if os.path.exists('requirements.txt'):
            print_info("Installing requirements...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                          check=True)
            print_success("Requirements installed")
        else:
            print_warning("requirements.txt not found")
        
        # Install development requirements
        if os.path.exists('requirements-dev.txt'):
            print_info("Installing development requirements...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"], 
                          check=True)
            print_success("Development requirements installed")
            
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        sys.exit(1)

def setup_git_hooks():
    """Setup git pre-commit hooks"""
    print_header("Setting up Git Hooks")
    
    try:
        # Initialize git if not already
        if not os.path.exists('.git'):
            subprocess.run(['git', 'init'], check=True)
            print_success("Git repository initialized")
        
        # Install pre-commit
        subprocess.run([sys.executable, "-m", "pip", "install", "pre-commit"], 
                      check=True, capture_output=True)
        
        # Create pre-commit config
        precommit_config = """repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-json
      - id: check-merge-conflict
      
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.9
        
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]
        
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: ["--max-line-length=100", "--ignore=E203,W503"]
"""
        
        with open('.pre-commit-config.yaml', 'w') as f:
            f.write(precommit_config)
        
        # Install pre-commit hooks
        subprocess.run(['pre-commit', 'install'], check=True)
        print_success("Pre-commit hooks installed")
        
    except subprocess.CalledProcessError as e:
        print_warning(f"Failed to setup git hooks: {e}")

def download_sample_data():
    """Download sample data for testing"""
    print_header("Setting up Sample Data")
    
    # Create sample data
    sample_data = {
        "sample_passenger.json": {
            "gender": "Male",
            "customer_type": "Loyal Customer",
            "age": 35,
            "type_of_travel": "Business travel",
            "travel_class": "Business",
            "flight_distance": 1500,
            "inflight_wifi_service": 4,
            "departure_arrival_time_convenient": 4,
            "ease_of_online_booking": 5,
            "gate_location": 3,
            "food_and_drink": 4,
            "online_boarding": 5,
            "seat_comfort": 4,
            "inflight_entertainment": 4,
            "on_board_service": 5,
            "leg_room_service": 4,
            "baggage_handling": 5,
            "checkin_service": 4,
            "inflight_service": 5,
            "cleanliness": 5,
            "departure_delay_in_minutes": 0,
            "arrival_delay_in_minutes": 0
        }
    }
    
    # Save sample data
    os.makedirs('data/samples', exist_ok=True)
    for filename, data in sample_data.items():
        with open(f'data/samples/{filename}', 'w') as f:
            json.dump(data, f, indent=2)
    
    print_success("Sample data created in data/samples/")
    print_info("Download full dataset from: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction")

def create_initial_tests():
    """Create initial test files"""
    print_header("Creating Initial Tests")
    
    # Test for model loading
    test_model = '''import pytest
import pickle
import os

def test_model_exists():
    """Test if model file exists"""
    assert os.path.exists('models/final_model.pkl'), "Model file not found"

def test_model_loads():
    """Test if model can be loaded"""
    if os.path.exists('models/final_model.pkl'):
        with open('models/final_model.pkl', 'rb') as f:
            model = pickle.load(f)
        assert model is not None, "Model loaded as None"

def test_scaler_exists():
    """Test if scaler file exists"""
    assert os.path.exists('models/scaler.pkl'), "Scaler file not found"
'''
    
    with open('tests/test_model.py', 'w') as f:
        f.write(test_model)
    
    # Test for API
    test_api = '''import pytest
from fastapi.testclient import TestClient
import sys
sys.path.append('.')

def test_health_endpoint():
    """Test health check endpoint"""
    from api.app import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"
'''
    
    with open('tests/test_api.py', 'w') as f:
        f.write(test_api)
    
    print_success("Initial tests created")

def print_next_steps():
    """Print next steps for the user"""
    print_header("Setup Complete! 🎉")
    
    print(f"{Colors.OKGREEN}Your Airline Passenger Satisfaction project is ready!{Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print("1. Activate virtual environment:")
    print(f"   {Colors.OKCYAN}source venv/bin/activate{Colors.ENDC} (Linux/Mac)")
    print(f"   {Colors.OKCYAN}venv\\Scripts\\activate{Colors.ENDC} (Windows)\n")
    
    print("2. Download the dataset:")
    print(f"   {Colors.OKCYAN}python scripts/download_data.py{Colors.ENDC}\n")
    
    print("3. Train the model:")
    print(f"   {Colors.OKCYAN}python notebooks/02_Modeling.ipynb{Colors.ENDC}\n")
    
    print("4. Run the application:")
    print(f"   {Colors.OKCYAN}streamlit run app.py{Colors.ENDC}\n")
    
    print("5. Start the API:")
    print(f"   {Colors.OKCYAN}uvicorn api.app:app --reload{Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}Useful Commands:{Colors.ENDC}")
    print(f"   {Colors.OKCYAN}make help{Colors.ENDC}     - Show all available commands")
    print(f"   {Colors.OKCYAN}make test{Colors.ENDC}     - Run tests")
    print(f"   {Colors.OKCYAN}make lint{Colors.ENDC}     - Check code quality")
    print(f"   {Colors.OKCYAN}make docker{Colors.ENDC}   - Build Docker image\n")
    
    print(f"{Colors.BOLD}Documentation:{Colors.ENDC}")
    print(f"   API Docs: {Colors.OKCYAN}http://localhost:8000/docs{Colors.ENDC}")
    print(f"   Project Docs: {Colors.OKCYAN}docs/README.md{Colors.ENDC}\n")
    
    print(f"{Colors.OKGREEN}Happy coding! 🚀{Colors.ENDC}")

def main():
    """Main setup function"""
    print_header("Airline Passenger Satisfaction Project Setup")
    
    try:
        # Run setup steps
        check_system_requirements()
        create_directory_structure()
        create_environment_files()
        install_dependencies()
        setup_git_hooks()
        download_sample_data()
        create_initial_tests()
        
        # Show completion message
        print_next_steps()
        
    except KeyboardInterrupt:
        print_error("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()