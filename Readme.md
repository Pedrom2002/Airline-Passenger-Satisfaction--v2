# ✈️ Airline Passenger Satisfaction Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![CatBoost](https://img.shields.io/badge/CatBoost-Latest-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A complete ML system for predicting airline passenger satisfaction using CatBoost with 45+ engineered features**

[📊 Dashboard Demo](#demo) | [🚀 Quick Start](#quick-start) | [📖 Documentation](#documentation) | [🔌 API](#api-documentation)

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [API Documentation](#-api-documentation)
- [Dashboard Features](#-dashboard-features)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

This project implements a complete Machine Learning system for predicting airline passenger satisfaction. Developed as a portfolio project, it demonstrates advanced ML Engineering skills including:

- **Advanced Feature Engineering**: 45+ features including 23 engineered ones
- **High-Performance Model**: CatBoost with 96.4% accuracy
- **RESTful API**: FastAPI with robust endpoints and automatic documentation
- **Interactive Dashboard**: Streamlit interface with real-time visualizations

### 🎯 Business Problem

Airlines need to understand and predict passenger satisfaction to:
- Improve service quality
- Increase customer retention
- Optimize resource allocation
- Enhance competitive advantage

## ✨ Features

### 🤖 Machine Learning
- **Algorithm**: CatBoost Enhanced (optimized for categorical features)
- **Feature Engineering**: 23 engineered features including aggregations, interactions, and statistics
- **Performance**: 96.4% accuracy, 99.5% ROC-AUC
- **Processing**: Complete pipeline with data validation and transformation


### 📊 Interactive Dashboard
- **Framework**: Streamlit
- **Visualizations**: Plotly for interactive charts
- **Features**: 
  - Real-time prediction
  - Batch data upload
  - Feature importance analysis
  - Performance monitoring


## 🚀 Installation

### Prerequisites
- Python 3.9+
- Docker (optional)
- Git

### 🐍 Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/Pedrom2002/Airline-Passenger-Satisfaction--v2.git
cd Airline-Passenger-Satisfaction--v2

# Create virtual environment (anaconda prompt)
conda create -n airline-satisfaction python=3.10 -y
conda activate airline-satisfaction


# Install dependencies
pip install -r requirements.txt

#Run app
python -m streamlit run app.py
or

#Run api
uvicorn api.main:app --reload --port 8000
```

### 🐳 Docker Installation

```bash
# 1. Clone the repository
git clone https://github.com/Pedrom2002/Airline-Passenger-Satisfaction.git
cd Airline-Passenger-Satisfaction

# 2. Build and start containers
cd api
docker-compose up -d

# 3. Access services
# API: http://localhost:8000
# Dashboard: http://localhost:8501
# Grafana: http://localhost:3000
```

## 📖 Usage

### 🎯 Streamlit Dashboard

1. Access http://localhost:8501
2. Navigate through pages:
   - **🏠 Dashboard**: Overview and metrics
   - **🎯 Prediction**: Make individual predictions
   - **📊 Batch Analysis**: Upload CSV files
   - **🔬 Model Insights**: Feature analysis
   - **📈 Monitoring**: Real-time performance


## 📁 Project Structure

```
airline-satisfaction/
├── 📊 app/                         # Streamlit application
│   ├── app.py                      # Main entry point
│   ├── pages/                      # Dashboard pages
│   │   ├── dashboard.py            # Main dashboard
│   │   ├── prediction.py           # Individual prediction
│   │   ├── batch_analysis.py       # Batch analysis
│   │   ├── model_insights.py       # Model insights
│   │   └── monitoring.py           # Monitoring
│   └── .env                        # Environment settings
│
├── 🌐 api/                         # FastAPI API
│   ├── main.py                     # FastAPI application
│   ├── start.py                    # Startup script
│   ├── requirements.txt            # Python dependencies
│   ├── Dockerfile                  # Docker configuration
│   ├── docker-compose.yml          # Service orchestration
│   ├── nginx.conf                  # Proxy configuration
│   └── prometheus.yml              # Metrics configuration
│
├── 📓 notebooks/                   # Jupyter notebooks
│   ├── 01_EDA.ipynb               # Exploratory analysis
│   ├── 02_Modeling.ipynb          # Model development
│   └── 03_Deployment.ipynb        # Deployment pipeline
│
├── 🤖 models/                      # Model artifacts
│   ├── catboost_model.pkl         # Trained model
│   ├── preprocessor.pkl           # Preprocessing pipeline
│   └── feature_names.json         # Feature list
│
├── 📊 data/                        # Datasets
│   └── test.csv
│   └── train.csv                                # CI/CD pipeline
│
├── 📋 requirements.txt             # Global dependencies
├── 📖 README.md                    # This file
├── 📜 LICENSE                      # MIT License
└── 🔒 .gitignore                  # Ignored files
```

## 📊 Model Performance

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 96.4% |
| **Precision** | 96.2% |
| **Recall** | 96.8% |
| **F1-Score** | 95.8% |
| **ROC-AUC** | 99.5% |

### Top 10 Most Important Features

1. 🧹 **Inflight wifi service**
2. 💺 **Type of Travel** 
3. 🎬 **Customer Type** 
4. 📱 **Online Boarding** 
5. 🍽️ **Checkin service** 
6. 📶 **Baggage handling** 
7. ✈️ **business_loyal** 
8. 🎂 **Seat comfort** 


## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Pedro M.**
- 📧 Email: pedrom02.dev@gmail.com
- 🔗 LinkedIn: [linkedin.com/in/pedro-morais-07163a275/](https://linkedin.com/in/pedro-morais-07163a275/)
- 🐙 GitHub: [@Pedrom2002](https://github.com/Pedrom2002)

## 🙏 Acknowledgments

- Dataset: [Kaggle - Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
- Inspiration: ML community projects
- Tools: Streamlit, FastAPI, CatBoost

---

<div align="center">
  <p>⭐ If you found this project useful, consider giving it a star!</p>
  <p>Made with ❤️ by Pedro M.</p>
</div>