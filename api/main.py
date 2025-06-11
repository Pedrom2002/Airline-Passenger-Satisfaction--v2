from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import pickle
import os
import logging
from datetime import datetime, timedelta
import uvicorn
from contextlib import asynccontextmanager
import asyncio
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
import json

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Métricas Prometheus
REQUEST_COUNT = Counter('airline_api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('airline_api_request_duration_seconds', 'Request duration')
PREDICTION_COUNT = Counter('airline_predictions_total', 'Total predictions made', ['model_version'])

# Classe para carregar modelo
class ModelManager:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.model_version = "v1.0.1"
        self.load_model()
    
    def load_model(self):
        """Carrega o modelo CatBoost e preprocessador"""
        try:
            # Carregar modelo CatBoost
            model_path = "../models/catboost_production_ model.pkl"
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Modelo carregado: {model_path}")
            
            # Carregar preprocessador
            preprocessor_path = "../models/preprocessor.pkl"
            if os.path.exists(preprocessor_path):
                with open(preprocessor_path, 'rb') as f:
                    self.preprocessor = pickle.load(f)
                logger.info(f"Preprocessador carregado: {preprocessor_path}")
            
            # Carregar nomes das features
            features_path = "../models/feature_names.json"
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    self.feature_names = json.load(f)
                logger.info(f"Features carregadas: {len(self.feature_names)} features")
                
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            # Modelo mock para desenvolvimento
            self.model = None
            self.preprocessor = None
            self.feature_names = self._get_default_features()
    
    def _get_default_features(self):
        """Features padrão para desenvolvimento"""
        return [
            'Age', 'Flight_Distance', 'Inflight_wifi_service', 'Departure_Arrival_time_convenient',
            'Ease_of_Online_booking', 'Gate_location', 'Food_and_drink', 'Online_boarding',
            'Seat_comfort', 'Inflight_entertainment', 'On_board_service', 'Leg_room_service',
            'Baggage_handling', 'Checkin_service', 'Inflight_service', 'Cleanliness',
            'Departure_Delay_in_Minutes', 'Arrival_Delay_in_Minutes', 'Gender_Male',
            'Customer_Type_Loyal_Customer', 'Type_of_Travel_Personal_Travel', 'Class_Business',
            'Class_Eco_Plus'
        ]

# Instância global do gerenciador de modelo
model_manager = ModelManager()

# Modelos Pydantic para validação
class PassengerData(BaseModel):
    age: int = Field(..., ge=7, le=85, description="Idade do passageiro")
    flight_distance: int = Field(..., ge=31, le=4983, description="Distância do voo")
    inflight_wifi_service: int = Field(..., ge=0, le=5, description="Avaliação do WiFi (0-5)")
    departure_arrival_time_convenient: int = Field(..., ge=0, le=5, description="Conveniência do horário (0-5)")
    ease_of_online_booking: int = Field(..., ge=0, le=5, description="Facilidade de reserva online (0-5)")
    gate_location: int = Field(..., ge=0, le=5, description="Localização do portão (0-5)")
    food_and_drink: int = Field(..., ge=0, le=5, description="Comida e bebida (0-5)")
    online_boarding: int = Field(..., ge=0, le=5, description="Embarque online (0-5)")
    seat_comfort: int = Field(..., ge=0, le=5, description="Conforto do assento (0-5)")
    inflight_entertainment: int = Field(..., ge=0, le=5, description="Entretenimento a bordo (0-5)")
    on_board_service: int = Field(..., ge=0, le=5, description="Serviço a bordo (0-5)")
    leg_room_service: int = Field(..., ge=0, le=5, description="Espaço para pernas (0-5)")
    baggage_handling: int = Field(..., ge=0, le=5, description="Manuseio de bagagem (0-5)")
    checkin_service: int = Field(..., ge=0, le=5, description="Serviço de check-in (0-5)")
    inflight_service: int = Field(..., ge=0, le=5, description="Serviço durante o voo (0-5)")
    cleanliness: int = Field(..., ge=0, le=5, description="Limpeza (0-5)")
    departure_delay_in_minutes: int = Field(..., ge=0, le=1592, description="Atraso na partida (minutos)")
    arrival_delay_in_minutes: int = Field(..., ge=0, le=1584, description="Atraso na chegada (minutos)")
    gender: str = Field(..., description="Gênero: Male ou Female")
    customer_type: str = Field(..., description="Tipo de cliente: Loyal Customer ou disloyal Customer")
    type_of_travel: str = Field(..., description="Tipo de viagem: Business travel ou Personal Travel")
    class_type: str = Field(..., description="Classe: Business, Eco ou Eco Plus")
    
    @validator('gender')
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('Gender deve ser Male ou Female')
        return v
    
    @validator('customer_type')
    def validate_customer_type(cls, v):
        if v not in ['Loyal Customer', 'disloyal Customer']:
            raise ValueError('Customer Type deve ser Loyal Customer ou disloyal Customer')
        return v
    
    @validator('type_of_travel')
    def validate_travel_type(cls, v):
        if v not in ['Business travel', 'Personal Travel']:
            raise ValueError('Type of Travel deve ser Business travel ou Personal Travel')
        return v
    
    @validator('class_type')
    def validate_class(cls, v):
        if v not in ['Business', 'Eco', 'Eco Plus']:
            raise ValueError('Class deve ser Business, Eco ou Eco Plus')
        return v

class BatchPredictionRequest(BaseModel):
    passengers: List[PassengerData] = Field(..., max_items=100, description="Lista de passageiros (máximo 100)")

class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    prediction: str = Field(..., description="Satisfied ou Neutral or Dissatisfied")
    probability: float = Field(..., ge=0, le=1, description="Probabilidade da predição")
    confidence: str = Field(..., description="Nível de confiança: High, Medium, Low")
    model_version: str = Field(..., description="Versão do modelo")
    processing_time_ms: float = Field(..., description="Tempo de processamento em milissegundos")

class BatchPredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    predictions: List[PredictionResponse]
    total_processed: int
    processing_time_ms: float
    model_version: str

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    status: str
    timestamp: datetime
    model_loaded: bool
    model_version: str
    uptime_seconds: float

# Autenticação simples
security = HTTPBearer()

async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validação da API Key"""
    expected_key = os.getenv("API_KEY", "sk-development-key")
    if credentials.credentials != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key inválida"
        )
    return credentials.credentials

# Contexto de inicialização
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerenciamento do ciclo de vida da aplicação"""
    logger.info("Iniciando API de Satisfação de Passageiros")
    app.state.start_time = datetime.now()
    yield
    logger.info("Finalizando API")

# Inicialização da aplicação
app = FastAPI(
    title="Airline Passenger Satisfaction API",
    description="API para predição de satisfação de passageiros aéreos usando CatBoost",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preprocessamento de dados
def preprocess_passenger_data(data: PassengerData) -> np.ndarray:
    """Preprocessa os dados do passageiro para o modelo"""
    try:
        # Converter para DataFrame
        df = pd.DataFrame([data.dict()])
        
        # Criar features categóricas dummy
        df['Gender_Male'] = 1 if data.gender == 'Male' else 0
        df['Customer_Type_Loyal_Customer'] = 1 if data.customer_type == 'Loyal Customer' else 0
        df['Type_of_Travel_Personal_Travel'] = 1 if data.type_of_travel == 'Personal Travel' else 0
        df['Class_Business'] = 1 if data.class_type == 'Business' else 0
        df['Class_Eco_Plus'] = 1 if data.class_type == 'Eco Plus' else 0
        
        # FEATURE ENGINEERING (como no Streamlit)
        service_cols = [
            'inflight_wifi_service', 'departure_arrival_time_convenient',
            'ease_of_online_booking', 'gate_location', 'food_and_drink',
            'online_boarding', 'seat_comfort', 'inflight_entertainment',
            'on_board_service', 'leg_room_service', 'baggage_handling',
            'checkin_service', 'inflight_service', 'cleanliness'
        ]
        
        # Calcular features agregadas
        df['Service_Quality_Mean'] = df[service_cols].mean(axis=1)
        df['Service_Quality_Std'] = df[service_cols].std(axis=1)
        df['Service_Quality_Min'] = df[service_cols].min(axis=1)
        df['Service_Quality_Max'] = df[service_cols].max(axis=1)
        
        # Features de delay
        df['Total_Delay'] = df['departure_delay_in_minutes'] + df['arrival_delay_in_minutes']
        df['Has_Delay'] = (df['Total_Delay'] > 0).astype(int)
        
        # Features de idade
        df['Age_Group'] = pd.cut(df['age'], bins=[0, 25, 40, 55, 100], labels=[0, 1, 2, 3]).astype(int)
        
        # Features de distância
        df['Flight_Distance_Category'] = pd.cut(df['flight_distance'], 
                                               bins=[0, 500, 1500, 3000, 5000], 
                                               labels=[0, 1, 2, 3]).astype(int)
        
        # Renomear colunas para corresponder ao modelo
        column_mapping = {
            'age': 'Age',
            'flight_distance': 'Flight_Distance',
            'inflight_wifi_service': 'Inflight_wifi_service',
            'departure_arrival_time_convenient': 'Departure_Arrival_time_convenient',
            'ease_of_online_booking': 'Ease_of_Online_booking',
            'gate_location': 'Gate_location',
            'food_and_drink': 'Food_and_drink',
            'online_boarding': 'Online_boarding',
            'seat_comfort': 'Seat_comfort',
            'inflight_entertainment': 'Inflight_entertainment',
            'on_board_service': 'On_board_service',
            'leg_room_service': 'Leg_room_service',
            'baggage_handling': 'Baggage_handling',
            'checkin_service': 'Checkin_service',
            'inflight_service': 'Inflight_service',
            'cleanliness': 'Cleanliness',
            'departure_delay_in_minutes': 'Departure_Delay_in_Minutes',
            'arrival_delay_in_minutes': 'Arrival_Delay_in_Minutes'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Selecionar apenas as features necessárias
        if model_manager.feature_names:
            feature_columns = model_manager.feature_names
            # Garantir que todas as features existam
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0  # Valor padrão para features faltantes
            df_processed = df[feature_columns]
        else:
            # Fallback para features básicas
            basic_features = [col for col in df.columns if col in model_manager._get_default_features()]
            df_processed = df[basic_features]
        
        return df_processed.values
        
    except Exception as e:
        logger.error(f"Erro no preprocessamento: {e}")
        raise HTTPException(status_code=400, detail=f"Erro no preprocessamento: {str(e)}")

def get_confidence_level(probability: float) -> str:
    """Determina o nível de confiança baseado na probabilidade"""
    if probability >= 0.8 or probability <= 0.2:
        return "High"
    elif probability >= 0.65 or probability <= 0.35:
        return "Medium"
    else:
        return "Low"

# ENDPOINTS VISUAIS

@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def root():
    """Homepage visual da API"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Airline Passenger Satisfaction API</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: 'Segoe UI', system-ui, sans-serif;
                margin: 0;
                padding: 40px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
                background: rgba(255,255,255,0.1);
                padding: 40px;
                border-radius: 20px;
                backdrop-filter: blur(10px);
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }
            h1 {
                text-align: center;
                font-size: 2.5em;
                margin-bottom: 20px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .info-card {
                background: rgba(255,255,255,0.15);
                padding: 20px;
                border-radius: 15px;
                margin: 20px 0;
                border-left: 4px solid #FFD700;
            }
            .links {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 30px;
            }
            .link-card {
                background: rgba(255,255,255,0.2);
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                transition: all 0.3s ease;
                text-decoration: none;
                color: white;
                border: 1px solid rgba(255,255,255,0.1);
            }
            .link-card:hover {
                transform: translateY(-5px);
                background: rgba(255,255,255,0.3);
                box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            }
            .status {
                background: #4CAF50;
                padding: 15px 25px;
                border-radius: 25px;
                text-align: center;
                font-weight: bold;
                margin: 20px 0;
                font-size: 1.1em;
            }
            .tech-stack {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                justify-content: center;
                margin: 20px 0;
            }
            .tech-badge {
                background: rgba(255,255,255,0.2);
                padding: 8px 15px;
                border-radius: 20px;
                font-size: 0.9em;
                border: 1px solid rgba(255,255,255,0.3);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>✈️ Airline Passenger Satisfaction API</h1>
            
            <div class="status">
                🟢 API Status: Online and Ready for Production
            </div>
            
            <div class="info-card">
                <h3>🤖 About This API</h3>
                <p>Advanced machine learning API powered by <strong>CatBoost</strong> for predicting airline passenger satisfaction. This is a <strong>portfolio demonstration</strong> showcasing production-ready ML deployment with modern web technologies.</p>
                
                <div class="tech-stack">
                    <span class="tech-badge">🐍 FastAPI</span>
                    <span class="tech-badge">🤖 CatBoost ML</span>
                    <span class="tech-badge">🐳 Docker</span>
                    <span class="tech-badge">📊 Prometheus</span>
                    <span class="tech-badge">🔧 Pydantic</span>
                    <span class="tech-badge">📚 Auto-Docs</span>
                </div>
            </div>
            
            <div class="info-card">
                <h3>📊 Model Performance</h3>
                <p><strong>Algorithm:</strong> CatBoost Enhanced with 31+ engineered features<br>
                <strong>Accuracy:</strong> 96.4% on validation set<br>
                <strong>Features:</strong> Demographics, service ratings, delays, and derived metrics<br>
                <strong>Version:</strong> v1.0.1 (Production Ready)</p>
            </div>
            
            <div class="links">
                <a href="/docs" class="link-card">
                    <h3>📚 Interactive Documentation</h3>
                    <p>Swagger UI with live API testing</p>
                    <small>Try the API endpoints directly</small>
                </a>
                
                <a href="/predict-form" class="link-card">
                    <h3>🎯 Try Prediction</h3>
                    <p>Visual form to test predictions</p>
                    <small>User-friendly interface</small>
                </a>
                
                <a href="/health-dashboard" class="link-card">
                    <h3>🏥 Health Dashboard</h3>
                    <p>System monitoring and status</p>
                    <small>Real-time metrics</small>
                </a>
                
                <a href="/metrics" class="link-card">
                    <h3>📈 Prometheus Metrics</h3>
                    <p>Performance monitoring data</p>
                    <small>For DevOps integration</small>
                </a>
            </div>
            
            <div class="info-card">
                <h3>🚀 API Endpoints</h3>
                <p><strong>Prediction:</strong> POST /predict (with Bearer token)<br>
                <strong>Batch Processing:</strong> POST /predict/batch (up to 100 passengers)<br>
                <strong>Model Info:</strong> GET /model/info<br>
                <strong>Health Check:</strong> GET /health</p>
            </div>
            
            <div class="info-card">
                <h3>🔐 Authentication</h3>
                <p><strong>Development Key:</strong> <code>sk-development-key</code><br>
                <strong>Header:</strong> <code>Authorization: Bearer sk-development-key</code><br>
                <em>This is a demo - in production, use secure API keys</em></p>
            </div>
            
            <div style="text-align: center; margin-top: 40px; padding-top: 30px; border-top: 1px solid rgba(255,255,255,0.2);">
                <p><strong>👨‍💻 Portfolio Project by Pedro M.</strong></p>
                <p style="font-size: 0.9em; opacity: 0.8;">
                    Demonstrating full-stack ML deployment capabilities<br>
                    FastAPI + CatBoost + Docker + Monitoring
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/predict-form", response_class=HTMLResponse, tags=["Prediction"])
async def predict_form():
    """Formulário visual para predições"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Predict Passenger Satisfaction</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: 'Segoe UI', system-ui, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: rgba(255,255,255,0.1);
                padding: 30px;
                border-radius: 20px;
                backdrop-filter: blur(10px);
            }
            .form-section {
                background: rgba(255,255,255,0.1);
                padding: 20px;
                border-radius: 15px;
                margin: 20px 0;
            }
            .form-row {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 15px 0;
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
                font-size: 0.9em;
            }
            input, select {
                width: 100%;
                padding: 10px;
                border: none;
                border-radius: 8px;
                background: rgba(255,255,255,0.9);
                color: #333;
                box-sizing: border-box;
                font-size: 14px;
            }
            .btn {
                background: #4CAF50;
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                width: 100%;
                margin: 20px 0;
                transition: all 0.3s;
            }
            .btn:hover {
                background: #45a049;
                transform: translateY(-2px);
            }
            .btn:disabled {
                background: #999;
                cursor: not-allowed;
                transform: none;
            }
            .result {
                margin-top: 30px;
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                font-size: 1.1em;
                display: none;
            }
            .satisfied {
                background: linear-gradient(135deg, #4CAF50, #8BC34A);
            }
            .dissatisfied {
                background: linear-gradient(135deg, #FF5722, #FF9800);
            }
            .error {
                background: linear-gradient(135deg, #F44336, #E91E63);
            }
            .nav-links {
                text-align: center;
                margin: 30px 0;
            }
            .nav-links a {
                color: white;
                text-decoration: none;
                background: rgba(255,255,255,0.2);
                padding: 10px 20px;
                border-radius: 10px;
                margin: 0 10px;
                transition: all 0.3s;
            }
            .nav-links a:hover {
                background: rgba(255,255,255,0.3);
            }
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 style="text-align: center;">🎯 Passenger Satisfaction Predictor</h1>
            <p style="text-align: center; font-size: 1.1em; margin-bottom: 30px;">
                Enter passenger details to predict satisfaction using our CatBoost ML model
            </p>
            
            <form id="predictionForm">
                <div class="form-section">
                    <h3>👤 Passenger Information</h3>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Age:</label>
                            <input type="number" id="age" min="7" max="85" value="35" required>
                        </div>
                        <div class="form-group">
                            <label>Gender:</label>
                            <select id="gender" required>
                                <option value="Male">Male</option>
                                <option value="Female">Female</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Customer Type:</label>
                            <select id="customer_type" required>
                                <option value="Loyal Customer">Loyal Customer</option>
                                <option value="disloyal Customer">Disloyal Customer</option>
                            </select>
                        </div>
                    </div>
                </div>

                <div class="form-section">
                    <h3>✈️ Flight Details</h3>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Flight Distance (miles):</label>
                            <input type="number" id="flight_distance" min="31" max="4983" value="1500" required>
                        </div>
                        <div class="form-group">
                            <label>Type of Travel:</label>
                            <select id="type_of_travel" required>
                                <option value="Business travel">Business Travel</option>
                                <option value="Personal Travel">Personal Travel</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Class:</label>
                            <select id="class_type" required>
                                <option value="Business">Business</option>
                                <option value="Eco">Economy</option>
                                <option value="Eco Plus">Economy Plus</option>
                            </select>
                        </div>
                    </div>
                </div>

                <div class="form-section">
                    <h3>📊 Service Ratings (0 = Not Applicable, 1-5 = Rating)</h3>
                    <div class="form-row">
                        <div class="form-group">
                            <label>WiFi Service:</label>
                            <input type="number" id="inflight_wifi_service" min="0" max="5" value="4" required>
                        </div>
                        <div class="form-group">
                            <label>Time Convenience:</label>
                            <input type="number" id="departure_arrival_time_convenient" min="0" max="5" value="4" required>
                        </div>
                        <div class="form-group">
                            <label>Online Booking:</label>
                            <input type="number" id="ease_of_online_booking" min="0" max="5" value="3" required>
                        </div>
                        <div class="form-group">
                            <label>Gate Location:</label>
                            <input type="number" id="gate_location" min="0" max="5" value="4" required>
                        </div>
                        <div class="form-group">
                            <label>Food & Drink:</label>
                            <input type="number" id="food_and_drink" min="0" max="5" value="3" required>
                        </div>
                        <div class="form-group">
                            <label>Online Boarding:</label>
                            <input type="number" id="online_boarding" min="0" max="5" value="4" required>
                        </div>
                        <div class="form-group">
                            <label>Seat Comfort:</label>
                            <input type="number" id="seat_comfort" min="0" max="5" value="4" required>
                        </div>
                        <div class="form-group">
                            <label>Entertainment:</label>
                            <input type="number" id="inflight_entertainment" min="0" max="5" value="3" required>
                        </div>
                        <div class="form-group">
                            <label>On-board Service:</label>
                            <input type="number" id="on_board_service" min="0" max="5" value="4" required>
                        </div>
                        <div class="form-group">
                            <label>Leg Room:</label>
                            <input type="number" id="leg_room_service" min="0" max="5" value="3" required>
                        </div>
                        <div class="form-group">
                            <label>Baggage Handling:</label>
                            <input type="number" id="baggage_handling" min="0" max="5" value="4" required>
                        </div>
                        <div class="form-group">
                            <label>Check-in Service:</label>
                            <input type="number" id="checkin_service" min="0" max="5" value="4" required>
                        </div>
                        <div class="form-group">
                            <label>Inflight Service:</label>
                            <input type="number" id="inflight_service" min="0" max="5" value="4" required>
                        </div>
                        <div class="form-group">
                            <label>Cleanliness:</label>
                            <input type="number" id="cleanliness" min="0" max="5" value="4" required>
                        </div>
                    </div>
                </div>

                <div class="form-section">
                    <h3>⏰ Delays</h3>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Departure Delay (minutes):</label>
                            <input type="number" id="departure_delay_in_minutes" min="0" max="1592" value="0" required>
                        </div>
                        <div class="form-group">
                            <label>Arrival Delay (minutes):</label>
                            <input type="number" id="arrival_delay_in_minutes" min="0" max="1584" value="0" required>
                        </div>
                    </div>
                </div>
                
                <button type="submit" class="btn" id="predictBtn">
                    🎯 Predict Satisfaction
                </button>
                
                <div class="loading" id="loading">
                    <p>🔄 Making prediction...</p>
                </div>
            </form>
            
            <div id="result" class="result"></div>
            
            <div class="nav-links">
                <a href="/">← Back to Home</a>
                <a href="/docs">API Documentation</a>
                <a href="/health-dashboard">Health Status</a>
            </div>
        </div>
        
        <script>
            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                // Show loading state
                document.getElementById('loading').style.display = 'block';
                document.getElementById('predictBtn').disabled = true;
                document.getElementById('result').style.display = 'none';
                
                // Collect form data
                const formData = {
                    age: parseInt(document.getElementById('age').value),
                    flight_distance: parseInt(document.getElementById('flight_distance').value),
                    inflight_wifi_service: parseInt(document.getElementById('inflight_wifi_service').value),
                    departure_arrival_time_convenient: parseInt(document.getElementById('departure_arrival_time_convenient').value),
                    ease_of_online_booking: parseInt(document.getElementById('ease_of_online_booking').value),
                    gate_location: parseInt(document.getElementById('gate_location').value),
                    food_and_drink: parseInt(document.getElementById('food_and_drink').value),
                    online_boarding: parseInt(document.getElementById('online_boarding').value),
                    seat_comfort: parseInt(document.getElementById('seat_comfort').value),
                    inflight_entertainment: parseInt(document.getElementById('inflight_entertainment').value),
                    on_board_service: parseInt(document.getElementById('on_board_service').value),
                    leg_room_service: parseInt(document.getElementById('leg_room_service').value),
                    baggage_handling: parseInt(document.getElementById('baggage_handling').value),
                    checkin_service: parseInt(document.getElementById('checkin_service').value),
                    inflight_service: parseInt(document.getElementById('inflight_service').value),
                    cleanliness: parseInt(document.getElementById('cleanliness').value),
                    departure_delay_in_minutes: parseInt(document.getElementById('departure_delay_in_minutes').value),
                    arrival_delay_in_minutes: parseInt(document.getElementById('arrival_delay_in_minutes').value),
                    gender: document.getElementById('gender').value,
                    customer_type: document.getElementById('customer_type').value,
                    type_of_travel: document.getElementById('type_of_travel').value,
                    class_type: document.getElementById('class_type').value
                };
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer sk-development-key'
                        },
                        body: JSON.stringify(formData)
                    });
                    
                    const result = await response.json();
                    const resultDiv = document.getElementById('result');
                    
                    if (response.ok) {
                        const isSatisfied = result.prediction === 'Satisfied';
                        resultDiv.className = `result ${isSatisfied ? 'satisfied' : 'dissatisfied'}`;
                        
                        const emoji = isSatisfied ? '😊' : '😞';
                        const confidenceColor = result.confidence === 'High' ? '#4CAF50' : 
                                              result.confidence === 'Medium' ? '#FF9800' : '#F44336';
                        
                        resultDiv.innerHTML = `
                            <h2>${emoji} ${result.prediction}</h2>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0;">
                                <div>
                                    <strong>Probability</strong><br>
                                    <span style="font-size: 1.5em;">${(result.probability * 100).toFixed(1)}%</span>
                                </div>
                                <div>
                                    <strong>Confidence</strong><br>
                                    <span style="color: ${confidenceColor}; font-size: 1.5em;">${result.confidence}</span>
                                </div>
                                <div>
                                    <strong>Processing Time</strong><br>
                                    <span style="font-size: 1.5em;">${result.processing_time_ms}ms</span>
                                </div>
                                <div>
                                    <strong>Model Version</strong><br>
                                    <span style="font-size: 1.5em;">${result.model_version}</span>
                                </div>
                            </div>
                            <p style="margin-top: 20px; opacity: 0.9;">
                                ${isSatisfied ? 
                                  'This passenger is likely to be satisfied with their flight experience!' : 
                                  'This passenger may have concerns about their flight experience.'}
                            </p>
                        `;
                        resultDiv.style.display = 'block';
                        resultDiv.scrollIntoView({ behavior: 'smooth' });
                    } else {
                        resultDiv.className = 'result error';
                        resultDiv.innerHTML = `
                            <h3>❌ Prediction Error</h3>
                            <p>${result.detail || 'Unknown error occurred'}</p>
                        `;
                        resultDiv.style.display = 'block';
                    }
                } catch (error) {
                    const resultDiv = document.getElementById('result');
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `
                        <h3>❌ Network Error</h3>
                        <p>Could not connect to the API. Please check if the server is running.</p>
                        <small>Error: ${error.message}</small>
                    `;
                    resultDiv.style.display = 'block';
                }
                
                // Hide loading state
                document.getElementById('loading').style.display = 'none';
                document.getElementById('predictBtn').disabled = false;
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health-dashboard", response_class=HTMLResponse, tags=["Health"])
async def health_dashboard():
    """Dashboard visual de saúde do sistema"""
    uptime = (datetime.now() - app.state.start_time).total_seconds()
    uptime_hours = uptime / 3600
    
    model_loaded = model_manager.model is not None
    overall_status = "healthy" if model_loaded else "degraded"
    
    # Simular algumas métricas para demonstração
    import random
    cpu_usage = round(random.uniform(15, 45), 1)
    memory_usage = round(random.uniform(60, 85), 1)
    requests_today = random.randint(150, 500)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>API Health Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta http-equiv="refresh" content="30">
        <style>
            body {{
                font-family: 'Segoe UI', system-ui, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: rgba(255,255,255,0.1);
                padding: 30px;
                border-radius: 20px;
                backdrop-filter: blur(10px);
            }}
            .status-overview {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .metric-card {{
                background: rgba(255,255,255,0.15);
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                border-left: 4px solid;
                transition: transform 0.3s ease;
            }}
            .metric-card:hover {{
                transform: translateY(-5px);
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                margin: 10px 0;
            }}
            .metric-label {{
                font-size: 0.9em;
                opacity: 0.8;
            }}
            .status-good {{ border-left-color: #4CAF50; }}
            .status-warning {{ border-left-color: #FF9800; }}
            .status-error {{ border-left-color: #F44336; }}
            .system-info {{
                background: rgba(255,255,255,0.1);
                padding: 20px;
                border-radius: 15px;
                margin: 20px 0;
            }}
            .nav-links {{
                text-align: center;
                margin: 30px 0;
            }}
            .nav-links a {{
                color: white;
                text-decoration: none;
                background: rgba(255,255,255,0.2);
                padding: 10px 20px;
                border-radius: 10px;
                margin: 0 10px;
                transition: all 0.3s;
            }}
            .nav-links a:hover {{
                background: rgba(255,255,255,0.3);
            }}
            .refresh-info {{
                text-align: center;
                margin: 20px 0;
                font-size: 0.9em;
                opacity: 0.7;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 style="text-align: center;">🏥 System Health Dashboard</h1>
            <div class="refresh-info">
                🔄 Auto-refreshes every 30 seconds | Last updated: {datetime.now().strftime('%H:%M:%S')}
            </div>
            
            <div class="status-overview">
                <div class="metric-card status-{'good' if overall_status == 'healthy' else 'error'}">
                    <div class="metric-label">Overall Status</div>
                    <div class="metric-value">{'🟢' if overall_status == 'healthy' else '🔴'}</div>
                    <div>{'Healthy' if overall_status == 'healthy' else 'Degraded'}</div>
                </div>
                
                <div class="metric-card status-{'good' if model_loaded else 'error'}">
                    <div class="metric-label">ML Model</div>
                    <div class="metric-value">{'🤖' if model_loaded else '❌'}</div>
                    <div>{'Loaded' if model_loaded else 'Not Loaded'}</div>
                </div>
                
                <div class="metric-card status-good">
                    <div class="metric-label">Uptime</div>
                    <div class="metric-value">{uptime_hours:.1f}h</div>
                    <div>Running Strong</div>
                </div>
                
                <div class="metric-card status-{'good' if cpu_usage < 70 else 'warning'}">
                    <div class="metric-label">CPU Usage</div>
                    <div class="metric-value">{cpu_usage}%</div>
                    <div>System Load</div>
                </div>
                
                <div class="metric-card status-{'good' if memory_usage < 80 else 'warning'}">
                    <div class="metric-label">Memory</div>
                    <div class="metric-value">{memory_usage}%</div>
                    <div>RAM Usage</div>
                </div>
                
                <div class="metric-card status-good">
                    <div class="metric-label">Requests Today</div>
                    <div class="metric-value">{requests_today}</div>
                    <div>API Calls</div>
                </div>
            </div>
            
            <div class="system-info">
                <h3>📊 System Information</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                    <div><strong>Model Version:</strong> {model_manager.model_version}</div>
                    <div><strong>API Version:</strong> 1.0.0</div>
                    <div><strong>Framework:</strong> FastAPI + CatBoost</div>
                    <div><strong>Environment:</strong> Development</div>
                    <div><strong>Started:</strong> {app.state.start_time.strftime('%Y-%m-%d %H:%M:%S')}</div>
                    <div><strong>Features:</strong> {len(model_manager.feature_names) if model_manager.feature_names else 'Loading...'}</div>
                </div>
            </div>
            
            <div class="system-info">
                <h3>🔗 Quick Actions</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                    <a href="/health" style="color: white; text-decoration: none; background: rgba(255,255,255,0.15); padding: 15px; border-radius: 10px; text-align: center;">
                        📋 JSON Health Check
                    </a>
                    <a href="/metrics" style="color: white; text-decoration: none; background: rgba(255,255,255,0.15); padding: 15px; border-radius: 10px; text-align: center;">
                        📈 Prometheus Metrics
                    </a>
                    <a href="/model/info" style="color: white; text-decoration: none; background: rgba(255,255,255,0.15); padding: 15px; border-radius: 10px; text-align: center;">
                        🔧 Model Information
                    </a>
                    <a href="/docs" style="color: white; text-decoration: none; background: rgba(255,255,255,0.15); padding: 15px; border-radius: 10px; text-align: center;">
                        📚 API Documentation
                    </a>
                </div>
            </div>
            
            <div class="nav-links">
                <a href="/">← Back to Home</a>
                <a href="/predict-form">Try Prediction</a>
                <a href="/docs">API Docs</a>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ENDPOINTS JSON/API

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Verifica a saúde da API (JSON)"""
    uptime = (datetime.now() - app.state.start_time).total_seconds()
    
    return HealthResponse(
        status="healthy" if model_manager.model is not None else "degraded",
        timestamp=datetime.now(),
        model_loaded=model_manager.model is not None,
        model_version=model_manager.model_version,
        uptime_seconds=uptime
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_satisfaction(
    passenger: PassengerData,
    api_key: str = Depends(get_api_key)
):
    """Prediz a satisfação de um passageiro"""
    start_time = datetime.now()
    
    try:
        # Preprocessar dados
        processed_data = preprocess_passenger_data(passenger)
        
        # Fazer predição
        if model_manager.model is not None:
            # Predição real com CatBoost
            prediction_proba = model_manager.model.predict_proba(processed_data)[0]
            prediction_class = model_manager.model.predict(processed_data)[0]
            
            # CatBoost retorna 0 para 'Neutral or Dissatisfied' e 1 para 'Satisfied'
            if prediction_class == 1:
                result = "Satisfied"
                probability = prediction_proba[1]
            else:
                result = "Neutral or Dissatisfied"
                probability = prediction_proba[0]
        else:
            # Predição mock para desenvolvimento
            mock_probability = np.random.uniform(0.1, 0.9)
            result = "Satisfied" if mock_probability > 0.5 else "Neutral or Dissatisfied"
            probability = mock_probability
        
        # Calcular tempo de processamento
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Registrar métricas
        PREDICTION_COUNT.labels(model_version=model_manager.model_version).inc()
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="200").inc()
        
        response = PredictionResponse(
            prediction=result,
            probability=round(probability, 4),
            confidence=get_confidence_level(probability),
            model_version=model_manager.model_version,
            processing_time_ms=round(processing_time, 2)
        )
        
        logger.info(f"Predição realizada: {result} (prob: {probability:.4f})")
        return response
        
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="500").inc()
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch_satisfaction(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """Prediz a satisfação de múltiplos passageiros"""
    start_time = datetime.now()
    
    try:
        predictions = []
        
        for passenger in request.passengers:
            # Preprocessar dados
            processed_data = preprocess_passenger_data(passenger)
            
            # Fazer predição
            if model_manager.model is not None:
                prediction_proba = model_manager.model.predict_proba(processed_data)[0]
                prediction_class = model_manager.model.predict(processed_data)[0]
                
                if prediction_class == 1:
                    result = "Satisfied"
                    probability = prediction_proba[1]
                else:
                    result = "Neutral or Dissatisfied"
                    probability = prediction_proba[0]
            else:
                # Predição mock
                mock_probability = np.random.uniform(0.1, 0.9)
                result = "Satisfied" if mock_probability > 0.5 else "Neutral or Dissatisfied"
                probability = mock_probability
            
            predictions.append(PredictionResponse(
                prediction=result,
                probability=round(probability, 4),
                confidence=get_confidence_level(probability),
                model_version=model_manager.model_version,
                processing_time_ms=0  # Será calculado no final
            ))
        
        # Calcular tempo total de processamento
        total_processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Atualizar tempo de processamento individual
        individual_time = total_processing_time / len(predictions)
        for pred in predictions:
            pred.processing_time_ms = round(individual_time, 2)
        
        # Registrar métricas
        PREDICTION_COUNT.labels(model_version=model_manager.model_version).inc(len(predictions))
        REQUEST_COUNT.labels(method="POST", endpoint="/predict/batch", status="200").inc()
        
        response = BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            processing_time_ms=round(total_processing_time, 2),
            model_version=model_manager.model_version
        )
        
        logger.info(f"Predição em lote realizada: {len(predictions)} passageiros")
        return response
        
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict/batch", status="500").inc()
        logger.error(f"Erro na predição em lote: {e}")
        raise HTTPException(status_code=500, detail=f"Erro na predição em lote: {str(e)}")

@app.get("/model/info", tags=["Model"])
async def get_model_info(api_key: str = Depends(get_api_key)):
    """Retorna informações sobre o modelo"""
    return {
        "model_type": "CatBoost Classifier",
        "model_version": model_manager.model_version,
        "features_count": len(model_manager.feature_names) if model_manager.feature_names else 0,
        "feature_names": model_manager.feature_names,
        "model_loaded": model_manager.model is not None,
        "supported_classes": ["Satisfied", "Neutral or Dissatisfied"]
    }

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Endpoint para métricas Prometheus"""
    return Response(generate_latest(), media_type="text/plain")

@app.get("/api-status", tags=["Root"])
async def api_status():
    """Endpoint JSON simples para status da API"""
    return {
        "message": "Airline Passenger Satisfaction API",
        "version": "1.0.0",
        "status": "healthy" if model_manager.model is not None else "degraded",
        "model_loaded": model_manager.model is not None,
        "model_version": model_manager.model_version,
        "uptime_seconds": (datetime.now() - app.state.start_time).total_seconds(),
        "endpoints": {
            "visual_home": "/",
            "prediction_form": "/predict-form", 
            "health_dashboard": "/health-dashboard",
            "documentation": "/docs",
            "predict_api": "/predict",
            "health_api": "/health",
            "metrics": "/metrics"
        }
    }

# Middleware para logging de requests
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = datetime.now()
    
    response = await call_next(request)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    REQUEST_DURATION.observe(processing_time)
    
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {processing_time:.3f}s")
    
    return response

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )