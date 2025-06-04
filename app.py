"""
FastAPI application for Airline Passenger Satisfaction Prediction
Production-ready API with monitoring, caching, and error handling
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import pickle
import json
import logging
import time
import os
import io
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import uvicorn
from pydantic import BaseModel, Field, validator
import redis
import asyncio
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import aiofiles
from functools import lru_cache
from fastapi import Response


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metrics
prediction_counter = Counter('predictions_total', 'Total number of predictions')
prediction_histogram = Histogram('prediction_duration_seconds', 'Prediction duration')
active_requests = Gauge('active_requests', 'Number of active requests')
error_counter = Counter('prediction_errors_total', 'Total number of prediction errors')

# Load environment variables
MODEL_PATH = os.getenv('MODEL_PATH', 'models/lightgbm_model.pkl')
SCALER_PATH = os.getenv('SCALER_PATH', 'models/scaler.pkl')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
API_KEY = os.getenv('API_KEY', 'your-secret-api-key')
CACHE_TTL = int(os.getenv('CACHE_TTL', '3600'))  # 1 hour

# Security
security = HTTPBearer()

class ModelManager:
    """Manage model loading and caching"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_version = None
        self.load_time = None
    
    def load_model(self):
        """Load model and scaler"""
        try:
            logger.info("Loading model...")
            
            # Load model
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load model metadata
            with open('models/model_info.json', 'r') as f:
                artifacts = json.load(f)
                self.feature_names = artifacts.get('selected_features', [])
                self.model_version = artifacts.get('model_version', '1.0.0')
            
            self.load_time = datetime.now()
            logger.info(f"Model loaded successfully. Version: {self.model_version}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

# Initialize model manager
model_manager = ModelManager()

# Cache manager
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    cache_available = True
except:
    logger.warning("Redis not available. Caching disabled.")
    cache_available = False
    redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting up...")
    model_manager.load_model()
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if redis_client:
        redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="Airline Satisfaction Prediction API",
    description="ML API for predicting passenger satisfaction with advanced features",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class PassengerFeatures(BaseModel):
    """Input features for a passenger"""
    gender: str = Field(..., description="Gender: Male or Female")
    customer_type: str = Field(..., description="Customer Type: Loyal Customer or Disloyal Customer")
    age: int = Field(..., ge=0, le=100, description="Passenger age")
    type_of_travel: str = Field(..., description="Type of Travel: Business travel or Personal Travel")
    travel_class: str = Field(..., alias="class", description="Class: Eco, Eco Plus, or Business")
    flight_distance: int = Field(..., ge=0, description="Flight distance in miles")
    inflight_wifi_service: int = Field(..., ge=1, le=5, description="WiFi service rating")
    departure_arrival_time_convenient: int = Field(..., ge=1, le=5)
    ease_of_online_booking: int = Field(..., ge=1, le=5)
    gate_location: int = Field(..., ge=1, le=5)
    food_and_drink: int = Field(..., ge=1, le=5)
    online_boarding: int = Field(..., ge=1, le=5)
    seat_comfort: int = Field(..., ge=1, le=5)
    inflight_entertainment: int = Field(..., ge=1, le=5)
    on_board_service: int = Field(..., ge=1, le=5)
    leg_room_service: int = Field(..., ge=1, le=5)
    baggage_handling: int = Field(..., ge=1, le=5)
    checkin_service: int = Field(..., ge=1, le=5)
    inflight_service: int = Field(..., ge=1, le=5)
    cleanliness: int = Field(..., ge=1, le=5)
    departure_delay_in_minutes: int = Field(..., ge=0)
    arrival_delay_in_minutes: int = Field(..., ge=0,)
    
    @validator('gender')
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('Gender must be Male or Female')
        return v
    
    @validator('customer_type')
    def validate_customer_type(cls, v):
        if v not in ['Loyal Customer', 'Disloyal Customer']:
            raise ValueError('Invalid customer type')
        return v
    
    @validator('type_of_travel')
    def validate_travel_type(cls, v):
        if v not in ['Business travel', 'Personal Travel']:
            raise ValueError('Invalid travel type')
        return v
    
    @validator('travel_class')
    def validate_class(cls, v):
        if v not in ['Eco', 'Eco Plus', 'Business']:
            raise ValueError('Invalid travel class')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "gender": "Male",
                "customer_type": "Loyal Customer",
                "age": 35,
                "type_of_travel": "Business travel",
                "class": "Business",
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

class PredictionResponse(BaseModel):
    """Prediction response model"""
    request_id: str
    prediction: str
    probability: float
    confidence_level: str
    processing_time: float
    model_version: str
    timestamp: str
    explanation: Optional[Dict[str, float]] = None

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    request_id: str
    predictions: List[PredictionResponse]
    total_processed: int
    processing_time: float
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: str
    uptime: str
    last_prediction: Optional[str]
    cache_status: str
    timestamp: str

# Dependency for API key validation
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key"""
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    return credentials.credentials

# Helper functions
def generate_request_id():
    """Generate unique request ID"""
    return f"req_{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"

def get_cache_key(features: dict) -> str:
    """Generate cache key from features"""
    return f"prediction:{hash(json.dumps(features, sort_keys=True))}"

async def get_cached_prediction(cache_key: str) -> Optional[dict]:
    """Get prediction from cache"""
    if not cache_available:
        return None
    
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.error(f"Cache error: {e}")
    
    return None

async def cache_prediction(cache_key: str, prediction: dict):
    """Cache prediction result"""
    if not cache_available:
        return
    
    try:
        redis_client.setex(
            cache_key,
            CACHE_TTL,
            json.dumps(prediction)
        )
    except Exception as e:
        logger.error(f"Cache error: {e}")

def preprocess_features(passenger: PassengerFeatures) -> pd.DataFrame:
    """Preprocess passenger features for model input"""
    # Convert to dataframe
    data = passenger.dict()
    
    # Rename class field
    if 'class' in data:
        data['Class'] = data.pop('class')
    
    # Create dataframe
    df = pd.DataFrame([data])
    
    # Apply feature engineering (simplified version)
    # In production, this would call the full feature engineering pipeline
    service_cols = ['inflight_wifi_service', 'seat_comfort', 'inflight_entertainment',
                   'cleanliness', 'food_and_drink', 'online_boarding']
    
    df['avg_service_rating'] = df[service_cols].mean(axis=1)
    df['total_delay'] = df['departure_delay_in_minutes'] + df['arrival_delay_in_minutes']
    
    # Encode categorical variables
    encoding_map = {
        'gender': {'Male': 0, 'Female': 1},
        'customer_type': {'Loyal Customer': 1, 'Disloyal Customer': 0},
        'type_of_travel': {'Business travel': 1, 'Personal Travel': 0},
        'Class': {'Eco': 0, 'Eco Plus': 1, 'Business': 2}
    }
    
    for col, mapping in encoding_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    return df

def get_confidence_level(probability: float) -> str:
    """Determine confidence level from probability"""
    if probability > 0.9 or probability < 0.1:
        return "Very High"
    elif probability > 0.75 or probability < 0.25:
        return "High"
    elif probability > 0.6 or probability < 0.4:
        return "Medium"
    else:
        return "Low"

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    html_content = """
    <html>
        <head>
            <title>Airline Satisfaction API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                .info { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                a { color: #007bff; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1>✈️ Airline Satisfaction Prediction API</h1>
            <div class="info">
                <p><strong>Version:</strong> 2.0.0</p>
                <p><strong>Documentation:</strong> <a href="/docs">Interactive API Docs</a></p>
                <p><strong>Alternative Docs:</strong> <a href="/redoc">ReDoc</a></p>
                <p><strong>Health Check:</strong> <a href="/health">System Status</a></p>
                <p><strong>Metrics:</strong> <a href="/metrics">Prometheus Metrics</a></p>
            </div>
        </body>
    </html>
    """
    return html_content

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = datetime.now() - model_manager.load_time if model_manager.load_time else timedelta(0)
    
    return HealthResponse(
        status="healthy" if model_manager.model else "unhealthy",
        model_loaded=model_manager.model is not None,
        model_version=model_manager.model_version or "unknown",
        uptime=str(uptime),
        last_prediction=None,  # Could track this
        cache_status="connected" if cache_available else "disabled",
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    passenger: PassengerFeatures,
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """Make a single prediction"""
    request_id = generate_request_id()
    start_time = time.time()
    
    # Track metrics
    active_requests.inc()
    prediction_counter.inc()
    
    try:
        # Check cache
        cache_key = get_cache_key(passenger.dict())
        cached_result = await get_cached_prediction(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit for request {request_id}")
            cached_result['request_id'] = request_id
            cached_result['timestamp'] = datetime.now().isoformat()
            active_requests.dec()
            return PredictionResponse(**cached_result)
        
        # Preprocess features
        features_df = preprocess_features(passenger)
        
        # Select model features
        model_features = [col for col in model_manager.feature_names if col in features_df.columns]
        X = features_df[model_features]
        
        # Scale features
        X_scaled = model_manager.scaler.transform(X)
        
        # Make prediction
        with prediction_histogram.time():
            prediction = model_manager.model.predict(X_scaled)[0]
            probability = model_manager.model.predict_proba(X_scaled)[0, 1]
        
        # Prepare response
        processing_time = time.time() - start_time
        
        result = {
            "request_id": request_id,
            "prediction": "satisfied" if prediction == 1 else "unsatisfied",
            "probability": float(probability),
            "confidence_level": get_confidence_level(probability),
            "processing_time": processing_time,
            "model_version": model_manager.model_version,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache result
        await cache_prediction(cache_key, result)
        
        logger.info(f"Prediction completed: {request_id} - {result['prediction']}")
        
        active_requests.dec()
        return PredictionResponse(**result)
        
    except Exception as e:
        active_requests.dec()
        error_counter.inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """Make batch predictions from CSV file"""
    request_id = generate_request_id()
    start_time = time.time()
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        logger.info(f"Batch prediction for {len(df)} passengers")
        
        # Process each passenger
        predictions = []
        
        for idx, row in df.iterrows():
            try:
                # Convert row to PassengerFeatures
                passenger = PassengerFeatures(**row.to_dict())
                
                # Make prediction (reuse single prediction logic)
                pred_response = await predict(passenger, None, api_key)
                predictions.append(pred_response)
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                # Add error prediction
                predictions.append(PredictionResponse(
                    request_id=f"{request_id}_{idx}",
                    prediction="error",
                    probability=0.0,
                    confidence_level="None",
                    processing_time=0.0,
                    model_version=model_manager.model_version,
                    timestamp=datetime.now().isoformat()
                ))
        
        processing_time = time.time() - start_time
        
        return BatchPredictionResponse(
            request_id=request_id,
            predictions=predictions,
            total_processed=len(predictions),
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

@app.post("/feedback")
async def feedback(
    request_id: str,
    actual_satisfaction: bool,
    api_key: str = Depends(verify_api_key)
):
    """Collect feedback on predictions"""
    # In production, this would store feedback for model monitoring
    logger.info(f"Feedback received for {request_id}: {actual_satisfaction}")
    
    return {"status": "feedback recorded", "request_id": request_id}

@app.get("/model/info")
async def model_info(api_key: str = Depends(verify_api_key)):
    """Get model information"""
    if not model_manager.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = {
        "model_type": type(model_manager.model).__name__,
        "model_version": model_manager.model_version,
        "feature_count": len(model_manager.feature_names),
        "features": model_manager.feature_names,
        "load_time": model_manager.load_time.isoformat() if model_manager.load_time else None
    }
    
    # Add model-specific info
    if hasattr(model_manager.model, 'feature_importances_'):
        importances = model_manager.model.feature_importances_
        top_features = sorted(
            zip(model_manager.feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        info["top_features"] = [{"name": f[0], "importance": f[1]} for f in top_features]
    
    return info

@app.post("/model/reload")
async def reload_model(api_key: str = Depends(verify_api_key)):
    """Reload the model"""
    try:
        model_manager.load_model()
        return {"status": "success", "message": "Model reloaded successfully"}
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )