"""
FastAPI application for Airline Passenger Satisfaction Prediction
Production-ready API with all features implemented
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
import pickle
import json
import logging
import time
import os
import io
from datetime import datetime
from typing import List, Dict, Any, Optional
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Airline Satisfaction Prediction API",
    description="ML API for predicting passenger satisfaction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model artifacts
MODEL_ARTIFACTS = {}

# Pydantic models
class PassengerFeatures(BaseModel):
    """Input model for passenger features"""
    gender: str = Field(..., description="Gender: Male or Female")
    customer_type: str = Field(..., description="Customer Type: Loyal Customer or Disloyal Customer")
    age: int = Field(..., ge=0, le=100, description="Passenger age")
    type_of_travel: str = Field(..., description="Type of Travel: Business travel or Personal Travel")
    travel_class: str = Field(..., description="Class: Eco, Eco Plus, or Business")
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
    departure_delay_in_minutes: int = Field(..., ge=0,)
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

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: str
    probability: float
    confidence_level: str
    processing_time: float
    model_version: str
    timestamp: str

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]
    total_processed: int
    successful: int
    failed: int
    processing_time: float
    timestamp: str

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    model_version: str
    api_version: str
    timestamp: str

# Helper functions
def load_model_artifacts():
    """Load all model artifacts"""
    global MODEL_ARTIFACTS
    
    try:
        logger.info("Loading model artifacts...")
        
        # Load model
        with open('models/lightgbm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load encoders
        with open('models/label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        # Load model info
        with open('models/model_info.json', 'r') as f:
            model_info = json.load(f)
        
        # Load fill values
        with open('models/fill_values.json', 'r') as f:
            fill_values = json.load(f)
        
        MODEL_ARTIFACTS = {
            'model': model,
            'scaler': scaler,
            'encoders': encoders,
            'model_info': model_info,
            'fill_values': fill_values
        }
        
        logger.info("Model artifacts loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        return False

def create_features(df):
    """Apply feature engineering to dataframe"""
    df_new = df.copy()
    
    # Service features
    service_cols = [col for col in df.columns if any(
        keyword in col.lower() for keyword in ['service', 'comfort', 'cleanliness', 'entertainment', 'food']
    )]
    
    if service_cols:
        df_new['avg_service_rating'] = df[service_cols].mean(axis=1)
        df_new['min_service_rating'] = df[service_cols].min(axis=1)
        df_new['max_service_rating'] = df[service_cols].max(axis=1)
        df_new['service_rating_std'] = df[service_cols].std(axis=1)
        df_new['low_rating_count'] = (df[service_cols] <= 2).sum(axis=1)
        df_new['high_rating_count'] = (df[service_cols] >= 4).sum(axis=1)
    
    # Delay features
    if 'Departure Delay in Minutes' in df.columns and 'Arrival Delay in Minutes' in df.columns:
        df_new['total_delay'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes']
        df_new['delay_difference'] = df['Arrival Delay in Minutes'] - df['Departure Delay in Minutes']
        df_new['is_delayed'] = (df_new['total_delay'] > 15).astype(int)
        df_new['severe_delay'] = (df_new['total_delay'] > 60).astype(int)
    
    # Age features
    if 'Age' in df.columns:
        df_new['age_squared'] = df['Age'] ** 2
        df_new['is_senior'] = (df['Age'] >= 60).astype(int)
        df_new['is_young'] = (df['Age'] <= 25).astype(int)
        df_new['age_group'] = pd.cut(df['Age'], bins=[0, 25, 40, 60, 100], 
                                     labels=['Young', 'Adult', 'Middle', 'Senior'])
    
    # Distance features
    if 'Flight Distance' in df.columns:
        df_new['log_distance'] = np.log1p(df['Flight Distance'])
        df_new['is_long_flight'] = (df['Flight Distance'] > 1000).astype(int)
    
        unique_distances = df['Flight Distance'].nunique()
        if unique_distances >= 4:
            df_new['distance_category'] = pd.qcut(
                df['Flight Distance'], q=4,
                labels=['Short', 'Medium', 'Long', 'Very Long'],
                duplicates='drop'
                )
    else:
        # Quando não há variabilidade suficiente
        df_new['distance_category'] = 'Unknown'
    
    return df_new

def preprocess_data(passenger_dict):
    """Preprocess passenger data for prediction"""
    # Convert to DataFrame
    df = pd.DataFrame([passenger_dict])
    
    # Rename columns to match training data
    column_mapping = {
        'gender': 'Gender',
        'customer_type': 'Customer Type',
        'age': 'Age',
        'type_of_travel': 'Type of Travel',
        'travel_class': 'Class',
        'flight_distance': 'Flight Distance',
        'inflight_wifi_service': 'Inflight wifi service',
        'departure_arrival_time_convenient': 'Departure/Arrival time convenient',
        'ease_of_online_booking': 'Ease of Online booking',
        'gate_location': 'Gate location',
        'food_and_drink': 'Food and drink',
        'online_boarding': 'Online boarding',
        'seat_comfort': 'Seat comfort',
        'inflight_entertainment': 'Inflight entertainment',
        'on_board_service': 'On-board service',
        'leg_room_service': 'Leg room service',
        'baggage_handling': 'Baggage handling',
        'checkin_service': 'Checkin service',
        'inflight_service': 'Inflight service',
        'cleanliness': 'Cleanliness',
        'departure_delay_in_minutes': 'Departure Delay in Minutes',
        'arrival_delay_in_minutes': 'Arrival Delay in Minutes'
    }
    
    df.rename(columns=column_mapping, inplace=True)
    
    # Apply feature engineering
    df = create_features(df)
    
    # Get feature columns from model info
    feature_cols = MODEL_ARTIFACTS['model_info']['feature_columns']
    
    # Add missing columns with default values
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Select only required features
    df = df[feature_cols]
    
    # Encode categorical variables
    for col, encoder in MODEL_ARTIFACTS['encoders'].items():
        if col in df.columns:
            df[col] = df[col].map(lambda x: encoder.transform([x])[0] 
                                 if pd.notna(x) and x in encoder.classes_ else -1)
    
    # Handle missing values
    for col, fill_value in MODEL_ARTIFACTS['fill_values'].items():
        if col in df.columns:
            df[col].fillna(fill_value, inplace=True)
    
    # Fill any remaining NaN with 0
    df.fillna(0, inplace=True)
    
    # Scale features
    df_scaled = MODEL_ARTIFACTS['scaler'].transform(df)
    
    return df_scaled

def get_confidence_level(probability):
    """Determine confidence level from probability"""
    if probability > 0.9 or probability < 0.1:
        return "Very High"
    elif probability > 0.75 or probability < 0.25:
        return "High"
    elif probability > 0.6 or probability < 0.4:
        return "Medium"
    else:
        return "Low"

# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load model artifacts on startup"""
    success = load_model_artifacts()
    if not success:
        logger.error("Failed to load model artifacts on startup")

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    html_content = """
    <html>
        <head>
            <title>Airline Satisfaction API</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 40px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                h1 { 
                    color: #667eea;
                    text-align: center;
                }
                .info { 
                    background-color: #f0f0f0; 
                    padding: 20px; 
                    border-radius: 5px;
                    margin: 20px 0;
                }
                a { 
                    color: #667eea; 
                    text-decoration: none;
                    font-weight: bold;
                }
                a:hover { 
                    text-decoration: underline; 
                }
                .endpoint {
                    background-color: #e8f4f8;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                    border-left: 4px solid #667eea;
                }
                code {
                    background-color: #f4f4f4;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: monospace;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>✈️ Airline Satisfaction Prediction API</h1>
                <div class="info">
                    <p><strong>Version:</strong> 1.0.0</p>
                    <p><strong>Description:</strong> ML-powered API for predicting passenger satisfaction</p>
                    <p><strong>Model:</strong> lightgbm with 96.5% accuracy</p>
                </div>
                
                <h2>📚 Documentation</h2>
                <ul>
                    <li><a href="/docs">Interactive API Documentation (Swagger UI)</a></li>
                    <li><a href="/redoc">Alternative API Documentation (ReDoc)</a></li>
                </ul>
                
                <h2>🔌 Available Endpoints</h2>
                
                <div class="endpoint">
                    <strong>GET /health</strong>
                    <p>Check API and model health status</p>
                </div>
                
                <div class="endpoint">
                    <strong>POST /predict</strong>
                    <p>Make a single passenger satisfaction prediction</p>
                    <p>Request body: <code>PassengerFeatures</code> object</p>
                </div>
                
                <div class="endpoint">
                    <strong>POST /batch_predict</strong>
                    <p>Process multiple predictions from a CSV file</p>
                    <p>Request: <code>multipart/form-data</code> with CSV file</p>
                </div>
                
                <div class="endpoint">
                    <strong>GET /model/info</strong>
                    <p>Get information about the loaded model</p>
                </div>
                
                <h2>🚀 Quick Start</h2>
                <p>Try the prediction endpoint with the example data in the <a href="/docs">interactive documentation</a>.</p>
                
                <h2>👨‍💻 Developer</h2>
                <p>Created by <strong>Pedro M.</strong> | 
                   <a href="https://github.com/Pedrom2002">GitHub</a> | 
                </p>
            </div>
        </body>
    </html>
    """
    return html_content

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_loaded = 'model' in MODEL_ARTIFACTS and MODEL_ARTIFACTS['model'] is not None
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_version=MODEL_ARTIFACTS.get('model_info', {}).get('model_name', 'unknown'),
        api_version="1.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(passenger: PassengerFeatures):
    """Make a single prediction"""
    start_time = time.time()
    
    # Check if model is loaded
    if not MODEL_ARTIFACTS or 'model' not in MODEL_ARTIFACTS:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess data
        passenger_dict = passenger.dict()
        X = preprocess_data(passenger_dict)
        
        # Make prediction
        prediction = MODEL_ARTIFACTS['model'].predict(X)[0]
        probability = MODEL_ARTIFACTS['model'].predict_proba(X)[0, 1]
        
        # Prepare response
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            prediction="satisfied" if prediction == 1 else "unsatisfied",
            probability=float(probability),
            confidence_level=get_confidence_level(probability),
            processing_time=processing_time,
            model_version=MODEL_ARTIFACTS['model_info']['model_name'],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(file: UploadFile = File(...)):
    """Make batch predictions from CSV file"""
    start_time = time.time()
    
    # Check if model is loaded
    if not MODEL_ARTIFACTS or 'model' not in MODEL_ARTIFACTS:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        logger.info(f"Processing batch prediction for {len(df)} records")
        
        predictions = []
        successful = 0
        failed = 0
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                # Convert row to PassengerFeatures format
                passenger_data = {
                    'gender': row.get('Gender', row.get('gender', 'Male')),
                    'customer_type': row.get('Customer Type', row.get('customer_type', 'Loyal Customer')),
                    'age': int(row.get('Age', row.get('age', 30))),
                    'type_of_travel': row.get('Type of Travel', row.get('type_of_travel', 'Business travel')),
                    'travel_class': row.get('Class', row.get('travel_class', 'Eco')),
                    'flight_distance': int(row.get('Flight Distance', row.get('flight_distance', 1000))),
                    'inflight_wifi_service': int(row.get('Inflight wifi service', row.get('inflight_wifi_service', 3))),
                    'departure_arrival_time_convenient': int(row.get('Departure/Arrival time convenient', row.get('departure_arrival_time_convenient', 3))),
                    'ease_of_online_booking': int(row.get('Ease of Online booking', row.get('ease_of_online_booking', 3))),
                    'gate_location': int(row.get('Gate location', row.get('gate_location', 3))),
                    'food_and_drink': int(row.get('Food and drink', row.get('food_and_drink', 3))),
                    'online_boarding': int(row.get('Online boarding', row.get('online_boarding', 3))),
                    'seat_comfort': int(row.get('Seat comfort', row.get('seat_comfort', 3))),
                    'inflight_entertainment': int(row.get('Inflight entertainment', row.get('inflight_entertainment', 3))),
                    'on_board_service': int(row.get('On-board service', row.get('on_board_service', 3))),
                    'leg_room_service': int(row.get('Leg room service', row.get('leg_room_service', 3))),
                    'baggage_handling': int(row.get('Baggage handling', row.get('baggage_handling', 3))),
                    'checkin_service': int(row.get('Checkin service', row.get('checkin_service', 3))),
                    'inflight_service': int(row.get('Inflight service', row.get('inflight_service', 3))),
                    'cleanliness': int(row.get('Cleanliness', row.get('cleanliness', 3))),
                    'departure_delay_in_minutes': int(row.get('Departure Delay in Minutes', row.get('departure_delay_in_minutes', 0))),
                    'arrival_delay_in_minutes': int(row.get('Arrival Delay in Minutes', row.get('arrival_delay_in_minutes', 0)))
                }
                
                # Make prediction
                X = preprocess_data(passenger_data)
                prediction = MODEL_ARTIFACTS['model'].predict(X)[0]
                probability = MODEL_ARTIFACTS['model'].predict_proba(X)[0, 1]
                
                pred_result = PredictionResponse(
                    prediction="satisfied" if prediction == 1 else "unsatisfied",
                    probability=float(probability),
                    confidence_level=get_confidence_level(probability),
                    processing_time=0.0,
                    model_version=MODEL_ARTIFACTS['model_info']['model_name'],
                    timestamp=datetime.now().isoformat()
                )
                
                predictions.append(pred_result)
                successful += 1
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                failed += 1
        
        processing_time = time.time() - start_time
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(df),
            successful=successful,
            failed=failed,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if not MODEL_ARTIFACTS or 'model_info' not in MODEL_ARTIFACTS:
        raise HTTPException(status_code=503, detail="Model information not available")
    
    model_info = MODEL_ARTIFACTS['model_info'].copy()
    
    # Add current status
    model_info['status'] = 'loaded'
    model_info['api_version'] = '1.0.0'
    model_info['timestamp'] = datetime.now().isoformat()
    
    return model_info

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )