"""
API tests for Airline Satisfaction Prediction
"""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "Airline Satisfaction" in response.text

def test_health():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "timestamp" in data

def test_predict_valid():
    """Test prediction with valid data"""
    test_passenger = {
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
    
    response = client.post("/predict", json=test_passenger)
    
    # Check if model is loaded
    health_response = client.get("/health")
    health_data = health_response.json()
    
    if health_data["model_loaded"]:
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in ["satisfied", "unsatisfied"]
        assert "probability" in data
        assert 0 <= data["probability"] <= 1
        assert "confidence_level" in data
    else:
        assert response.status_code == 503

def test_predict_invalid_gender():
    """Test prediction with invalid gender"""
    test_passenger = {
        "gender": "Invalid",
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
    
    response = client.post("/predict", json=test_passenger)
    assert response.status_code == 422  # Validation error

def test_predict_invalid_age():
    """Test prediction with invalid age"""
    test_passenger = {
        "gender": "Male",
        "customer_type": "Loyal Customer",
        "age": 150,  # Invalid age
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
    
    response = client.post("/predict", json=test_passenger)
    assert response.status_code == 422  # Validation error

def test_predict_invalid_rating():
    """Test prediction with invalid service rating"""
    test_passenger = {
        "gender": "Male",
        "customer_type": "Loyal Customer",
        "age": 35,
        "type_of_travel": "Business travel",
        "travel_class": "Business",
        "flight_distance": 1500,
        "inflight_wifi_service": 10,  # Invalid rating (should be 1-5)
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
    
    response = client.post("/predict", json=test_passenger)
    assert response.status_code == 422  # Validation error

def test_model_info():
    """Test model info endpoint"""
    response = client.get("/model/info")
    
    # Check if model is loaded
    health_response = client.get("/health")
    health_data = health_response.json()
    
    if health_data["model_loaded"]:
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "n_features" in data
    else:
        assert response.status_code == 503

if __name__ == "__main__":
    pytest.main([__file__, "-v"])