import pytest
import asyncio
from fastapi.testclient import TestClient
from main import app
import json

# Cliente de teste
client = TestClient(app)

# Headers com autenticação
headers = {"Authorization": "Bearer sk-change-this-in-production"}

class TestAPI:
    """Testes para a API de Satisfação de Passageiros"""
    
    def test_root_endpoint(self):
        """Testa o endpoint raiz"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "Airline Passenger Satisfaction API"
    
    def test_health_check(self):
        """Testa o endpoint de health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "uptime_seconds" in data
    
    def test_predict_valid_data(self):
        """Testa predição com dados válidos"""
        passenger_data = {
            "age": 35,
            "flight_distance": 1500,
            "inflight_wifi_service": 4,
            "departure_arrival_time_convenient": 4,
            "ease_of_online_booking": 3,
            "gate_location": 4,
            "food_and_drink": 3,
            "online_boarding": 4,
            "seat_comfort": 4,
            "inflight_entertainment": 3,
            "on_board_service": 4,
            "leg_room_service": 3,
            "baggage_handling": 4,
            "checkin_service": 4,
            "inflight_service": 4,
            "cleanliness": 4,
            "departure_delay_in_minutes": 0,
            "arrival_delay_in_minutes": 0,
            "gender": "Male",
            "customer_type": "Loyal Customer",
            "type_of_travel": "Business travel",
            "class_type": "Business"
        }
        
        response = client.post("/predict", json=passenger_data, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "confidence" in data
        assert "model_version" in data
        assert "processing_time_ms" in data
        assert data["prediction"] in ["Satisfied", "Neutral or Dissatisfied"]
        assert 0 <= data["probability"] <= 1
        assert data["confidence"] in ["High", "Medium", "Low"]
    
    def test_predict_invalid_age(self):
        """Testa predição com idade inválida"""
        passenger_data = {
            "age": 150,  # Idade inválida
            "flight_distance": 1500,
            "inflight_wifi_service": 4,
            "departure_arrival_time_convenient": 4,
            "ease_of_online_booking": 3,
            "gate_location": 4,
            "food_and_drink": 3,
            "online_boarding": 4,
            "seat_comfort": 4,
            "inflight_entertainment": 3,
            "on_board_service": 4,
            "leg_room_service": 3,
            "baggage_handling": 4,
            "checkin_service": 4,
            "inflight_service": 4,
            "cleanliness": 4,
            "departure_delay_in_minutes": 0,
            "arrival_delay_in_minutes": 0,
            "gender": "Male",
            "customer_type": "Loyal Customer",
            "type_of_travel": "Business travel",
            "class_type": "Business"
        }
        
        response = client.post("/predict", json=passenger_data, headers=headers)
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_gender(self):
        """Testa predição com gênero inválido"""
        passenger_data = {
            "age": 35,
            "flight_distance": 1500,
            "inflight_wifi_service": 4,
            "departure_arrival_time_convenient": 4,
            "ease_of_online_booking": 3,
            "gate_location": 4,
            "food_and_drink": 3,
            "online_boarding": 4,
            "seat_comfort": 4,
            "inflight_entertainment": 3,
            "on_board_service": 4,
            "leg_room_service": 3,
            "baggage_handling": 4,
            "checkin_service": 4,
            "inflight_service": 4,
            "cleanliness": 4,
            "departure_delay_in_minutes": 0,
            "arrival_delay_in_minutes": 0,
            "gender": "Other",  # Gênero inválido
            "customer_type": "Loyal Customer",
            "type_of_travel": "Business travel",
            "class_type": "Business"
        }
        
        response = client.post("/predict", json=passenger_data, headers=headers)
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict_valid_data(self):
        """Testa predição em lote com dados válidos"""
        passenger_data = {
            "age": 35,
            "flight_distance": 1500,
            "inflight_wifi_service": 4,
            "departure_arrival_time_convenient": 4,
            "ease_of_online_booking": 3,
            "gate_location": 4,
            "food_and_drink": 3,
            "online_boarding": 4,
            "seat_comfort": 4,
            "inflight_entertainment": 3,
            "on_board_service": 4,
            "leg_room_service": 3,
            "baggage_handling": 4,
            "checkin_service": 4,
            "inflight_service": 4,
            "cleanliness": 4,
            "departure_delay_in_minutes": 0,
            "arrival_delay_in_minutes": 0,
            "gender": "Male",
            "customer_type": "Loyal Customer",
            "type_of_travel": "Business travel",
            "class_type": "Business"
        }
        
        batch_request = {
            "passengers": [passenger_data, passenger_data]
        }
        
        response = client.post("/predict/batch", json=batch_request, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "total_processed" in data
        assert "processing_time_ms" in data
        assert "model_version" in data
        assert len(data["predictions"]) == 2
        assert data["total_processed"] == 2
    
    def test_batch_predict_too_many_passengers(self):
        """Testa predição em lote com muitos passageiros"""
        passenger_data = {
            "age": 35,
            "flight_distance": 1500,
            "inflight_wifi_service": 4,
            "departure_arrival_time_convenient": 4,
            "ease_of_online_booking": 3,
            "gate_location": 4,
            "food_and_drink": 3,
            "online_boarding": 4,
            "seat_comfort": 4,
            "inflight_entertainment": 3,
            "on_board_service": 4,
            "leg_room_service": 3,
            "baggage_handling": 4,
            "checkin_service": 4,
            "inflight_service": 4,
            "cleanliness": 4,
            "departure_delay_in_minutes": 0,
            "arrival_delay_in_minutes": 0,
            "gender": "Male",
            "customer_type": "Loyal Customer",
            "type_of_travel": "Business travel",
            "class_type": "Business"
        }
        
        # Criar lista com mais de 100 passageiros
        batch_request = {
            "passengers": [passenger_data] * 101
        }
        
        response = client.post("/predict/batch", json=batch_request, headers=headers)
        assert response.status_code == 422  # Validation error
    
    def test_model_info(self):
        """Testa o endpoint de informações do modelo"""
        response = client.get("/model/info", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "model_version" in data
        assert "features_count" in data
        assert "feature_names" in data
        assert "model_loaded" in data
        assert "supported_classes" in data
        assert data["model_type"] == "CatBoost Classifier"
        assert "Satisfied" in data["supported_classes"]
        assert "Neutral or Dissatisfied" in data["supported_classes"]
    
    def test_metrics_endpoint(self):
        """Testa o endpoint de métricas"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
    
    def test_unauthorized_access(self):
        """Testa acesso não autorizado"""
        passenger_data = {
            "age": 35,
            "flight_distance": 1500,
            "inflight_wifi_service": 4,
            "departure_arrival_time_convenient": 4,
            "ease_of_online_booking": 3,
            "gate_location": 4,
            "food_and_drink": 3,
            "online_boarding": 4,
            "seat_comfort": 4,
            "inflight_entertainment": 3,
            "on_board_service": 4,
            "leg_room_service": 3,
            "baggage_handling": 4,
            "checkin_service": 4,
            "inflight_service": 4,
            "cleanliness": 4,
            "departure_delay_in_minutes": 0,
            "arrival_delay_in_minutes": 0,
            "gender": "Male",
            "customer_type": "Loyal Customer",
            "type_of_travel": "Business travel",
            "class_type": "Business"
        }
        
        # Tentar fazer predição sem headers de autenticação
        response = client.post("/predict", json=passenger_data)
        assert response.status_code == 403  # Forbidden
    
    def test_invalid_api_key(self):
        """Testa API key inválida"""
        invalid_headers = {"Authorization": "Bearer invalid-key"}
        
        passenger_data = {
            "age": 35,
            "flight_distance": 1500,
            "inflight_wifi_service": 4,
            "departure_arrival_time_convenient": 4,
            "ease_of_online_booking": 3,
            "gate_location": 4,
            "food_and_drink": 3,
            "online_boarding": 4,
            "seat_comfort": 4,
            "inflight_entertainment": 3,
            "on_board_service": 4,
            "leg_room_service": 3,
            "baggage_handling": 4,
            "checkin_service": 4,
            "inflight_service": 4,
            "cleanliness": 4,
            "departure_delay_in_minutes": 0,
            "arrival_delay_in_minutes": 0,
            "gender": "Male",
            "customer_type": "Loyal Customer",
            "type_of_travel": "Business travel",
            "class_type": "Business"
        }
        
        response = client.post("/predict", json=passenger_data, headers=invalid_headers)
        assert response.status_code == 401  # Unauthorized

@pytest.fixture
def sample_passenger():
    """Fixture com dados de passageiro para testes"""
    return {
        "age": 35,
        "flight_distance": 1500,
        "inflight_wifi_service": 4,
        "departure_arrival_time_convenient": 4,
        "ease_of_online_booking": 3,
        "gate_location": 4,
        "food_and_drink": 3,
        "online_boarding": 4,
        "seat_comfort": 4,
        "inflight_entertainment": 3,
        "on_board_service": 4,
        "leg_room_service": 3,
        "baggage_handling": 4,
        "checkin_service": 4,
        "inflight_service": 4,
        "cleanliness": 4,
        "departure_delay_in_minutes": 0,
        "arrival_delay_in_minutes": 0,
        "gender": "Male",
        "customer_type": "Loyal Customer",
        "type_of_travel": "Business travel",
        "class_type": "Business"
    }

def test_confidence_levels():
    """Testa os níveis de confiança"""
    from main import get_confidence_level
    
    assert get_confidence_level(0.9) == "High"
    assert get_confidence_level(0.1) == "High"
    assert get_confidence_level(0.7) == "Medium"
    assert get_confidence_level(0.3) == "Medium"
    assert get_confidence_level(0.5) == "Low"

if __name__ == "__main__":
    # Executar testes
    pytest.main([__file__, "-v"])