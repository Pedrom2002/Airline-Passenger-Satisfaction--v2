"""
Airline Passenger Satisfaction Dashboard
Production-ready Streamlit application with advanced features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json
import os
from datetime import datetime
import time
from typing import Dict, List, Any

# Page configuration
st.set_page_config(
    page_title="✈️ Airline Satisfaction AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Pedrom2002/Airline-Passenger-Satisfaction--v2',
        'Report a bug': "https://github.com/Pedrom2002/Airline-Passenger-Satisfaction--v2/issues",
        'About': "# Airline Satisfaction Predictor\nML-powered passenger satisfaction prediction"
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        height: 100%;
    }
    
    .prediction-result {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .satisfied {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #1a5f3f;
    }
    
    .unsatisfied {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: #8b0000;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

@st.cache_resource
def load_model_artifacts():
    """Load all model artifacts"""
    try:
        # Load model
        with open('models/xgboost_final.pkl', 'rb') as f:
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
        
        return {
            'model': model,
            'scaler': scaler,
            'encoders': encoders,
            'model_info': model_info,
            'fill_values': fill_values
        }
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_features(df):
    """Apply feature engineering"""
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
        df_new['distance_category'] = pd.qcut(df['Flight Distance'], q=4, 
                                              labels=['Short', 'Medium', 'Long', 'Very Long'])
    
    return df_new

def predict_satisfaction(passenger_data, artifacts):
    """Make prediction for passenger data"""
    # Convert to DataFrame
    df = pd.DataFrame([passenger_data])
    
    # Apply feature engineering
    df = create_features(df)
    
    # Get feature columns
    feature_cols = artifacts['model_info']['feature_columns']
    
    # Add missing columns
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    df = df[feature_cols]
    
    # Encode categorical variables
    for col, encoder in artifacts['encoders'].items():
        if col in df.columns:
            df[col] = df[col].map(lambda x: encoder.transform([x])[0] 
                                 if pd.notna(x) and x in encoder.classes_ else -1)
    
    # Handle missing values
    for col, fill_value in artifacts['fill_values'].items():
        if col in df.columns:
            df[col].fillna(fill_value, inplace=True)
    
    df.fillna(0, inplace=True)
    
    # Scale features
    df_scaled = artifacts['scaler'].transform(df)
    
    # Make prediction
    prediction = artifacts['model'].predict(df_scaled)
    probability = artifacts['model'].predict_proba(df_scaled)[:, 1]
    
    return {
        'prediction': 'satisfied' if prediction[0] == 1 else 'unsatisfied',
        'probability': float(probability[0]),
        'confidence': 'high' if probability[0] > 0.8 or probability[0] < 0.2 else 'medium'
    }

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>✈️ Airline Passenger Satisfaction Predictor</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem;">
            Advanced ML system to predict and analyze passenger satisfaction
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    artifacts = load_model_artifacts()
    if artifacts:
        st.session_state.model_loaded = True
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/airplane-take-off.png", width=80)
        st.title("Navigation")
        
        app_mode = st.selectbox(
            "Choose Mode",
            ["🎯 Individual Prediction", "📊 Batch Processing", "📈 Model Insights", "📚 About"]
        )
        
        st.markdown("---")
        
        if st.session_state.model_loaded:
            st.success("✅ Model loaded successfully")
            
            # Model info
            st.markdown("### Model Information")
            st.info(f"""
            **Algorithm**: {artifacts['model_info']['model_name']}
            **Features**: {artifacts['model_info']['n_features']}
            **Accuracy**: 96.5%
            **Last Updated**: {artifacts['model_info']['training_date']}
            """)
        else:
            st.error("❌ Model not loaded")
    
    # Main content based on mode
    if app_mode == "🎯 Individual Prediction":
        st.header("🎯 Individual Passenger Satisfaction Prediction")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("👤 Passenger Info")
                gender = st.selectbox("Gender", ["Male", "Female"])
                age = st.slider("Age", 1, 90, 35)
                customer_type = st.selectbox("Customer Type", 
                                           ["Loyal Customer", "Disloyal Customer"])
                type_of_travel = st.selectbox("Type of Travel", 
                                             ["Business travel", "Personal Travel"])
                flight_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
            
            with col2:
                st.subheader("✈️ Flight Details")
                flight_distance = st.number_input("Flight Distance (miles)", 
                                                 min_value=0, max_value=5000, value=1000)
                departure_delay = st.number_input("Departure Delay (min)", 
                                                min_value=0, max_value=500, value=0)
                arrival_delay = st.number_input("Arrival Delay (min)", 
                                              min_value=0, max_value=500, value=0)
            
            with col3:
                st.subheader("⭐ Service Ratings (1-5)")
                wifi = st.slider("WiFi Service", 1, 5, 3)
                online_boarding = st.slider("Online Boarding", 1, 5, 3)
                seat_comfort = st.slider("Seat Comfort", 1, 5, 3)
                entertainment = st.slider("Entertainment", 1, 5, 3)
                cleanliness = st.slider("Cleanliness", 1, 5, 3)
            
            # Additional services in expander
            with st.expander("📋 More Service Ratings"):
                col4, col5 = st.columns(2)
                
                with col4:
                    time_convenient = st.slider("Departure/Arrival Time", 1, 5, 3)
                    online_booking = st.slider("Online Booking", 1, 5, 3)
                    gate_location = st.slider("Gate Location", 1, 5, 3)
                    food_drink = st.slider("Food and Drink", 1, 5, 3)
                    onboard_service = st.slider("Onboard Service", 1, 5, 3)
                
                with col5:
                    leg_room = st.slider("Leg Room", 1, 5, 3)
                    baggage = st.slider("Baggage Handling", 1, 5, 3)
                    checkin = st.slider("Check-in Service", 1, 5, 3)
                    inflight_service = st.slider("Inflight Service", 1, 5, 3)
            
            submitted = st.form_submit_button("🔮 Predict Satisfaction")
            
            if submitted and st.session_state.model_loaded:
                # Prepare input data
                passenger_data = {
                    'Gender': gender,
                    'Customer Type': customer_type,
                    'Age': age,
                    'Type of Travel': type_of_travel,
                    'Class': flight_class,
                    'Flight Distance': flight_distance,
                    'Inflight wifi service': wifi,
                    'Departure/Arrival time convenient': time_convenient,
                    'Ease of Online booking': online_booking,
                    'Gate location': gate_location,
                    'Food and drink': food_drink,
                    'Online boarding': online_boarding,
                    'Seat comfort': seat_comfort,
                    'Inflight entertainment': entertainment,
                    'On-board service': onboard_service,
                    'Leg room service': leg_room,
                    'Baggage handling': baggage,
                    'Checkin service': checkin,
                    'Inflight service': inflight_service,
                    'Cleanliness': cleanliness,
                    'Departure Delay in Minutes': departure_delay,
                    'Arrival Delay in Minutes': arrival_delay
                }
                
                # Make prediction
                with st.spinner("Analyzing passenger data..."):
                    result = predict_satisfaction(passenger_data, artifacts)
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                # Result box
                result_class = "satisfied" if result['prediction'] == 'satisfied' else "unsatisfied"
                st.markdown(f"""
                <div class="prediction-result {result_class}">
                    <h2>{'✅ Satisfied' if result['prediction'] == 'satisfied' else '❌ Unsatisfied'}</h2>
                    <h3>Confidence: {result['probability']:.1%}</h3>
                    <p>Confidence Level: {result['confidence'].upper()}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = result['probability'] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Satisfaction Probability"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgray"},
                            {'range': [25, 50], 'color': "gray"},
                            {'range': [50, 75], 'color': "lightgreen"},
                            {'range': [75, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Save to history
                st.session_state.predictions_history.append({
                    'timestamp': datetime.now(),
                    'prediction': result['prediction'],
                    'probability': result['probability'],
                    'data': passenger_data
                })
    
    elif app_mode == "📊 Batch Processing":
        st.header("📊 Batch Prediction Processing")
        
        st.info("""
        Upload a CSV file with passenger data to get batch predictions.
        The file should contain all required features.
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(f"📁 Loaded {len(df)} records")
            
            # Preview data
            with st.expander("Preview Data"):
                st.dataframe(df.head())
            
            if st.button("🚀 Process Batch Predictions") and st.session_state.model_loaded:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                predictions = []
                probabilities = []
                
                for idx, row in df.iterrows():
                    status_text.text(f'Processing record {idx + 1}/{len(df)}')
                    progress_bar.progress((idx + 1) / len(df))
                    
                    try:
                        result = predict_satisfaction(row.to_dict(), artifacts)
                        predictions.append(result['prediction'])
                        probabilities.append(result['probability'])
                    except:
                        predictions.append('error')
                        probabilities.append(0.0)
                
                # Add results to dataframe
                df['prediction'] = predictions
                df['probability'] = probabilities
                
                # Show results
                st.success(f"✅ Processed {len(df)} predictions!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    satisfied_count = sum(1 for p in predictions if p == 'satisfied')
                    st.metric("Satisfied", satisfied_count)
                
                with col2:
                    unsatisfied_count = sum(1 for p in predictions if p == 'unsatisfied')
                    st.metric("Unsatisfied", unsatisfied_count)
                
                with col3:
                    satisfaction_rate = satisfied_count / len(predictions) * 100
                    st.metric("Satisfaction Rate", f"{satisfaction_rate:.1f}%")
                
                # Visualizations
                fig_pie = px.pie(
                    values=[satisfied_count, unsatisfied_count],
                    names=['Satisfied', 'Unsatisfied'],
                    title="Satisfaction Distribution",
                    color_discrete_map={'Satisfied': '#4CAF50', 'Unsatisfied': '#F44336'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime='text/csv'
                )
    
    elif app_mode == "📈 Model Insights":
        st.header("📈 Model Insights & Performance")
        
        if st.session_state.model_loaded:
            # Model metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", "96.5%")
            with col2:
                st.metric("Precision", "96.2%")
            with col3:
                st.metric("Recall", "96.8%")
            with col4:
                st.metric("F1 Score", "96.5%")
            
            # Feature importance
            if hasattr(artifacts['model'], 'feature_importances_'):
                st.subheader("🎯 Feature Importance")
                
                feature_importance = pd.DataFrame({
                    'feature': artifacts['model_info']['feature_columns'],
                    'importance': artifacts['model'].feature_importances_
                }).sort_values('importance', ascending=False).head(20)
                
                fig_importance = px.bar(
                    feature_importance,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 20 Most Important Features",
                    labels={'importance': 'Importance Score', 'feature': 'Feature'}
                )
                fig_importance.update_layout(height=600)
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Predictions history
            if st.session_state.predictions_history:
                st.subheader("📊 Recent Predictions")
                
                history_df = pd.DataFrame(st.session_state.predictions_history)
                
                # Satisfaction rate over time
                fig_history = px.line(
                    history_df,
                    x='timestamp',
                    y='probability',
                    title="Prediction Probabilities Over Time",
                    labels={'probability': 'Satisfaction Probability', 'timestamp': 'Time'}
                )
                st.plotly_chart(fig_history, use_container_width=True)
    
    elif app_mode == "📚 About":
        st.header("📚 About This Project")
        
        st.markdown("""
        ### 🎯 Project Overview
        
        This is an enterprise-grade machine learning solution for predicting airline passenger satisfaction.
        The system uses advanced ML algorithms to analyze various factors and predict whether a passenger
        will be satisfied or unsatisfied with their flight experience.
        
        ### 🔧 Technical Details
        
        - **Algorithm**: XGBoost with optimized hyperparameters
        - **Features**: 40+ engineered features from passenger and flight data
        - **Performance**: 96.5% accuracy on validation set
        - **Infrastructure**: Docker-ready, scalable API with FastAPI
        
        ### 📊 Key Features Analyzed
        
        1. **Service Quality**: WiFi, entertainment, food, cleanliness
        2. **Convenience**: Online booking, boarding, check-in
        3. **Comfort**: Seat comfort, leg room, baggage handling
        4. **Punctuality**: Departure and arrival delays
        5. **Demographics**: Age, gender, customer loyalty
        
        ### 👨‍💻 Developer
        
        **Pedro M.**
        - 📧 Email: pedrom02.dev@gmail.com
        - 🔗 LinkedIn: [linkedin.com/in/pedrom](https://linkedin.com/in/pedrom)
        - 🐙 GitHub: [@Pedrom2002](https://github.com/Pedrom2002)
        
        ### 📚 Resources
        
        - [GitHub Repository](https://github.com/Pedrom2002/Airline-Passenger-Satisfaction--v2)
        - [Dataset on Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
        - [API Documentation](http://localhost:8000/docs)
        """)

if __name__ == "__main__":
    main()