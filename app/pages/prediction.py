"""
Prediction Module for Airline Passenger Satisfaction
Handles individual and batch predictions using CatBoost model
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
import joblib
import catboost as cb

import streamlit as st
import pandas as pd
import numpy as np
import catboost as cb
import joblib
import json
import os
from typing import Dict, List, Any

# ============================================================================
# CLEAN MODEL LOADING AND PREDICTION FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model_artifacts():
    """Load the trained CatBoost model and preprocessing info - CLEAN VERSION"""
    try:
        # Load the production model we saved
        model_path = '../models/catboost_production_model.pkl'
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            return None
            
        model = joblib.load(model_path)
        
        # Load preprocessing info
        preprocessing_path = '../models/preprocessing_info.pkl'
        if not os.path.exists(preprocessing_path):
            st.error(f"❌ Preprocessing info not found: {preprocessing_path}")
            return None
            
        preprocessing_info = joblib.load(preprocessing_path)
        
        return {
            'model': model,
            'feature_columns': preprocessing_info['feature_columns'],
            'numeric_columns': preprocessing_info['numeric_columns'],
            'categorical_columns': preprocessing_info['categorical_columns'],
            'categorical_indices': preprocessing_info['categorical_indices']
        }
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def create_enhanced_features(df):
    """
    Apply the EXACT same feature engineering we used during training
    FIXED: Now creates ALL expected features without missing any
    """
    df_new = df.copy()
    
    # Service columns (as defined in our training)
    service_cols = [
        'Inflight wifi service', 'Departure/Arrival time convenient',
        'Ease of Online booking', 'Gate location', 'Food and drink',
        'Online boarding', 'Seat comfort', 'Inflight entertainment',
        'On-board service', 'Leg room service', 'Baggage handling',
        'Checkin service', 'Inflight service', 'Cleanliness'
    ]
    
    # Check which service columns exist and fill missing ones with default value (3)
    for col in service_cols:
        if col not in df.columns:
            df_new[col] = 3  # Default neutral rating
    
    # Now all service columns are guaranteed to exist
    available_service_cols = [col for col in service_cols if col in df_new.columns]
    
    # Service aggregate features
    df_new['avg_service_rating'] = df_new[available_service_cols].mean(axis=1)
    df_new['min_service_rating'] = df_new[available_service_cols].min(axis=1)
    df_new['max_service_rating'] = df_new[available_service_cols].max(axis=1)
    df_new['service_rating_std'] = df_new[available_service_cols].std(axis=1).fillna(0)
    df_new['low_rating_count'] = (df_new[available_service_cols] <= 2).sum(axis=1)
    df_new['high_rating_count'] = (df_new[available_service_cols] >= 4).sum(axis=1)
    df_new['service_consistency'] = df_new['max_service_rating'] - df_new['min_service_rating']
    
    # Domain-specific scores
    digital_cols = ['Inflight wifi service', 'Ease of Online booking', 'Online boarding']
    df_new['digital_experience_score'] = df_new[digital_cols].mean(axis=1)
    
    comfort_cols = ['Seat comfort', 'Leg room service', 'Cleanliness']
    df_new['comfort_score'] = df_new[comfort_cols].mean(axis=1)
    
    # Delay features
    if 'Departure Delay in Minutes' not in df.columns:
        df_new['Departure Delay in Minutes'] = 0
    if 'Arrival Delay in Minutes' not in df.columns:
        df_new['Arrival Delay in Minutes'] = 0
        
    df_new['total_delay'] = (df_new['Departure Delay in Minutes'].fillna(0) + 
                            df_new['Arrival Delay in Minutes'].fillna(0))
    df_new['delay_difference'] = (df_new['Arrival Delay in Minutes'].fillna(0) - 
                                 df_new['Departure Delay in Minutes'].fillna(0))
    df_new['is_delayed'] = (df_new['total_delay'] > 15).astype(int)
    df_new['severe_delay'] = (df_new['total_delay'] > 60).astype(int)
    df_new['no_delay'] = (df_new['total_delay'] == 0).astype(int)
    
    # Age features
    if 'Age' not in df.columns:
        df_new['Age'] = 35  # Default age
        
    age = df_new['Age'].fillna(35)
    df_new['age_squared'] = age ** 2
    df_new['is_senior'] = (age >= 60).astype(int)
    df_new['is_young'] = (age <= 25).astype(int)
    df_new['is_middle_age'] = ((age > 35) & (age < 55)).astype(int)
    
    # Distance features
    if 'Flight Distance' not in df.columns:
        df_new['Flight Distance'] = 1000  # Default distance
        
    distance = df_new['Flight Distance'].fillna(1000)
    df_new['log_distance'] = np.log1p(distance)
    df_new['is_long_flight'] = (distance > 1000).astype(int)
    df_new['is_short_flight'] = (distance < 500).astype(int)
    
    # Customer interaction features
    if 'Customer Type' not in df.columns:
        df_new['Customer Type'] = 'Loyal Customer'
    if 'Class' not in df.columns:
        df_new['Class'] = 'Business'
        
    is_loyal = (df_new['Customer Type'] == 'Loyal Customer')
    is_business = (df_new['Class'] == 'Business')
    df_new['business_loyal'] = (is_loyal & is_business).astype(int)
    
    # Delay satisfaction risk
    risk_base = 1 - (df_new['avg_service_rating'] / 5)
    df_new['delay_satisfaction_risk'] = df_new['is_delayed'] * risk_base
    
    return df_new

def predict_satisfaction_clean(passenger_data, artifacts):
    """CLEAN prediction function using our trained model"""
    try:
        # Convert to DataFrame if it's a dict
        if isinstance(passenger_data, dict):
            df = pd.DataFrame([passenger_data])
        else:
            df = passenger_data.copy()
        
        # Apply feature engineering
        df_engineered = create_enhanced_features(df)
        
        # Get expected features from our training
        expected_features = artifacts['feature_columns']
        
        # Ensure all required features exist
        missing_features = []
        for feature in expected_features:
            if feature not in df_engineered.columns:
                missing_features.append(feature)
                # Add default values for missing features
                if 'rating' in feature.lower() or 'score' in feature.lower():
                    df_engineered[feature] = 3.0
                elif feature in ['Age', 'Flight Distance']:
                    default_val = 35 if 'Age' in feature else 1000
                    df_engineered[feature] = default_val
                else:
                    df_engineered[feature] = 0
        
        # Select only the features used during training, in the correct order
        df_final = df_engineered[expected_features]
        
        # Handle categorical features - CatBoost expects string categories
        for col in artifacts['categorical_columns']:
            if col in df_final.columns:
                df_final[col] = df_final[col].astype(str)
        
        # Make prediction
        model = artifacts['model']
        prediction = model.predict(df_final)[0]
        probability = model.predict_proba(df_final)[0, 1]
        
        # Determine confidence level
        if probability >= 0.8 or probability <= 0.2:
            confidence = 'high'
        elif probability >= 0.65 or probability <= 0.35:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'prediction': 'satisfied' if prediction == 1 else 'neutral or dissatisfied',
            'probability': float(probability),
            'confidence': confidence,
            'features_used': len(expected_features),
            'missing_features': missing_features
        }
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return {
            'prediction': 'error',
            'probability': 0.0,
            'confidence': 'none',
            'error': str(e)
        }

def predict_batch_clean(df_batch, artifacts):
    """CLEAN batch prediction function"""
    try:
        # Apply feature engineering
        df_engineered = create_enhanced_features(df_batch)
        
        # Get expected features
        expected_features = artifacts['feature_columns']
        
        # Ensure all required features exist
        for feature in expected_features:
            if feature not in df_engineered.columns:
                if 'rating' in feature.lower() or 'score' in feature.lower():
                    df_engineered[feature] = 3.0
                elif feature in ['Age', 'Flight Distance']:
                    default_val = 35 if 'Age' in feature else 1000
                    df_engineered[feature] = default_val
                else:
                    df_engineered[feature] = 0
        
        # Select features in correct order
        df_final = df_engineered[expected_features]
        
        # Handle categorical features
        for col in artifacts['categorical_columns']:
            if col in df_final.columns:
                df_final[col] = df_final[col].astype(str)
        
        # Make predictions
        model = artifacts['model']
        predictions = model.predict(df_final)
        probabilities = model.predict_proba(df_final)[:, 1]
        
        # Format results
        results = []
        for i in range(len(predictions)):
            prob = probabilities[i]
            confidence = 'high' if prob >= 0.8 or prob <= 0.2 else 'medium' if prob >= 0.65 or prob <= 0.35 else 'low'
            
            results.append({
                'prediction': 'satisfied' if predictions[i] == 1 else 'neutral or dissatisfied',
                'probability': float(prob),
                'confidence': confidence
            })
        
        return results
        
    except Exception as e:
        st.error(f"❌ Batch prediction error: {str(e)}")
        return [{'prediction': 'error', 'probability': 0.0, 'confidence': 'none'} for _ in range(len(df_batch))]

# ============================================================================
# INDIVIDUAL PREDICTION INTERFACE - UPDATED
# ============================================================================

def render_individual_prediction_page(artifacts):
    """Render the individual prediction interface - UPDATED VERSION"""
    
    # Add the beautiful CSS animations from the original
    st.markdown("""
    <style>
    @keyframes slideInFromLeft {
        0% { transform: translateX(-100%); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes fadeInUp {
        0% { transform: translateY(30px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes floatAnimation {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .prediction-result {
        animation: fadeInUp 0.8s ease-out;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-result:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.2) !important;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-size: 200% 200%;
        animation: gradientShift 3s ease infinite, floatAnimation 3s ease-in-out infinite;
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05) rotate(2deg);
        animation-play-state: paused;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 4s ease infinite;
        background-size: 200% 200%;
    }
    
    .section-divider {
        height: 4px;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FECA57, #FF9FF3);
        border-radius: 2px;
        margin: 30px 0;
        animation: gradientShift 4s ease infinite;
        background-size: 300% 300%;
    }
    
    .floating-icon {
        animation: floatAnimation 3s ease-in-out infinite;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Enhanced header
    st.markdown('''
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 class="main-header" style="font-size: 3em; margin-bottom: 15px;">
            <span class="floating-icon">🎯</span> 
            Individual Passenger Satisfaction Prediction
            <span class="floating-icon" style="animation-delay: 1s;">✈️</span>
        </h1>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Model information
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 25px; border-radius: 15px; color: white; margin: 20px 0;
                animation: fadeInUp 0.6s ease-out;">
        <h3 style="margin: 0; color: white;">🤖 Advanced AI Prediction</h3>
        <p style="margin: 10px 0 0 0; font-size: 1.1em;">
            This model uses CatBoost with advanced feature engineering to predict passenger 
            satisfaction based on demographics, flight details, and service ratings.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Input form
    with st.form("prediction_form"):
        # Form header
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
            padding: 25px;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin: 25px 0;
            box-shadow: 0 8px 25px rgba(78, 205, 196, 0.3);
        ">
            <h2 style="margin: 0; color: white;">✈️ Passenger Information</h2>
            <p style="margin: 10px 0 0 0; opacity: 0.9;">
                Fill in the details below to get a personalized prediction
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Demographics section
        st.markdown("""
        <div style="
            background: white;
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        ">
            <h3 style="color: #667eea; margin-bottom: 20px; text-align: center;">
                👤 Demographics and Travel Information
            </h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🚻 Gender**")
            gender = st.selectbox("Gender", ["Male", "Female"], label_visibility="collapsed")
            
            st.markdown("**👥 Customer Type**")
            customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"], label_visibility="collapsed")
        
        with col2:
            st.markdown("**🎂 Age**")
            age = st.number_input("Age", min_value=7, max_value=85, value=35, label_visibility="collapsed")
            
            st.markdown("**✈️ Travel Type**")
            type_of_travel = st.selectbox("Travel Type", ["Business travel", "Personal Travel"], label_visibility="collapsed")
        
        with col3:
            st.markdown("**🎫 Flight Class**")
            flight_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"], label_visibility="collapsed")
            
            st.markdown("**📏 Flight Distance (miles)**")
            flight_distance = st.number_input("Flight Distance (miles)", min_value=31, max_value=5000, value=1000, label_visibility="collapsed")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Delays section
        st.markdown("""
        <div style="
            background: white;
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        ">
            <h3 style="color: #667eea; margin-bottom: 20px; text-align: center;">
                ⏰ Delay Information
            </h3>
        """, unsafe_allow_html=True)
        
        delay_col1, delay_col2 = st.columns(2)
        
        with delay_col1:
            st.markdown("**🛫 Departure Delay (minutes)**")
            departure_delay = st.number_input("Departure Delay (minutes)", min_value=0, max_value=1592, value=0, label_visibility="collapsed")
        
        with delay_col2:
            st.markdown("**🛬 Arrival Delay (minutes)**")
            arrival_delay = st.number_input("Arrival Delay (minutes)", min_value=0, max_value=1584, value=0, label_visibility="collapsed")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Service ratings section
        st.markdown("""
        <div style="
            background: white;
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        ">
            <h3 style="color: #667eea; margin-bottom: 20px; text-align: center;">
                📊 Service Ratings (scale 1-5)
            </h3>
            <p style="text-align: center; color: #666; margin-bottom: 25px;">
                1 = Very Poor | 2 = Poor | 3 = Fair | 4 = Good | 5 = Excellent
            </p>
        """, unsafe_allow_html=True)
        
        # Primary service ratings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📶 WiFi Service**")
            wifi = st.slider("WiFi Service", 1, 5, 3, label_visibility="collapsed")
            
            st.markdown("**💺 Seat Comfort**")
            seat_comfort = st.slider("Seat Comfort", 1, 5, 3, label_visibility="collapsed")
        
        with col2:
            st.markdown("**🌐 Online Boarding**")
            online_boarding = st.slider("Online Boarding", 1, 5, 3, label_visibility="collapsed")
            
            st.markdown("**🎬 Entertainment**")
            entertainment = st.slider("Entertainment", 1, 5, 3, label_visibility="collapsed")
        
        with col3:
            st.markdown("**🧽 Cleanliness**")
            cleanliness = st.slider("Cleanliness", 1, 5, 3, label_visibility="collapsed")
            
            st.markdown("**🍽️ Food and Drink**")
            food_drink = st.slider("Food and Drink", 1, 5, 3, label_visibility="collapsed")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Additional services
        with st.expander("📋 Additional Services (Click to expand)", expanded=False):
            col4, col5 = st.columns(2)
            
            with col4:
                st.markdown("**⏰ Schedule Convenience**")
                time_convenient = st.slider("Departure/Arrival Times", 1, 5, 3, label_visibility="collapsed")
                
                st.markdown("**💻 Online Booking**")
                online_booking = st.slider("Ease of Online Booking", 1, 5, 3, label_visibility="collapsed")
                
                st.markdown("**🚪 Gate Location**")
                gate_location = st.slider("Gate Location", 1, 5, 3, label_visibility="collapsed")
                
                st.markdown("**👨‍✈️ Onboard Service**")
                onboard_service = st.slider("Onboard Service", 1, 5, 3, label_visibility="collapsed")
            
            with col5:
                st.markdown("**🦵 Leg Room**")
                leg_room = st.slider("Leg Room Space", 1, 5, 3, label_visibility="collapsed")
                
                st.markdown("**🧳 Baggage Handling**")
                baggage = st.slider("Baggage Handling", 1, 5, 3, label_visibility="collapsed")
                
                st.markdown("**✅ Check-in Service**")
                checkin = st.slider("Check-in Service", 1, 5, 3, label_visibility="collapsed")
                
                st.markdown("**✈️ Inflight Service**")
                inflight_service = st.slider("Inflight Service", 1, 5, 3, label_visibility="collapsed")
        
        # Submit button
        st.markdown('<div style="margin: 30px 0; text-align: center;">', unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮 Predict Satisfaction", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if submitted and artifacts:
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
            
            # Make prediction using our clean function
            with st.spinner("🤖 Analyzing passenger data..."):
                result = predict_satisfaction_clean(passenger_data, artifacts)
                
                if result['prediction'] != 'error':
                    # Show missing features info if any     
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<h2 style="text-align: center; color: #667eea;">📊 Prediction Results</h2>', unsafe_allow_html=True)
                    
                    # Main result display
                    result_class = "satisfied" if result['prediction'] == 'satisfied' else "neutral_or_dissatisfied"
                    color = "#4CAF50" if result['prediction'] == 'satisfied' else "#F44336"
                    icon = "✅" if result['prediction'] == 'satisfied' else "❌"
                    
                    st.markdown(f"""
                    <div class="prediction-result" style="
                        padding: 40px; 
                        border-radius: 20px; 
                        border-left: 10px solid {color}; 
                        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                        margin: 30px 0; 
                        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                        text-align: center;">
                        <h1 style="
                            margin: 0; 
                            color: {color}; 
                            font-size: 3em; 
                            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
                            animation: pulse 2s infinite;">
                            {icon} {'SATISFIED' if result['prediction'] == 'satisfied' else 'NEUTRAL OR DISSATISFIED'}
                        </h1>
                        <h2 style="
                            margin: 20px 0; 
                            color: #333; 
                            font-size: 1.8em;">
                            🎯 Probability: {result['probability']:.1%}
                        </h2>
                        <div style="
                            display: flex; 
                            justify-content: center; 
                            gap: 30px; 
                            margin-top: 25px;">
                            <div class="metric-card">
                                <h4 style="margin: 0; font-size: 1.1em;">Confidence</h4>
                                <p style="margin: 5px 0 0 0; font-size: 1.3em; font-weight: bold;">
                                    {result['confidence'].upper()}
                                </p>
                            </div>
                            <div class="metric-card">
                                <h4 style="margin: 0; font-size: 1.1em;">Features Used</h4>
                                <p style="margin: 5px 0 0 0; font-size: 1.3em; font-weight: bold;">
                                    {"45"}
                                </p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability gauge
                    st.markdown('<h2 style="text-align: center; color: #667eea; margin-top: 40px;">📈 Probability Gauge</h2>', unsafe_allow_html=True)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Indicator(
                        mode = "gauge+number+delta",
                        value = result['probability'] * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Satisfaction Probability (%)", 'font': {'size': 24, 'color': '#667eea'}},
                        delta = {'reference': 50, 'increasing': {'color': "#4CAF50"}, 'decreasing': {'color': "#F44336"}},
                        number = {'font': {'size': 48, 'color': color}},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 3, 'tickcolor': "#667eea"},
                            'bar': {'color': color, 'thickness': 0.4},
                            'bgcolor': "rgba(255,255,255,0.8)",
                            'borderwidth': 4,
                            'bordercolor': "#667eea",
                            'steps': [
                                {'range': [0, 20], 'color': "#ffcdd2"},
                                {'range': [20, 40], 'color': "#ffecb3"},
                                {'range': [40, 60], 'color': "#fff3e0"},
                                {'range': [60, 80], 'color': "#e8f5e8"},
                                {'range': [80, 100], 'color': "#c8e6c9"}
                            ],
                            'threshold': {
                                'line': {'color': "#FF6B6B", 'width': 8},
                                'thickness': 0.8,
                                'value': 90
                            }
                        }
                    ))
                    
                    fig.update_layout(
                        height=600,
                        font={'color': "#667eea"},
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Generate insights
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<h2 style="text-align: center; color: #667eea;">🔍 Key Insights</h2>', unsafe_allow_html=True)
                    
                    # Generate smart insights based on input
                    insights = []
                    
                    # Service quality analysis
                    service_ratings = [wifi, online_boarding, seat_comfort, entertainment, cleanliness, food_drink,
                                     time_convenient, online_booking, gate_location, onboard_service, 
                                     leg_room, baggage, checkin, inflight_service]
                    avg_rating = sum(service_ratings) / len(service_ratings)
                    
                    if avg_rating >= 4.2:
                        insights.append({
                            'icon': '🌟',
                            'title': 'Excellent Service Quality',
                            'text': f'Average service rating of {avg_rating:.1f}/5 indicates premium experience.',
                            'color': '#4CAF50'
                        })
                    elif avg_rating <= 2.5:
                        insights.append({
                            'icon': '⚠️',
                            'title': 'Service Quality Concern',
                            'text': f'Low average rating of {avg_rating:.1f}/5 may impact satisfaction.',
                            'color': '#F44336'
                        })
                    
                    # Delay analysis
                    total_delay = departure_delay + arrival_delay
                    if total_delay > 60:
                        insights.append({
                            'icon': '🚨',
                            'title': 'Significant Delays',
                            'text': f'Total delay of {total_delay} minutes significantly impacts satisfaction.',
                            'color': '#F44336'
                        })
                    elif total_delay == 0:
                        insights.append({
                            'icon': '⏰',
                            'title': 'Perfect Timing',
                            'text': 'On-time performance enhances passenger satisfaction.',
                            'color': '#4CAF50'
                        })
                    
                    # Customer type insight
                    if customer_type == "Loyal Customer":
                        insights.append({
                            'icon': '⭐',
                            'title': 'Loyal Customer',
                            'text': 'Loyal customers are more tolerant but expect consistent quality.',
                            'color': '#FF9800'
                        })
                    
                    # Class analysis
                    if flight_class == "Business":
                        insights.append({
                            'icon': '💼',
                            'title': 'Business Class Expectations',
                            'text': 'Business class passengers expect premium service across all touchpoints.',
                            'color': '#2196F3'
                        })
                    
                    # WiFi specific insight (important feature)
                    if wifi <= 2:
                        insights.append({
                            'icon': '📶',
                            'title': 'WiFi Concern',
                            'text': 'Poor WiFi service is a critical factor affecting modern traveler satisfaction.',
                            'color': '#F44336'
                        })
                    elif wifi >= 4:
                        insights.append({
                            'icon': '📶',
                            'title': 'Strong Connectivity',
                            'text': 'Good WiFi service significantly enhances passenger experience.',
                            'color': '#4CAF50'
                        })
                    
                    # Display insights
                    for i, insight in enumerate(insights):
                        st.markdown(f"""
                        <div style="
                            border-left: 6px solid {insight['color']};
                            background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,249,250,0.95) 100%);
                            padding: 20px;
                            margin: 15px 0;
                            border-radius: 12px;
                            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                            animation: slideInFromLeft 0.6s ease-out;
                            animation-delay: {i * 0.15}s;
                        ">
                            <div style="display: flex; align-items: center; gap: 20px;">
                                <div style="font-size: 2.5em;">{insight['icon']}</div>
                                <div>
                                    <h4 style="margin: 0 0 5px 0; color: {insight['color']}; font-size: 1.2em;">
                                        {insight['title']}
                                    </h4>
                                    <p style="margin: 0; color: #333; line-height: 1.5;">
                                        {insight['text']}
                                    </p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Store prediction in session state
                    prediction_record = {
                        'timestamp': datetime.now(),
                        'prediction': result['prediction'],
                        'probability': result['probability'],
                        'passenger_data': passenger_data
                    }
                    st.session_state.predictions_history.append(prediction_record)
                    
                    # Executive Summary
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    
                    prediction_text = 'SATISFIED' if result['prediction'] == 'satisfied' else 'NOT SATISFIED'
                    probability_text = f"{result['probability']:.1%}"
                    confidence_text = result['confidence'].upper()
                    features_text = str(result['features_used'])
                    
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 35px;
                        border-radius: 20px;
                        color: white;
                        margin: 40px 0;
                        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
                    ">
                        <h2 style="margin: 0 0 20px 0; color: white; text-align: center; font-size: 2em;">
                            📋 Executive Summary
                        </h2>
                        <div style="
                            display: grid; 
                            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                            gap: 20px; 
                            margin-top: 25px;
                        ">
                            <div style="text-align: center; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px;">
                                <h4 style="margin: 0 0 5px 0; color: white;">Result</h4>
                                <p style="margin: 0; font-size: 1.2em; font-weight: bold; color: white;">
                                    {prediction_text}
                                </p>
                            </div>
                            <div style="text-align: center; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px;">
                                <h4 style="margin: 0 0 5px 0; color: white;">Probability</h4>
                                <p style="margin: 0; font-size: 1.2em; font-weight: bold; color: white;">
                                    {probability_text}
                                </p>
                            </div>
                            <div style="text-align: center; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px;">
                                <h4 style="margin: 0 0 5px 0; color: white;">Confidence</h4>
                                <p style="margin: 0; font-size: 1.2em; font-weight: bold; color: white;">
                                    {confidence_text}
                                </p>
                            </div>
                            <div style="text-align: center; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px;">
                                <h4 style="margin: 0 0 5px 0; color: white;">Features</h4>
                                <p style="margin: 0; font-size: 1.2em; font-weight: bold; color: white;">
                                    {features_text}
                                </p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Error handling
                    st.error("❌ Prediction failed. Please check your inputs and try again.")
                    if 'error' in result:
                        with st.expander("🔍 Error Details"):
                            st.code(result['error'])

def render_batch_prediction_page(artifacts):
    """Render the batch prediction interface - COMPLETE VERSION"""
    
    # Add CSS for batch processing
    st.markdown("""
    <style>
    .batch-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 4s ease infinite;
        background-size: 200% 200%;
        text-align: center;
    }
    
    .upload-area {
        border: 3px dashed #667eea;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #4ECDC4;
        background: linear-gradient(135deg, rgba(78, 205, 196, 0.1) 0%, rgba(255, 107, 107, 0.1) 100%);
        transform: scale(1.02);
    }
    
    @keyframes bounceIn {
        0% { transform: scale(0.3); opacity: 0; }
        50% { transform: scale(1.05); }
        70% { transform: scale(0.9); }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .success-card {
        animation: bounceIn 0.8s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="batch-header">📊 Batch Prediction Processing</h1>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Information card
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        margin: 25px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    ">
        <h2 style="margin: 0 0 15px 0; color: white; text-align: center;">
            🚀 High Performance Batch Processing
        </h2>
        <p style="margin: 0; font-size: 1.2em; text-align: center; line-height: 1.6;">
            Upload a CSV file with passenger data to get quick mass predictions. 
            Optimized to efficiently process hundreds or thousands of records.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    st.markdown('<h2 style="text-align: center; color: #667eea; margin-top: 40px;">📁 Upload Your Data</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="upload-area">
        <h3 style="color: #667eea; margin-bottom: 20px;">
            📎 Drag and drop or click to select
        </h3>
        <p style="color: #666; margin: 0;">
            Your CSV should contain columns for demographics, flight details and service ratings
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file with passenger data", 
        type="csv",
        help="Your CSV should contain columns for demographics, flight details and service ratings",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.markdown(f"""
            <div class="success-card" style="
                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                padding: 25px;
                border-radius: 15px;
                color: white;
                text-align: center;
                margin: 25px 0;
                box-shadow: 0 8px 25px rgba(76, 175, 80, 0.3);
            ">
                <h2 style="margin: 0 0 10px 0; color: white;">
                    ✅ File Loaded Successfully!
                </h2>
                <p style="margin: 0; font-size: 1.3em; font-weight: bold;">
                    📊 {len(df):,} records ready for processing
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Data preview
            col1, col2 = st.columns([2, 1])
            
            with col1:
                with st.expander("📋 Data Preview", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 25px;
                    border-radius: 15px;
                    color: white;
                    text-align: center;
                ">
                    <h3 style="margin: 0 0 15px 0; color: white;">📊 Data Info</h3>
                    <p style="margin: 5px 0; color: white;"><strong>Rows:</strong> {len(df):,}</p>
                    <p style="margin: 5px 0; color: white;"><strong>Columns:</strong> {len(df.columns)}</p>
                    <p style="margin: 5px 0; color: white;"><strong>Size:</strong> {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Process button
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="text-align: center; margin: 30px 0;">
                <h2 style="color: #667eea; margin-bottom: 20px;">🚀 Ready to Process?</h2>
                <p style="color: #666; font-size: 1.1em;">
                    Click the button below to start batch processing your data
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Start Batch Processing", use_container_width=True, type="primary"):
                if artifacts:
                    start_time = time.time()
                    
                    # Show progress
                    progress_container = st.container()
                    with progress_container:
                        st.markdown("""
                        <div style="
                            background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
                            padding: 15px;
                            border-radius: 10px;
                            color: white;
                            text-align: center;
                            margin: 10px 0;
                        ">
                            <h4 style="margin: 0; color: white;">🔄 Processing batch...</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                    
                    # Update progress
                    for i in range(10):
                        progress_bar.progress((i + 1) / 10)
                        status_text.text(f"Processing... {(i + 1) * 10}%")
                        time.sleep(0.1)
                    
                    # Process batch using our clean function
                    results = predict_batch_clean(df, artifacts)
                    
                    # Add results to dataframe
                    predictions = [r['prediction'] for r in results]
                    probabilities = [r['probability'] for r in results]
                    confidences = [r['confidence'] for r in results]
                    
                    df['prediction'] = predictions
                    df['satisfaction_probability'] = probabilities
                    df['confidence'] = confidences
                    
                    total_time = time.time() - start_time
                    
                    # Clear progress
                    progress_container.empty()
                    
                    # Success message
                    st.markdown(f"""
                    <div class="success-card" style="
                        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                        padding: 40px;
                        border-radius: 20px;
                        color: white;
                        text-align: center;
                        margin: 30px 0;
                        box-shadow: 0 15px 35px rgba(76, 175, 80, 0.4);
                    ">
                        <h1 style="margin: 0 0 20px 0; color: white; font-size: 2.5em;">
                            🎉 Processing Completed!
                        </h1>
                        <div style="display: flex; justify-content: center; gap: 40px; margin-top: 25px;">
                            <div>
                                <h3 style="margin: 0; color: white;">📊 Predictions</h3>
                                <p style="margin: 5px 0 0 0; font-size: 2em; font-weight: bold; color: white;">
                                    {len(df):,}
                                </p>
                            </div>
                            <div>
                                <h3 style="margin: 0; color: white;">⏱️ Time</h3>
                                <p style="margin: 5px 0 0 0; font-size: 2em; font-weight: bold; color: white;">
                                    {total_time:.2f}s
                                </p>
                            </div>
                            <div>
                                <h3 style="margin: 0; color: white;">⚡ Speed</h3>
                                <p style="margin: 5px 0 0 0; font-size: 2em; font-weight: bold; color: white;">
                                    {len(df)/total_time:.0f}/s
                                </p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Results summary
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<h1 style="text-align: center; color: #667eea;">📈 Results Summary</h1>', unsafe_allow_html=True)
                    
                    # Calculate metrics
                    satisfied_count = sum(1 for p in predictions if p == 'satisfied')
                    dissatisfied_count = sum(1 for p in predictions if p == 'neutral or dissatisfied')
                    error_count = sum(1 for p in predictions if p == 'error')
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                            padding: 25px;
                            border-radius: 15px;
                            color: white;
                            text-align: center;
                            box-shadow: 0 8px 25px rgba(76, 175, 80, 0.3);
                        ">
                            <h3 style="margin: 0 0 10px 0; color: white;">😊 Satisfied</h3>
                            <p style="margin: 0; font-size: 2.5em; font-weight: bold; color: white;">
                                {satisfied_count:,}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
                            padding: 25px;
                            border-radius: 15px;
                            color: white;
                            text-align: center;
                            box-shadow: 0 8px 25px rgba(255, 152, 0, 0.3);
                        ">
                            <h3 style="margin: 0 0 10px 0; color: white;">😐 Not Satisfied</h3>
                            <p style="margin: 0; font-size: 2.5em; font-weight: bold; color: white;">
                                {dissatisfied_count:,}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        satisfaction_rate = satisfied_count / (satisfied_count + dissatisfied_count) * 100 if (satisfied_count + dissatisfied_count) > 0 else 0
                        
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
                            padding: 25px;
                            border-radius: 15px;
                            color: white;
                            text-align: center;
                            box-shadow: 0 8px 25px rgba(33, 150, 243, 0.3);
                        ">
                            <h3 style="margin: 0 0 10px 0; color: white;">📊 Satisfaction Rate</h3>
                            <p style="margin: 0; font-size: 2.5em; font-weight: bold; color: white;">
                                {satisfaction_rate:.1f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        valid_probabilities = [p for p in probabilities if p > 0]
                        avg_confidence = sum(valid_probabilities) / len(valid_probabilities) if valid_probabilities else 0
                        
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #9C27B0 0%, #7B1FA2 100%);
                            padding: 25px;
                            border-radius: 15px;
                            color: white;
                            text-align: center;
                            box-shadow: 0 8px 25px rgba(156, 39, 176, 0.3);
                        ">
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Visualizations
                    if satisfied_count > 0 or dissatisfied_count > 0:
                        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                        st.markdown('<h2 style="text-align: center; color: #667eea;">📊 Results Visualizations</h2>', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Pie chart
                            fig_pie = px.pie(
                                values=[satisfied_count, dissatisfied_count],
                                names=['Satisfied', 'Not Satisfied'],
                                title="Satisfaction Distribution",
                                color_discrete_map={'Satisfied': '#4CAF50', 'Not Satisfied': '#FF9800'},
                                hole=0.4
                            )
                            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                            fig_pie.update_layout(height=450, showlegend=True)
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            # Probability distribution
                            valid_probs = [p for p in probabilities if p > 0]
                            if valid_probs:
                                fig_hist = px.histogram(
                                    x=valid_probs,
                                    title="Probability Distribution",
                                    labels={'x': 'Satisfaction Probability', 'y': 'Count'},
                                    nbins=25,
                                    color_discrete_sequence=['#667eea']
                                )
                                fig_hist.update_layout(height=450, bargap=0.1)
                                st.plotly_chart(fig_hist, use_container_width=True)
                            else:
                                st.markdown("""
                                <div style="
                                    text-align: center;
                                    padding: 60px;
                                    color: #666;
                                    background: #f8f9fa;
                                    border-radius: 10px;
                                    margin: 20px 0;
                                ">
                                    <h3>📊 No Valid Data</h3>
                                    <p>No valid probabilities to display histogram</p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Sample results
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<h2 style="text-align: center; color: #667eea;">🔍 Sample Results</h2>', unsafe_allow_html=True)
                    
                    # Show sample with formatted results
                    display_df = df.head(10).copy()
                    
                    # Format columns for display
                    if 'prediction' in display_df.columns:
                        display_df['prediction'] = display_df['prediction'].map({
                            'satisfied': '😊 Satisfied',
                            'neutral or dissatisfied': '😐 Not Satisfied',
                            'error': '❌ Error'
                        })
                    
                    if 'satisfaction_probability' in display_df.columns:
                        display_df['satisfaction_probability'] = display_df['satisfaction_probability'].apply(
                            lambda x: f"{x:.1%}" if x > 0 else "N/A"
                        )
                    
                    if 'confidence' in display_df.columns:
                        display_df['confidence'] = display_df['confidence'].map({
                            'high': '🎯 High',
                            'medium': '📊 Medium',
                            'low': '📉 Low',
                            'none': '❌ N/A'
                        })
                    
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Download section
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<h2 style="text-align: center; color: #667eea;">📥 Download Results</h2>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        <div style="
                            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                            padding: 25px;
                            border-radius: 15px;
                            margin: 10px 0;
                            text-align: center;
                            color: white;
                        ">
                            <h3 style="margin: 0 0 10px 0; color: white;">📊 Complete Results</h3>
                            <p style="margin: 0; font-size: 0.9em; opacity: 0.9;">
                                Original data + predictions + probabilities
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        csv_full = df.to_csv(index=False)
                        st.download_button(
                            label="📊 Download Complete Results",
                            data=csv_full,
                            file_name=f"predictions_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime='text/csv',
                            use_container_width=True
                        )
                    
                    with col2:
                        st.markdown("""
                        <div style="
                            background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
                            padding: 25px;
                            border-radius: 15px;
                            margin: 10px 0;
                            text-align: center;
                            color: white;
                        ">
                            <h3 style="margin: 0 0 10px 0; color: white;">📋 Summary Only</h3>
                            <p style="margin: 0; font-size: 0.9em; opacity: 0.9;">
                                Only predictions and probabilities
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        summary_df = df[['prediction', 'satisfaction_probability', 'confidence']].copy()
                        csv_summary = summary_df.to_csv()
                        st.download_button(
                            label="📋 Download Summary Only",
                            data=csv_summary,
                            file_name=f"predictions_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime='text/csv',
                            use_container_width=True
                        )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            with st.expander("🔍 Error Details"):
                st.code(str(e))
                st.write("**Tips:**")
                st.write("- Make sure your CSV has the necessary columns")
                st.write("- Check for special characters in column names")
                st.write("- Confirm data format consistency")
    
    else:
        # Enhanced sample data format section
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<h2 style="text-align: center; color: #667eea;">📝 Expected Data Format</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 20px;
            color: white;
            text-align: center;
            margin: 25px 0;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        ">
            <h3 style="margin: 0 0 15px 0; color: white;">
                📋 Recommended CSV Structure
            </h3>
            <p style="margin: 0; font-size: 1.1em; line-height: 1.6;">
                For best predictions, your CSV file should include all service rating 
                columns and passenger demographic information.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced sample data
        sample_data = {
            'Gender': ['Male', 'Female', 'Male'],
            'Age': [35, 28, 45],
            'Customer Type': ['Loyal Customer', 'disloyal Customer', 'Loyal Customer'],
            'Type of Travel': ['Business travel', 'Personal Travel', 'Business travel'],
            'Class': ['Business', 'Eco', 'Eco Plus'],
            'Flight Distance': [1000, 500, 2000],
            'Inflight wifi service': [4, 2, 5],
            'Seat comfort': [4, 3, 5],
            'Food and drink': [3, 2, 4],
            '...': ['...', '...', '...']
        }
        
        sample_df = pd.DataFrame(sample_data)
        
        st.markdown("""
        <div style="
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            margin: 25px 0;
        ">
        """, unsafe_allow_html=True)
        
        st.dataframe(sample_df, use_container_width=True, hide_index=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Enhanced tip section
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
            padding: 25px;
            border-radius: 15px;
            color: white;
            margin: 25px 0;
            box-shadow: 0 8px 25px rgba(78, 205, 196, 0.3);
        ">
            <h3 style="margin: 0 0 15px 0; color: white;">💡 Tips for Best Results</h3>
            <ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
                <li>Include all service rating columns (1-5 scale)</li>
                <li>Ensure demographic data is complete</li>
                <li>Check for no blank values in main columns</li>
                <li>Use the same column names shown in the example above</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)