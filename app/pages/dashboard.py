"""
Dashboard Module for Airline Passenger Satisfaction
Main dashboard with visualizations, summaries, and overview metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

def render_main_dashboard(artifacts):
    """Render the main dashboard with overview and key metrics"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>✈️ Airline Passenger Satisfaction Dashboard</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem;">
            Advanced ML system powered by CatBoost Enhanced with 45 engineered features
        </p>
        <div style="background: rgba(255,255,255,0.15); padding: 1rem; border-radius: 10px; margin-top: 1.5rem;">
            <p style="font-size: 1rem; margin: 0; font-weight: 500;">
                🎓 <strong>Portfolio Demonstration</strong> | Production-Ready ML with Feature Engineering
            </p>
            <p style="font-size: 0.9rem; margin: 0.5rem 0; opacity: 0.9;">
                High performance (96.4% accuracy, 99.5% ROC AUC) with comprehensive feature engineering.
                This project showcases advanced ML engineering and automated feature creation.
            </p>
            <p style="font-size: 0.85rem; margin: 0; background: rgba(255,255,255,0.25); padding: 0.5rem; border-radius: 5px; border-left: 3px solid #FFD700;">
                <strong>🤖 Model:</strong> CatBoost Enhanced - 45 features including 23 engineered features for superior performance.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model status section
    render_model_status_section(artifacts)
    
    # Key metrics overview
    render_key_metrics_overview()
    
    # Recent predictions overview
    render_recent_predictions_overview()
    
    # Quick insights section
    render_quick_insights_section()

def render_model_status_section(artifacts):
    """Render model status and information section"""
    st.header("🤖 Model Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if artifacts:
        with col1:
            st.metric(
                "Model Status", 
                "✅ Active",
                help="CatBoost model is loaded and ready for predictions"
            )
        
        with col2:
            st.metric(
                "Algorithm", 
                "CatBoost",
                help="Gradient boosting algorithm optimized for categorical features"
            )
        
        with col3:
            model_accuracy = artifacts.get('model_info', {}).get('performance_metrics', {}).get('validation_accuracy', 0.9636)
            st.metric(
                "Accuracy", 
                f"{model_accuracy:.1%}",
                help="Model performance on validation dataset"
            )
        
        with col4:
            n_features = artifacts.get('model_info', {}).get('n_features', 45)
            n_engineered = artifacts.get('model_info', {}).get('n_engineered_features', 23)
            st.metric(
                "Features", 
                f"{n_features} (+{n_engineered} eng.)",
                help="Total features including engineered features"
            )
    else:
        with col1:
            st.metric("Model Status", "❌ Not Loaded")
        with col2:
            st.metric("Algorithm", "N/A")
        with col3:
            st.metric("Accuracy", "N/A")
        with col4:
            st.metric("Features", "N/A")

def render_key_metrics_overview():
    """Render key performance metrics overview"""
    st.header("📊 Key Performance Metrics")
    
    # Performance metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin: 0; color: #1f77b4;">Accuracy</h3>
            <h2 style="margin: 0.5rem 0; color: #1f77b4;">96.4%</h2>
            <p style="margin: 0; font-size: 0.9rem; color: #666;">Overall prediction accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin: 0; color: #ff7f0e;">F1 Score</h3>
            <h2 style="margin: 0.5rem 0; color: #ff7f0e;">95.8%</h2>
            <p style="margin: 0; font-size: 0.9rem; color: #666;">Balanced performance metric</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin: 0; color: #2ca02c;">ROC AUC</h3>
            <h2 style="margin: 0.5rem 0; color: #2ca02c;">99.5%</h2>
            <p style="margin: 0; font-size: 0.9rem; color: #666;">Area under ROC curve</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin: 0; color: #d62728;">Features</h3>
            <h2 style="margin: 0.5rem 0; color: #d62728;">45</h2>
            <p style="margin: 0; font-size: 0.9rem; color: #666;">Including 23 engineered features</p>
        </div>
        """, unsafe_allow_html=True)

def render_recent_predictions_overview():
    """Render overview of recent predictions"""
    st.header("🔮 Recent Predictions Overview")
    
    if 'predictions_history' in st.session_state and st.session_state.predictions_history:
        history_df = pd.DataFrame(st.session_state.predictions_history)
        
        # Recent stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_predictions = len(history_df)
            st.metric("Total Predictions", total_predictions)
        
        with col2:
            satisfied_count = sum(1 for p in history_df['prediction'] if p == 'satisfied')
            satisfaction_rate = (satisfied_count / total_predictions * 100) if total_predictions > 0 else 0
            st.metric("Satisfaction Rate", f"{satisfaction_rate:.1f}%")
        
        with col3:
            avg_confidence = history_df['probability'].mean() if len(history_df) > 0 else 0
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        # Predictions timeline
        if len(history_df) > 1:
            st.subheader("📈 Predictions Timeline")
            
            # Prepare data for timeline
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df = history_df.sort_values('timestamp')
            
            # Line chart of probabilities over time
            fig_timeline = px.line(
                history_df,
                x='timestamp',
                y='probability',
                title="Satisfaction Probabilities Over Time",
                labels={'probability': 'Satisfaction Probability', 'timestamp': 'Time'},
                color_discrete_sequence=['#1f77b4']
            )
            fig_timeline.add_hline(y=0.5, line_dash="dash", line_color="red", 
                                 annotation_text="Satisfaction Threshold")
            fig_timeline.update_layout(height=400)
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Distribution of predictions
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart of satisfaction distribution
                satisfied_count = sum(1 for p in history_df['prediction'] if p == 'satisfied')
                unsatisfied_count = len(history_df) - satisfied_count
                
                fig_pie = px.pie(
                    values=[satisfied_count, unsatisfied_count],
                    names=['Satisfied', 'Unsatisfied'],
                    title="Satisfaction Distribution",
                    color_discrete_map={'Satisfied': '#4CAF50', 'Unsatisfied': '#F44336'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Histogram of probability distribution
                fig_hist = px.histogram(
                    history_df,
                    x='probability',
                    nbins=20,
                    title="Probability Distribution",
                    labels={'probability': 'Satisfaction Probability', 'count': 'Frequency'}
                )
                fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red",
                                 annotation_text="Threshold")
                st.plotly_chart(fig_hist, use_container_width=True)
    
    else:
        st.info("""
        📝 **No predictions yet!** 
        
        Start making predictions to see insights and trends here. You can:
        - Make individual predictions using the prediction form
        - Upload a CSV file for batch processing
        - View model explanations and feature importance
        """)

def render_quick_insights_section():
    """Render quick insights and tips section"""
    st.header("💡 Quick Insights & Tips")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Model Insights
        
        **Top Predictive Factors:**
        - **WiFi Service Quality** - Most influential feature
        - **Online Boarding Experience** - Critical for satisfaction
        - **Seat Comfort** - Physical comfort matters
        - **Entertainment Quality** - Keeps passengers happy
        - **Customer Loyalty** - Loyal customers more forgiving
        
        **Key Patterns:**
        - Business travelers have higher expectations
        - Longer flights require better entertainment
        - Delays significantly impact satisfaction
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Usage Tips
        
        **For Best Results:**
        - Use balanced mode to see non-WiFi influenced predictions
        - Compare both modes to understand WiFi impact
        - Focus on service ratings for actionable insights
        - Consider customer type and travel purpose
        
        **Interpretation Guide:**
        - **Probability > 80%**: High confidence prediction
        - **Probability 50-80%**: Moderate confidence
        - **Probability < 50%**: Unsatisfied prediction
        """)
    
    # Feature importance quick view
    st.subheader("🔍 Feature Importance Quick View")
    
    # Sample feature importance data (replace with actual from model)
    sample_features = {
        'WiFi Service': 0.35,
        'Online Boarding': 0.18,
        'Seat Comfort': 0.12,
        'Entertainment': 0.10,
        'Customer Type': 0.08,
        'Flight Class': 0.07,
        'Food & Drink': 0.05,
        'Others': 0.05
    }
    
    # Create a horizontal bar chart
    fig_importance = px.bar(
        x=list(sample_features.values()),
        y=list(sample_features.keys()),
        orientation='h',
        title="Top Features by Importance",
        labels={'x': 'Importance Score', 'y': 'Feature'},
        color=list(sample_features.values()),
        color_continuous_scale='viridis'
    )
    fig_importance.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_importance, use_container_width=True)

def render_about_section():
    """Render about section with project information"""
    st.header("📚 About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This is an enterprise-grade machine learning solution for predicting airline passenger satisfaction.
    The system uses CatBoost, a gradient boosting algorithm optimized for categorical features,
    to analyze various factors and predict whether a passenger will be satisfied or unsatisfied 
    with their flight experience.
    
    ### 🔧 Technical Details
    
    - **Algorithm**: CatBoost with optimized hyperparameters
    - **Features**: 20+ engineered features from passenger and flight data
    - **Performance**: 96.5% accuracy on validation set
    - **Infrastructure**: Modular design with separate prediction, monitoring, and analysis modules
    
    ### 📊 Key Features Analyzed
    
    1. **Service Quality**: WiFi, entertainment, food, cleanliness
    2. **Convenience**: Online booking, boarding, check-in
    3. **Comfort**: Seat comfort, leg room, baggage handling
    4. **Punctuality**: Departure and arrival delays
    5. **Demographics**: Age, gender, customer loyalty
    
    ### 🏗️ Architecture
    
    The application is modularized into four main components:
    - **prediction.py**: Individual and batch prediction functionality
    - **dashboard.py**: Main dashboard with visualizations and summaries
    - **monitoring.py**: Model performance monitoring and tracking
    - **shap_analyses.py**: SHAP explainability and feature importance analysis
    
    ### 👨‍💻 Developer
    
    **Pedro M.**
    - 📧 Email: pedrom02.dev@gmail.com
    - 🔗 LinkedIn: [linkedin.com/in/pedrom](https://linkedin.com/in/pedrom)
    - 🐙 GitHub: [@Pedrom2002](https://github.com/Pedrom2002)
    
    ### 📚 Resources
    
    - [GitHub Repository](https://github.com/Pedrom2002/Airline-Passenger-Satisfaction--v2)
    - [Dataset on Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
    - [CatBoost Documentation](https://catboost.ai/docs/)
    """)

# CSS styles for the dashboard
def load_dashboard_styles():
    """Load custom CSS styles for the dashboard"""
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
            margin-bottom: 1rem;
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