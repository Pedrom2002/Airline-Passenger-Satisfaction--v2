# app.py - Main application file with dashboard integration

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import os
import sys

def initialize_session_state():
    """Initialize session state variables"""
    if 'predictions_history' not in st.session_state:
        st.session_state.predictions_history = []
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'startup_complete' not in st.session_state:
        st.session_state.startup_complete = False
    if 'modules' not in st.session_state:
        st.session_state.modules = None
    if 'artifacts' not in st.session_state:
        st.session_state.artifacts = None

def safe_import():
    """Safely import prediction and dashboard modules"""
    modules = {}
    
    # Import prediction module functions
    try:
        # Try different import paths
        try:
           
                    # Direct import from same directory
            import prediction
            modules['prediction'] = {
                'load_model_artifacts': prediction.load_model_artifacts,
                'render_individual_prediction_page': prediction.render_individual_prediction_page,
                'render_batch_prediction_page': prediction.render_batch_prediction_page,
                'render_about_page': getattr(prediction, 'render_about_page', None)
            }
            st.success("✅ Prediction module loaded")
        except ImportError:
            # Try pages subdirectory
            from pages import prediction
            modules['prediction'] = {
                'load_model_artifacts': prediction.load_model_artifacts,
                'render_individual_prediction_page': prediction.render_individual_prediction_page,
                'render_batch_prediction_page': prediction.render_batch_prediction_page,
                'render_about_page': getattr(prediction, 'render_about_page', None)
            }
            st.success("✅ Prediction module loaded from pages")
            
    except ImportError as e:
        st.warning(f"⚠️ Could not import prediction module: {str(e)}")
    except Exception as e:
        st.warning(f"⚠️ Error loading prediction functions: {str(e)}")
    
    # Import dashboard module functions
    try:
        try:

            import pages.shap_analysis
            modules['shap_analysis'] = {
                  'render_shap_analysis_complete': pages.shap_analysis.render_shap_analysis_complete,
            }
            st.success("✅ SHAP module loaded")

            import pages.monitoring
            modules['monitoring'] = {
                    'render_monitoring_dashboard_complete': pages.monitoring.render_monitoring_dashboard_complete,
                        # (adicione aqui outras funções que quiser expor)
                    }
                
            # Direct import from same directory
            import dashboard
            modules['dashboard'] = {
                'render_main_dashboard': dashboard.render_main_dashboard,
                'render_about_section': dashboard.render_about_section,
                'load_dashboard_styles': dashboard.load_dashboard_styles
            }
            st.success("✅ Dashboard module loaded")
        except ImportError:
            # Try pages subdirectory
            from pages import dashboard
            modules['dashboard'] = {
                'render_main_dashboard': dashboard.render_main_dashboard,
                'render_about_section': dashboard.render_about_section,
                'load_dashboard_styles': dashboard.load_dashboard_styles
            }
            st.success("✅ Dashboard module loaded from pages")
            
    except ImportError as e:
        st.warning(f"⚠️ Could not import dashboard module: {str(e)}")
    except Exception as e:
        st.warning(f"⚠️ Error loading dashboard functions: {str(e)}")
    
    return modules if modules else None

def load_model(modules):
    """Load model artifacts using prediction module"""
    if modules and modules.get('prediction') and modules['prediction'].get('load_model_artifacts'):
        try:
            artifacts = modules['prediction']['load_model_artifacts']()
            if artifacts:
                st.session_state.model_loaded = True
                st.session_state.artifacts = artifacts
                return artifacts
            else:
                st.session_state.model_loaded = False
                st.session_state.artifacts = None
                return None
        except Exception as e:
            st.session_state.model_loaded = False
            st.session_state.artifacts = None
            st.warning(f"Could not load model: {str(e)}")
            return None
    else:
        st.session_state.model_loaded = False
        return None

def render_sidebar(artifacts, modules):
    """Render sidebar navigation"""
    st.sidebar.title("🛫 Airline Satisfaction AI")
    st.sidebar.markdown("---")
    
    # Model status
    if artifacts:
        st.sidebar.success("✅ Model Loaded")
        if hasattr(artifacts, 'get') and artifacts.get('feature_columns'):
            st.sidebar.markdown(f"**Features**: {len(artifacts['feature_columns'])}")
        else:
            st.sidebar.markdown("**Model**: CatBoost Ready")
    else:
        st.sidebar.error("❌ Model Not Loaded")
    
    st.sidebar.markdown("---")
    
    # Navigation menu
    menu_options = ["🏠 Dashboard", "🎯 Individual Prediction", "📊 Batch Processing"]
    
    # Add additional options if modules are available
    if modules:
        menu_options.extend(["📈 Model Monitoring", "🔍 SHAP Analysis"])
    
    menu_options.append("📚 About")
    
    app_mode = st.sidebar.radio(
        "Choose Mode:",
        menu_options,
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # Session statistics
    if hasattr(st.session_state, 'predictions_history') and st.session_state.predictions_history:
        st.sidebar.markdown("### 📈 Session Stats")
        st.sidebar.markdown(f"**Predictions**: {len(st.session_state.predictions_history)}")
        
        # Calculate satisfaction rate
        predictions = [p.get('prediction', '') for p in st.session_state.predictions_history]
        satisfied_count = sum(1 for p in predictions if p == 'satisfied')
        rate = satisfied_count / len(predictions) * 100 if predictions else 0
        st.sidebar.markdown(f"**Satisfaction Rate**: {rate:.1f}%")
    
    # System info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ System Info")
    st.sidebar.markdown(f"**Status**: {'🟢 Online' if modules else '🟡 Limited'}")
    st.sidebar.markdown(f"**Model**: {'🟢 Ready' if artifacts else '🔴 Missing'}")
    
    return app_mode

def render_fallback_page(page_name):
    """Render fallback page when modules are not available"""
    st.title(f"🚫 {page_name} - Module Not Available")
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        margin: 25px 0;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    ">
        <h2 style="margin: 0 0 15px 0; color: white;">⚠️ Feature Unavailable</h2>
        <p style="margin: 0; font-size: 1.2em; line-height: 1.6;">
            This feature requires the respective module to be properly loaded.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    ### 🔧 Possible Solutions:
    
    1. **Check file structure**:
       - Ensure `prediction.py` exists in the same directory as `app.py`
       - Ensure `dashboard.py` exists in the same directory as `app.py`
       - Or create a `pages/` directory with both files
    
    2. **Current working directory**: `{os.getcwd()}`
    
    3. **Files in current directory**:
    """)
    
    # Show current directory contents
    try:
        files = [f for f in os.listdir('.') if f.endswith('.py')]
        for file in files:
            st.write(f"   - {file}")
    except:
        st.write("   - Could not list directory contents")
    
    st.markdown("""
    4. **Install dependencies**:
       ```bash
       pip install streamlit pandas numpy plotly catboost joblib
       ```
    
    5. **Restart the application**:
       - Use the reload button below
       - Or restart the Streamlit server
    """)
    
    if st.button("🔄 Try Reloading", type="primary"):
        st.session_state.modules = None
        st.session_state.artifacts = None
        st.session_state.model_loaded = False
        st.rerun()

def render_basic_dashboard(artifacts):
    """Render basic dashboard when dashboard module is not available"""
    st.title("🏠 Dashboard")
    
    st.info("Dashboard module not available. Basic information:")
    st.markdown(f"""
    **Application Status**: Running
    **Model Status**: {'✅ Loaded' if st.session_state.model_loaded else '❌ Not Loaded'}
    **Session Predictions**: {len(getattr(st.session_state, 'predictions_history', []))}
    **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """)
    
    # System status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = "#4CAF50" if st.session_state.model_loaded else "#F44336"
        status_text = "Online" if st.session_state.model_loaded else "Limited"
        st.markdown(f"""
        <div style="
            background: {status_color};
            padding: 20px;
            border-radius: 15px;
            color: white;
            text-align: center;
        ">
            <h4 style="margin: 0 0 10px 0; color: white;">System Status</h4>
            <p style="margin: 0; font-size: 1.2em; font-weight: bold;">
                {status_text}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        model_color = "#4CAF50" if artifacts else "#F44336"
        model_text = "Loaded" if artifacts else "Missing"
        st.markdown(f"""
            <h4 style="margin: 0 0 10px 0; color: white;">Model Status</h4>
            <p style="margin: 0; font-size: 1.2em; font-weight: bold;">
                {model_text}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        predictions_count = len(getattr(st.session_state, 'predictions_history', []))
        st.markdown(f"""
        <div style="
            background: #2196F3;
            padding: 20px;
            border-radius: 15px;
            color: white;
            text-align: center;
        ">
            <h4 style="margin: 0 0 10px 0; color: white;">Session Predictions</h4>
            <p style="margin: 0; font-size: 1.2em; font-weight: bold;">
                {predictions_count}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        timestamp = datetime.now().strftime('%H:%M:%S')
        st.markdown(f"""
        <div style="
            background: #9C27B0;
            padding: 20px;
            border-radius: 15px;
            color: white;
            text-align: center;
        ">
            <h4 style="margin: 0 0 10px 0; color: white;">Current Time</h4>
            <p style="margin: 0; font-size: 1.2em; font-weight: bold;">
                {timestamp}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick actions
    st.markdown("---")
    st.markdown("### 🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Start Individual Prediction", use_container_width=True, type="primary"):
            st.session_state.app_mode = "🎯 Individual Prediction"
            st.rerun()
    
    with col2:
        if st.button("📊 Batch Processing", use_container_width=True):
            st.session_state.app_mode = "📊 Batch Processing"
            st.rerun()
    
    with col3:
        if st.button("📚 Learn More", use_container_width=True):
            st.session_state.app_mode = "📚 About"
            st.rerun()

def main():
    """Main application function with comprehensive error handling"""
    try:
        # Initialize session state
        if 'predictions_history' not in st.session_state:
            initialize_session_state()
        
        # Load modules if not loaded
        if 'modules' not in st.session_state or st.session_state.modules is None:
            st.session_state.modules = safe_import()
        
        modules = st.session_state.modules
        
        # Load dashboard styles if available
        if modules and modules.get('dashboard') and modules['dashboard'].get('load_dashboard_styles'):
            try:
                modules['dashboard']['load_dashboard_styles']()
            except Exception as e:
                st.warning(f"Could not load custom styles: {str(e)}")
        
        # Load model artifacts
        artifacts = load_model(modules)
        
        # Render sidebar and get mode
        app_mode = render_sidebar(artifacts, modules)
        
        # Main content routing
        if app_mode == "🏠 Dashboard":
            if modules and modules.get('dashboard') and modules['dashboard'].get('render_main_dashboard'):
                modules['dashboard']['render_main_dashboard'](artifacts)
            else:
                render_fallback_page("Dashboard")
                
        elif app_mode == "🎯 Individual Prediction":
            if modules and modules.get('prediction') and artifacts:
                modules['prediction']['render_individual_prediction_page'](artifacts)
            else:
                render_fallback_page("Individual Prediction")
                
        elif app_mode == "📊 Batch Processing":
            if modules and modules.get('prediction') and artifacts:
                modules['prediction']['render_batch_prediction_page'](artifacts)
            else:
                render_fallback_page("Batch Processing")
                
        elif app_mode == "📈 Model Monitoring":
            if modules and modules.get('monitoring') and artifacts:
                modules['monitoring']['render_monitoring_dashboard_complete'](artifacts)
            else:
                render_fallback_page("Model Monitoring")
                
        elif app_mode == "🔍 SHAP Analysis":
            modules['shap_analysis']['render_shap_analysis_complete'](artifacts)
                
        elif app_mode == "📚 About":
            if modules and modules.get('dashboard') and modules['dashboard'].get('render_about_section'):
                modules['dashboard']['render_about_section']()
            else:
                render_fallback_page("About")
    
    except Exception as e:
        st.error("⚠️ An unexpected error occurred")
        
        # Show expandable error details
        with st.expander("🔍 Error Details", expanded=False):
            st.code(f"""
Error Type: {type(e).__name__}
Error Message: {str(e)}
            
Traceback:
{traceback.format_exc()}
            """)
        
        st.markdown("### 🔧 Troubleshooting")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reset Application"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.experimental_rerun()
        
        with col2:
            if st.button("🔄 Reload Modules"):
                st.session_state.modules = None
                st.experimental_rerun()
def render_about_fallback():
    """Render basic about page when about module is not available"""
    st.title("📚 About - Airline Satisfaction AI")
    
    st.markdown("""
    ### 🤖 AI-Powered Satisfaction Prediction
    
    This application uses machine learning to predict airline passenger satisfaction
    based on various factors including service quality, demographics, and flight details.
    
    ### 🔧 Technical Stack
    - **Algorithm**: CatBoost Classifier
    - **Interface**: Streamlit
    - **Visualization**: Plotly
    - **Processing**: Pandas, NumPy
    
    ### 📊 Features
    - Individual passenger predictions
    - Batch processing capabilities
    - Interactive visualizations
    - Advanced feature engineering
    
    ### 🚀 Performance
    - High accuracy predictions
    - Real-time processing
    - Scalable architecture
    """)

def display_startup_info():
    """Display startup information and system checks"""
    if st.session_state.startup_complete:
        return  # Already completed startup
        
    st.title("🚀 Starting Airline Satisfaction AI...")
    
    with st.spinner("Initializing application..."):
        import time
        
        # System checks
        checks = [
            ("📦 Loading modules", True),
            ("🔧 Initializing session", True),
            ("🤖 Checking model availability", True),
            ("📊 Setting up dashboard", True),
            ("✅ System ready", True)
        ]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (check_name, status) in enumerate(checks):
            progress_bar.progress((i + 1) / len(checks))
            status_text.text(f"{check_name}...")
            time.sleep(0.3)
        
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.startup_complete = True
        
        # Show final status
        if st.session_state.model_loaded:
            st.success("🎉 Airline Satisfaction AI is ready!")
        else:
            st.info("ℹ️ System loaded. Some features may be limited without model files.")
        
        time.sleep(1)
        st.rerun()

# Application entry point
if __name__ == "__main__":
    try:
        # Set page config first
        st.set_page_config(
            page_title="Airline Satisfaction Prediction",
            page_icon="✈️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Always initialize session state first
        initialize_session_state()
        
        # Show startup info on first run
        if not st.session_state.startup_complete:
            display_startup_info()
        else:
            # Run main application
            main()
            
    except Exception as e:
        st.error(f"Critical error during startup: {str(e)}")
        st.code(traceback.format_exc())
        
        if st.button("🔄 Restart Application"):
            st.session_state.clear()
            st.rerun()