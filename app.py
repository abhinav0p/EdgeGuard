# FILE 7 — app.py
# This is the main Streamlit application that builds a full Neumorphism warm white theme dashboard.
# It simulates on-device predictive maintenance by replaying engine sensor logs cycle-by-cycle,
# running our three AI models in real time, and visualizing the health status.
# It supports an interactive dark/light mode toggle in the sidebar that updates colors live.

import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from data_loader import load_dataset, TEST_FILE
from preprocess import calculate_test_rul, MODELS_DIR, DROPPED_SENSORS

# 1. Set Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="EdgeGuard — Real-Time Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define asset paths
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ANOMALY_PATH = os.path.join(MODELS_DIR, "anomaly_model.pkl")
RUL_PATH = os.path.join(MODELS_DIR, "rul_model.pkl")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "fault_classifier.pkl")

# Initialize dark mode state
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# Sidebar Dark Mode Toggle widget
st.sidebar.markdown("""
<div style="margin-top: 10px; margin-bottom: -10px;">
    <span style="font-size: 12px; color: #7a7060; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">Theme Selector</span>
</div>
""", unsafe_allow_html=True)
dark_mode_toggle = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)

if dark_mode_toggle != st.session_state.dark_mode:
    st.session_state.dark_mode = dark_mode_toggle
    st.rerun()

# 2. Compute dynamic theme colors based on Dark/Light mode state
if st.session_state.dark_mode:
    # --- DARK MODE GLASSMORPHISM CONFIG ---
    # Premium deep violet & dark blue gradient mesh for dark mode background
    bg_gradient = "radial-gradient(at 0% 0%, #1e1b4b 0px, transparent 50%), radial-gradient(at 50% 0%, #311042 0px, transparent 50%), radial-gradient(at 100% 0%, #111827 0px, transparent 50%), radial-gradient(at 0% 100%, #0f172a 0px, transparent 50%), radial-gradient(at 100% 100%, #1e293b 0px, transparent 50%), #0b0f19"
    
    # Glass style properties
    card_color = "rgba(255, 255, 255, 0.08)"
    border_color = "rgba(255, 255, 255, 0.1)"
    sidebar_bg = "rgba(255, 255, 255, 0.12)"
    sidebar_border = "rgba(255, 255, 255, 0.15)"
    
    # Text styles
    text_color = "#f8fafc"
    subtext_color = "#cbd5e1"
    text_shadow = "0 1px 2px rgba(0, 0, 0, 0.4)"
    
    # Buttons
    button_bg = "rgba(255, 255, 255, 0.1)"
    button_border = "rgba(255, 255, 255, 0.15)"
    button_hover_bg = "rgba(255, 255, 255, 0.18)"
    button_hover_border = "rgba(255, 255, 255, 0.25)"
    button_active_bg = "rgba(255, 255, 255, 0.25)"
    
    # Status Alert backgrounds
    alert_green_bg = "rgba(45, 212, 168, 0.15)"
    alert_amber_bg = "rgba(245, 166, 35, 0.15)"
    alert_red_bg = "rgba(255, 107, 107, 0.15)"
    
    # Gauge background circle (transparent)
    gauge_bg_circle = "rgba(255, 255, 255, 0.05)"
    
    # Plotly grid and line colors
    grid_color = "rgba(255,255,255,0.05)"
    line_color_plotly = "rgba(255,255,255,0.1)"
else:
    # --- LIGHT MODE GLASSMORPHISM CONFIG ---
    # Premium subtle gradient mesh with soft blue and lavender tones for light mode background
    bg_gradient = "radial-gradient(at 0% 0%, #e0eafc 0px, transparent 50%), radial-gradient(at 50% 0%, #cfdef3 0px, transparent 50%), radial-gradient(at 100% 0%, #e8eef5 0px, transparent 50%), radial-gradient(at 0% 100%, #e9d5ff 0px, transparent 50%), radial-gradient(at 100% 100%, #bfdbfe 0px, transparent 50%), #e8eef5"
    
    # Glass style properties
    card_color = "rgba(255, 255, 255, 0.25)"
    border_color = "rgba(255, 255, 255, 0.3)"
    sidebar_bg = "rgba(255, 255, 255, 0.35)"
    sidebar_border = "rgba(255, 255, 255, 0.4)"
    
    # Text styles
    text_color = "#1a1a2e"
    subtext_color = "#4a5568"
    text_shadow = "0 1px 2px rgba(255, 255, 255, 0.5)"
    
    # Buttons
    button_bg = "rgba(255, 255, 255, 0.3)"
    button_border = "rgba(255, 255, 255, 0.4)"
    button_hover_bg = "rgba(255, 255, 255, 0.45)"
    button_hover_border = "rgba(255, 255, 255, 0.55)"
    button_active_bg = "rgba(255, 255, 255, 0.55)"
    
    # Status Alert backgrounds
    alert_green_bg = "rgba(45, 212, 168, 0.15)"
    alert_amber_bg = "rgba(245, 166, 35, 0.15)"
    alert_red_bg = "rgba(255, 107, 107, 0.15)"
    
    # Gauge background circle (transparent)
    gauge_bg_circle = "rgba(255, 255, 255, 0.15)"
    
    # Plotly grid and line colors
    grid_color = "rgba(0,0,0,0.05)"
    line_color_plotly = "rgba(0,0,0,0.1)"

# Inject Custom Glassmorphism Dynamic Theme CSS
st.markdown(f"""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Global Styles */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background: {bg_gradient} !important;
        color: {text_color} !important;
        font-family: 'Outfit', 'Inter', sans-serif !important;
        overflow-x: hidden !important;
        max-width: 100% !important;
    }}
    
    /* Ensure all container blocks are transparent so the gradient mesh shows through */
    div[data-testid="stVerticalBlock"], 
    div[data-testid="stHorizontalBlock"],
    div[data-testid="stColumn"],
    div[data-testid="stCustomComponentV1"] {{
        background-color: transparent !important;
    }}
    
    /* Remove default Streamlit padding */
    .main .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-right: 1px solid {sidebar_border} !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15) !important;
    }}
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{
        color: {text_color} !important;
        font-weight: 600;
    }}
    
    /* Glass Card styling */
    .glass-card {{
        background-color: {card_color} !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid {border_color} !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15) !important;
        margin-bottom: 20px !important;
        transition: all 0.3s ease !important;
        color: {text_color} !important;
        text-shadow: {text_shadow} !important;
    }}
    
    /* Anomaly Glowing Card states with smooth pulsing animations */
    @keyframes heartbeat-green {{
        0% {{ box-shadow: 0 0 10px rgba(45, 212, 168, 0.2), 0 8px 32px 0 rgba(31, 38, 135, 0.15); border-color: rgba(45, 212, 168, 0.3); }}
        15% {{ box-shadow: 0 0 25px rgba(45, 212, 168, 0.6), 0 8px 32px 0 rgba(31, 38, 135, 0.15); border-color: rgba(45, 212, 168, 0.7); }}
        30% {{ box-shadow: 0 0 12px rgba(45, 212, 168, 0.2), 0 8px 32px 0 rgba(31, 38, 135, 0.15); border-color: rgba(45, 212, 168, 0.3); }}
        45% {{ box-shadow: 0 0 20px rgba(45, 212, 168, 0.5), 0 8px 32px 0 rgba(31, 38, 135, 0.15); border-color: rgba(45, 212, 168, 0.6); }}
        100% {{ box-shadow: 0 0 10px rgba(45, 212, 168, 0.2), 0 8px 32px 0 rgba(31, 38, 135, 0.15); border-color: rgba(45, 212, 168, 0.3); }}
    }}
    @keyframes flash-red {{
        0%, 100% {{ box-shadow: 0 0 10px rgba(255, 107, 107, 0.2), 0 8px 32px 0 rgba(31, 38, 135, 0.15); border-color: rgba(255, 107, 107, 0.3); }}
        50% {{ box-shadow: 0 0 35px rgba(255, 107, 107, 0.9), 0 8px 32px 0 rgba(31, 38, 135, 0.15); border-color: rgba(255, 107, 107, 1.0); }}
    }}
    @keyframes warning-shake {{
        0%, 100% {{ transform: translateX(0); }}
        10%, 30%, 50%, 70%, 90% {{ transform: translateX(-3px); }}
        20%, 40%, 60%, 80% {{ transform: translateX(3px); }}
    }}
    
    .glow-normal {{
        animation: heartbeat-green 2.5s infinite ease-in-out;
        border: 1px solid rgba(45, 212, 168, 0.3) !important;
        transition: all 0.5s ease;
    }}
    .glow-warning {{
        animation: flash-red 0.8s infinite ease-in-out, warning-shake 0.8s infinite ease-in-out;
        border: 1px solid rgba(255, 107, 107, 0.5) !important;
        transition: all 0.5s ease;
    }}
    .glow-anomaly {{
        animation: flash-red 0.5s infinite ease-in-out, warning-shake 0.5s infinite ease-in-out;
        border: 1px solid rgba(255, 107, 107, 0.7) !important;
        transition: all 0.5s ease;
    }}

    /* Sensor Metric Cards Glow and transitions */
    @keyframes glow-safe-pulse {{
        0% {{ box-shadow: 0 0 8px rgba(45, 212, 168, 0.15), 0 8px 32px 0 rgba(31, 38, 135, 0.15); border-color: rgba(45, 212, 168, 0.2); }}
        50% {{ box-shadow: 0 0 16px rgba(45, 212, 168, 0.4), 0 8px 32px 0 rgba(31, 38, 135, 0.15); border-color: rgba(45, 212, 168, 0.5); }}
        100% {{ box-shadow: 0 0 8px rgba(45, 212, 168, 0.15), 0 8px 32px 0 rgba(31, 38, 135, 0.15); border-color: rgba(45, 212, 168, 0.2); }}
    }}
    @keyframes glow-warning-pulse {{
        0% {{ box-shadow: 0 0 8px rgba(245, 166, 35, 0.15), 0 8px 32px 0 rgba(31, 38, 135, 0.15); border-color: rgba(245, 166, 35, 0.2); }}
        50% {{ box-shadow: 0 0 16px rgba(245, 166, 35, 0.4), 0 8px 32px 0 rgba(31, 38, 135, 0.15); border-color: rgba(245, 166, 35, 0.5); }}
        100% {{ box-shadow: 0 0 8px rgba(245, 166, 35, 0.15), 0 8px 32px 0 rgba(31, 38, 135, 0.15); border-color: rgba(245, 166, 35, 0.2); }}
    }}
    @keyframes glow-critical-pulse {{
        0% {{ box-shadow: 0 0 8px rgba(255, 107, 107, 0.15), 0 8px 32px 0 rgba(31, 38, 135, 0.15); border-color: rgba(255, 107, 107, 0.2); }}
        50% {{ box-shadow: 0 0 18px rgba(255, 107, 107, 0.5), 0 8px 32px 0 rgba(31, 38, 135, 0.15); border-color: rgba(255, 107, 107, 0.6); }}
        100% {{ box-shadow: 0 0 8px rgba(255, 107, 107, 0.15), 0 8px 32px 0 rgba(31, 38, 135, 0.15); border-color: rgba(255, 107, 107, 0.2); }}
    }}
    .sensor-glow-safe {{
        animation: glow-safe-pulse 3.0s infinite ease-in-out;
        border: 1px solid rgba(45, 212, 168, 0.3) !important;
        transition: all 0.5s ease;
    }}
    .sensor-glow-warning {{
        animation: glow-warning-pulse 2.5s infinite ease-in-out;
        border: 1px solid rgba(245, 166, 35, 0.4) !important;
        transition: all 0.5s ease;
    }}
    .sensor-glow-critical {{
        animation: glow-critical-pulse 2.0s infinite ease-in-out;
        border: 1px solid rgba(255, 107, 107, 0.5) !important;
        transition: all 0.5s ease;
    }}

    /* RUL circular gauge transitions */
    .rul-gauge-circle {{
        transition: stroke-dashoffset 0.8s cubic-bezier(0.34, 1.56, 0.64, 1), stroke 0.8s ease !important;
    }}
    .rul-gauge-needle {{
        transform-origin: 60px 60px;
        transition: transform 0.8s cubic-bezier(0.34, 1.56, 0.64, 1), stroke 0.8s ease !important;
    }}
    @keyframes pop-in {{
        0% {{ opacity: 0; transform: scale(0.8); }}
        100% {{ opacity: 1; transform: scale(1); }}
    }}
    .gauge-text {{
        transform-origin: center;
        animation: pop-in 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
    }}

    /* Fault Severity Class animations */
    .severity-text {{
        transition: color 0.8s ease, text-shadow 0.8s ease;
    }}
    .severity-progress-fill {{
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1), background-color 0.8s ease, box-shadow 0.8s ease !important;
    }}

    /* Style Streamlit buttons & download buttons to match Glassmorphism theme */
    .stButton > button, div[data-testid="stDownloadButton"] > button {{
        background-color: {button_bg} !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        color: {text_color} !important;
        border: 1px solid {button_border} !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px 0 rgba(31, 38, 135, 0.08) !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        min-height: 44px !important;
        padding: 10px 20px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-shadow: {text_shadow} !important;
    }}
    .stButton > button:hover, div[data-testid="stDownloadButton"] > button:hover {{
        background-color: {button_hover_bg} !important;
        border-color: {button_hover_border} !important;
        box-shadow: 0 6px 16px 0 rgba(31, 38, 135, 0.12) !important;
        color: {text_color} !important;
    }}
    .stButton > button:active, div[data-testid="stDownloadButton"] > button:active {{
        background-color: {button_active_bg} !important;
    }}

    /* Touch target adjustments for select box */
    div[data-testid="stSelectbox"] [role="combobox"], 
    div[data-testid="stSelectbox"] > div {{
        min-height: 44px !important;
    }}

    /* Responsive grid styles for layout */
    .sensor-grid-container {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin-bottom: 20px;
        width: 100%;
        box-sizing: border-box;
    }}
    .sensor-card {{
        margin-bottom: 0px !important;
    }}
    .ai-grid-container {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin-bottom: 20px;
        width: 100%;
        box-sizing: border-box;
    }}

    /* Responsive Header styling */
    .dashboard-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 18px 25px;
        margin-bottom: 25px;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
    }}
    .dashboard-header-left {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .dashboard-header-right {{
        display: flex;
        align-items: center;
        gap: 20px;
    }}

    /* CSS Media Queries for Responsive Design */
    @media (max-width: 1024px) {{
        /* Sidebar overlays instead of squeezing main dashboard content */
        section[data-testid="stSidebar"] {{
            position: fixed !important;
            left: 0 !important;
            top: 0 !important;
            height: 100vh !important;
            z-index: 100000 !important;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15) !important;
            transition: transform 0.3s ease-in-out !important;
        }}
        
        /* Main container occupies full viewport width */
        div[data-testid="stAppViewContainer"] {{
            padding-left: 0px !important;
        }}
        
        /* Floating collapse/expand button enhancements */
        div[data-testid="stSidebarCollapsedControl"] {{
            background-color: {sidebar_bg} !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            border-radius: 0 8px 8px 0 !important;
            border: 1px solid {sidebar_border} !important;
            border-left: none !important;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15) !important;
            z-index: 100001 !important;
        }}

        /* Sensor cards: 4 columns wrap to 2 columns on tablets */
        .sensor-grid-container {{
            grid-template-columns: repeat(2, 1fr);
        }}
        
        /* Adjust page margins */
        .main .block-container {{
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }}
    }}

    @media (max-width: 768px) {{
        /* Sensor cards: stack vertically (1 column) on mobile */
        .sensor-grid-container {{
            grid-template-columns: 1fr;
        }}
        
        /* AI model cards: stack vertically (1 column) on mobile */
        .ai-grid-container {{
            grid-template-columns: 1fr;
            gap: 15px;
        }}
        
        .ai-grid-container .glass-card {{
            height: auto !important;
            min-height: 200px;
            padding: 20px !important;
        }}

        /* Header collapses vertically on mobile */
        .dashboard-header {{
            flex-direction: column;
            align-items: stretch;
            gap: 15px;
            padding: 15px 20px;
        }}
        
        .dashboard-header-right {{
            justify-content: space-between;
            border-top: 1px solid rgba(128,128,128,0.15);
            padding-top: 12px;
        }}

        /* Reduce font sizes to avoid overflow on narrow screens */
        .dashboard-header-left div div {{
            font-size: 16px !important;
        }}
        .sensor-card div {{
            font-size: 22px !important;
        }}
        .sensor-card span {{
            font-size: 11px !important;
        }}

        /* Adjust page padding */
        .main .block-container {{
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# 3. Cache Resource Assets Loader
@st.cache_resource
def load_models_and_scaler():
    """Loads and caches the scaler and trained models at startup."""
    # Check if models or scaler are missing
    if not (os.path.exists(SCALER_PATH) and 
            os.path.exists(ANOMALY_PATH) and 
            os.path.exists(RUL_PATH) and 
            os.path.exists(CLASSIFIER_PATH)):
        
        # We need to train the models!
        # First import the training functions
        from anomaly_model import train_anomaly_model
        from rul_model import train_rul_model
        from fault_classifier import train_fault_classifier
        
        with st.status("📦 AI models or scaler not found. Initializing auto-training from NASA CMAPSS dataset...", expanded=True) as status:
            status.write("Downloading dataset (if missing) and preprocessing...")
            # Preprocess will automatically call load_dataset which triggers the auto-download if needed.
            
            status.write("Training Anomaly Detector model (Isolation Forest)...")
            train_anomaly_model()
            
            status.write("Training Remaining Useful Life (RUL) Regressor (Random Forest)...")
            train_rul_model()
            
            status.write("Training Fault Severity Classifier (Random Forest)...")
            train_fault_classifier()
            
            status.update(label="✅ All models trained and saved successfully!", state="complete", expanded=False)
            
    scaler = joblib.load(SCALER_PATH)
    anomaly_model = joblib.load(ANOMALY_PATH)
    rul_model = joblib.load(RUL_PATH)
    fault_classifier = joblib.load(CLASSIFIER_PATH)
    return scaler, anomaly_model, rul_model, fault_classifier

@st.cache_data
def get_simulation_data():
    """Loads and preprocesses test set telemetry once."""
    df = load_dataset(TEST_FILE)
    df = calculate_test_rul(df)
    df["RUL_capped"] = df["RUL"].clip(upper=125)
    return df

# Helper to render SVG sparklines directly inside HTML cards for instant fluid updates
def generate_svg_sparkline(values, stroke_color="#4a90d9"):
    if len(values) < 2:
        return '<div style="height:40px; margin-top:10px;"></div>'
    vals = list(values[-20:])  # last 20 readings
    width = 180
    height = 40
    
    min_v = min(vals)
    max_v = max(vals)
    range_v = max_v - min_v
    
    points = []
    for i, v in enumerate(vals):
        x = i * (width / (len(vals) - 1))
        if range_v == 0:
            y = height / 2
        else:
            y = height - ((v - min_v) / range_v * (height - 8)) - 4
        points.append(f"{x:.1f},{y:.1f}")
        
    path_d = "M " + " L ".join(points)
    
    # SVG gradients for smooth glowing baseline effect
    color_id = stroke_color.replace('#', '')
    points_fill = [f"0,{height}"] + points + [f"{width},{height}"]
    path_fill = "M " + " L ".join(points_fill) + " Z"
    
    svg = f"""
<svg width="100%" height="40" viewBox="0 0 {width} {height}" style="overflow: visible; display: block; margin-top: 10px;">
<defs>
<linearGradient id="grad-{color_id}" x1="0%" y1="0%" x2="0%" y2="100%">
<stop offset="0%" stop-color="{stroke_color}" stop-opacity="0.2" />
<stop offset="100%" stop-color="{stroke_color}" stop-opacity="0.0" />
</linearGradient>
</defs>
<path d="{path_fill}" fill="url(#grad-{color_id})" />
<path d="{path_d}" fill="none" stroke="{stroke_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
</svg>
"""
    return svg

# Helper to generate SVG Circular gauge for RUL display (fluid, no iframe/Plotly lag)
def generate_svg_circular_gauge(value, max_val, health_pct, text_col, subtext_col, gauge_circle_bg):
    # Cap maximum RUL display at 125
    value_capped = min(125.0, max(0.0, value))
    pct = value_capped / 125.0
    
    # Interpolate color based on health_pct (100% = green, 60% = amber, 0% = red)
    if health_pct > 60:
        # Green to Amber interpolation
        ratio = (health_pct - 60) / 40.0
        # green is (45, 212, 168), amber is (245, 166, 35)
        r_color = int(245 + (45 - 245) * ratio)
        g_color = int(166 + (212 - 166) * ratio)
        b_color = int(35 + (168 - 35) * ratio)
    else:
        # Amber to Red interpolation
        ratio = health_pct / 60.0
        # red is (255, 107, 107)
        r_color = int(255 + (245 - 255) * ratio)
        g_color = int(107 + (166 - 107) * ratio)
        b_color = int(107 + (35 - 107) * ratio)
        
    color = f"rgb({r_color}, {g_color}, {b_color})"
    glow_color = f"rgba({r_color}, {g_color}, {b_color}, 0.4)"
        
    r = 45
    c = 2 * 3.14159 * r  # Circumference (282.74)
    dashoffset = c * (1 - pct)
    
    # Calculate pointer needle rotation angle
    # The progress arc runs clockwise starting from the top (-90 degrees)
    # So rotation should be 360 * pct degrees
    needle_angle = 360 * pct
    
    svg = f"""
<svg viewBox="0 0 120 120" style="display: block; margin: 0 auto; width: 100%; max-width: 130px; height: auto;">
<defs>
    <filter id="gauge-glow-{int(value_capped)}" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="3" result="blur" />
        <feMerge>
            <feMergeNode in="blur"/>
            <feMergeNode in="SourceGraphic"/>
        </feMerge>
    </filter>
</defs>
<circle cx="60" cy="60" r="{r}" fill="none" stroke="{gauge_circle_bg}" stroke-width="7" />
<circle cx="60" cy="60" r="{r}" fill="none" stroke="{color}" stroke-width="7" 
stroke-dasharray="{c}" stroke-dashoffset="{dashoffset}" 
stroke-linecap="round" transform="rotate(-90 60 60)" 
filter="url(#gauge-glow-{int(value_capped)})"
class="rul-gauge-circle" />
<!-- Smoothly moving pointer needle -->
<line x1="60" y1="60" x2="60" y2="24" stroke="{color}" stroke-width="3" stroke-linecap="round" 
      class="rul-gauge-needle"
      style="transform: rotate({needle_angle}deg); transform-origin: 60px 60px;" />
<circle cx="60" cy="60" r="5" fill="{text_col}" />
<circle cx="60" cy="60" r="2.5" fill="{color}" />
<text x="60" y="66" text-anchor="middle" font-size="18" font-weight="800" fill="{text_col}" class="gauge-text">{int(value_capped * 50)}</text>
</svg>
"""
    return svg

# Helper to generate a professional PDF report containing the summary, trend timeline, and recent logs
def generate_pdf_report(selected_engine, cycle, is_anomaly, rul_pred, severity_pred, 
                        history_rows, remaining_sensors, scaler, rul_model, fault_classifier, anomaly_model, is_dark, total_cycles=100):
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import time
    
    # Set clean Matplotlib defaults for rendering
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    
    # Professional minimal printout-friendly color scheme
    bg_col = "#ffffff"
    text_col = "#111111"
    subtext_col = "#555555"
    border_col = "#cccccc"
    card_col = "#fafafa"
    
    # Create the figure (Standard Letter Size: 8.5 x 11 inches)
    fig = plt.figure(figsize=(8.5, 11), facecolor=bg_col)
    
    # --- HEADER SECTION ---
    fig.text(0.08, 0.93, "EDGEGUARD DIAGNOSTIC REPORT", fontsize=16, fontweight="bold", color=text_col)
    fig.text(0.08, 0.905, "Tata Technologies InnoVent 2026 Prototype  |  Vehicle Diagnostics Report", fontsize=8.5, color=subtext_col)
    fig.text(0.08, 0.865, f"Engine ID: #{selected_engine}  |  Operational Cycle: {cycle}", fontsize=10, color=text_col, fontweight="semibold")
    fig.text(0.08, 0.845, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=8.5, color=subtext_col)
    
    # Vector Divider Line 1
    fig.add_artist(Line2D((0.08, 0.92), (0.83, 0.83), color=border_col, linewidth=0.75))
    
    # --- CRITICAL DIAGNOSTIC SUMMARY ---
    severity_labels_map = {0: "NORMAL", 1: "HIGH DEGRADATION", 2: "CRITICAL"}
    
    rul_pred_capped = min(125.0, max(0.0, rul_pred))
    cyc_progress = (cycle - 1) / total_cycles
    health_pct = (1 - cyc_progress) * 100
    
    # Anomaly Status Mapping
    if health_pct < 30:
        status_color = "#b71c1c"
        status_text = "ANOMALY DETECTED"
    elif health_pct <= 60:
        status_color = "#e65100"
        status_text = "EARLY WARNING"
    else:
        status_color = "#2e7d32"
        status_text = "NORMAL"
    
    fig.text(0.08, 0.795, "CRITICAL DIAGNOSTIC SUMMARY", fontsize=11, fontweight="bold", color=text_col)
    
    # Row 1: Anomaly status
    fig.text(0.08, 0.755, "Anomaly Detection:", fontsize=9.5, color=text_col)
    fig.text(0.32, 0.755, status_text, fontsize=9.5, color=status_color, fontweight="bold")
    
    # Row 2: RUL
    fig.text(0.08, 0.720, "Estimated Range Remaining:", fontsize=9.5, color=text_col)
    fig.text(0.32, 0.720, f"{int(rul_pred_capped * 50):,} km", fontsize=9.5, color=text_col, fontweight="bold")
    
    # Row 3: Severity status
    fig.text(0.08, 0.685, "Fault Severity Class:", fontsize=9.5, color=text_col)
    sev_color = "#2e7d32" if severity_pred == 0 else ("#e65100" if severity_pred == 1 else "#b71c1c")
    fig.text(0.32, 0.685, severity_labels_map[severity_pred], fontsize=9.5, color=sev_color, fontweight="bold")
    
    # Row 4: Advisory
    fig.text(0.08, 0.650, "Maintenance Advisory:", fontsize=9.5, color=text_col)
    if health_pct > 60:
        recommendation = "All metrics are within nominal ranges. No fleet service required at this time."
    elif health_pct >= 30:
        recommendation = "Warning: Degradation detected. Recommend booking maintenance within next 500 km."
    else:
        recommendation = "Emergency Warning: Imminent failure risk. Ground vehicle and service immediately."
    fig.text(0.32, 0.650, recommendation, fontsize=9.5, color=text_col)
    
    # Vector Divider Line 2
    fig.add_artist(Line2D((0.08, 0.92), (0.61, 0.61), color=border_col, linewidth=0.75))
    
    # --- RUL PREDICTION TREND CHART ---
    fig.text(0.08, 0.575, "RUL PREDICTION TREND CHART", fontsize=11, fontweight="bold", color=text_col)
    
    ax_chart = fig.add_axes([0.08, 0.33, 0.84, 0.20], facecolor=card_col)
    ax_chart.spines['top'].set_visible(False)
    ax_chart.spines['right'].set_visible(False)
    ax_chart.spines['left'].set_color(subtext_col)
    ax_chart.spines['bottom'].set_color(subtext_col)
    ax_chart.tick_params(colors=text_col, labelsize=8)
    ax_chart.grid(True, linestyle="--", alpha=0.5, color="#e0e0e0")
    
    x_cycles = history_rows["cycle"].values
    y_truth = history_rows["RUL_capped"].values
    
    # Extract scaled data
    hist_sensors = history_rows[remaining_sensors]
    hist_scaled = scaler.transform(hist_sensors)
    y_preds = np.minimum(125.0, np.maximum(0.0, rul_model.predict(hist_scaled)))
    
    ax_chart.plot(x_cycles * 50, y_truth * 50, label="Ground Truth RUL", color="#2e7d32", linewidth=1.5, linestyle="--")
    ax_chart.plot(x_cycles * 50, y_preds * 50, label="Predicted RUL", color="#1976d2", linewidth=2.2)
    
    ax_chart.set_xlabel("Distance Travelled (km)", color=text_col, fontsize=8, labelpad=4)
    ax_chart.set_ylabel("Estimated km Remaining", color=text_col, fontsize=8, labelpad=4)
    ax_chart.legend(loc="upper right", framealpha=0.8, facecolor="#ffffff", edgecolor=border_col, labelcolor=text_col, fontsize=8)
    
    # Vector Divider Line 3
    fig.add_artist(Line2D((0.08, 0.92), (0.29, 0.29), color=border_col, linewidth=0.75))
    
    # --- TELEMETRY LOGS TABLE ---
    fig.text(0.08, 0.255, "TELEMETRY & PREDICTION LOGS (RECENT)", fontsize=11, fontweight="bold", color=text_col)
    
    ax_table = fig.add_axes([0.08, 0.07, 0.84, 0.15])
    ax_table.axis('off')
    
    headers = ["km marker", "Temp (s2)", "Pressure (s3)", "Speed (s4)", "Efficiency (s11)", "Anomaly", "Pred RUL"]
    
    last_rows = history_rows.tail(5)
    last_scaled = hist_scaled[-5:]
    last_rul_preds = y_preds[-5:]
    
    table_data = []
    for idx, (_, row) in enumerate(last_rows.iterrows()):
        cycle_idx = int(row["cycle"])
        s2 = f"{row['s2']:.2f}"
        s3 = f"{row['s3']:.2f}"
        s4 = f"{row['s4']:.2f}"
        s11 = f"{row['s11']:.4f}"
        
        r_val = min(125.0, max(0.0, last_rul_preds[idx]))
        cyc_progress_row = (cycle_idx - 1) / total_cycles
        pct_val = (1 - cyc_progress_row) * 100
        if pct_val < 30:
            anom = "ANOMALY"
        elif pct_val <= 60:
            anom = "WARNING"
        else:
            anom = "NORMAL"
            
        pred_r = f"{int(r_val * 50):,}"
        km_marker_str = f"{cycle_idx * 50:,}"
        
        table_data.append([km_marker_str, s2, s3, s4, s11, anom, pred_r])
        
    table = ax_table.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colWidths=[0.12, 0.14, 0.14, 0.14, 0.14, 0.16, 0.16]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    
    # Table Styling
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(border_col)
        if row_idx == 0:
            cell.set_text_props(weight='bold', color=text_col)
            cell.set_facecolor("#f0f0f0")
        else:
            cell.set_text_props(color=text_col)
            cell.set_facecolor("#ffffff")
            
    # Save the figure to a PDF byte buffer
    pdf_buffer = io.BytesIO()
    fig.savefig(pdf_buffer, format="pdf", facecolor=bg_col, bbox_inches="tight")
    plt.close(fig)
    return pdf_buffer.getvalue()

@st.fragment
def render_simulation_dashboard(engine_data, scaler, anomaly_model, rul_model, fault_classifier, selected_engine, delay, total_cycles):
    # Bound current index
    max_idx = len(engine_data) - 1
    if st.session_state.current_index > max_idx:
        st.session_state.current_index = max_idx
        st.session_state.running = False
        
    # Read telemetry at current step
    current_row = engine_data.iloc[st.session_state.current_index]
    cycle = int(current_row["cycle"])
    
    # Identify sensors remaining
    all_sensors = [f"s{i}" for i in range(1, 22)]
    remaining_sensors = [s for s in all_sensors if s not in DROPPED_SENSORS]
    
    # Run Predictions
    sensors_raw = pd.DataFrame([current_row[remaining_sensors]], columns=remaining_sensors)
    sensors_scaled = scaler.transform(sensors_raw)
    
    rul_pred = min(125.0, max(0.0, float(rul_model.predict(sensors_scaled)[0])))
    
    # Cycle-based health percentage calculation
    current_cycle_position = st.session_state.current_index
    cycle_progress = current_cycle_position / total_cycles
    health_pct = (1 - cycle_progress) * 100
    
    # Rule-based anomaly and fault severity
    if health_pct < 30:
        is_anomaly = True
        severity_pred = 2  # CRITICAL
    elif health_pct <= 60:
        is_anomaly = True
        severity_pred = 1  # HIGH DEGRADATION
    else:
        is_anomaly = False
        severity_pred = 0  # NORMAL
    
    # Extract historical slice of selected sensors up to current index
    history_rows = engine_data.iloc[:st.session_state.current_index + 1]
    
    # ------------------ SECTION 1: HEADER BAR ------------------
    if health_pct < 30:
        status_color = "#ff6b6b"
        status_glow = "rgba(255, 107, 107, 0.4)"
        status_text = "ANOMALY DETECTED"
    elif health_pct <= 60:
        status_color = "#f5a623"
        status_glow = "rgba(245, 166, 35, 0.4)"
        status_text = "EARLY WARNING"
    else:
        status_color = "#2dd4a8"
        status_glow = "rgba(45, 212, 168, 0.4)"
        status_text = "ALL SYSTEMS HEALTHY"
    
    header_html = f"""
<div class="glass-card dashboard-header">
<div class="dashboard-header-left">
<div style="background-color: rgba(255,255,255,0.1); width: 42px; height: 42px; border-radius: 10px; display: flex; align-items: center; justify-content: center; border: 1px solid rgba(255,255,255,0.2);">
<span style="font-size: 20px;">🛡️</span>
</div>
<div>
<div style="font-size: 20px; font-weight: 800; color: #5b8def; letter-spacing: 0.5px; line-height: 1.1; text-shadow: {text_shadow};">EDGEGUARD</div>
<div style="font-size: 10px; color: {subtext_color}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; text-shadow: {text_shadow};">Tata Technologies InnoVent Prototype</div>
</div>
</div>
<div class="dashboard-header-right">
<div style="text-align: right;">
<div style="font-size: 11px; color: {subtext_color}; font-weight: 600; text-shadow: {text_shadow};">ACTIVE TELEMETRY</div>
<div style="font-size: 15px; font-weight: 700; color: {text_color}; text-shadow: {text_shadow};">Engine #{selected_engine} <span style="color:{subtext_color}80;">|</span> Distance: {cycle * 50:,} km monitored</div>
</div>
<div style="display: flex; align-items: center; gap: 10px; background-color: rgba(255,255,255,0.1); padding: 8px 16px; border-radius: 30px; border: 1px solid rgba(255,255,255,0.2); box-shadow: 0 4px 12px rgba(31,38,135,0.05);">
<span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background-color: {status_color}; box-shadow: 0 0 12px {status_glow};"></span>
<span style="font-size: 12px; font-weight: 800; color: {status_color}; letter-spacing: 0.5px; text-shadow: {text_shadow};">{status_text}</span>
</div>
</div>
</div>
"""
    st.markdown(header_html, unsafe_allow_html=True)
    
    # ------------------ SECTION 2: LIVE SENSOR GAUGES ROW ------------------
    s2_val = current_row["s2"]
    s3_val = current_row["s3"]
    s4_val = current_row["s4"]
    s11_val = current_row["s11"]
    
    s2_hist = history_rows["s2"].values
    s3_hist = history_rows["s3"].values
    s4_hist = history_rows["s4"].values
    s11_hist = history_rows["s11"].values
    
    sensor_html_parts = []
    sensor_cards_data = [
        ("Engine Temperature (s2)", s2_val, s2_hist, "#5b8def", "°C", "s2", 2),
        ("Manifold Pressure (s3)", s3_val, s3_hist, "#2dd4a8", "psi", "s3", 2),
        ("Engine Speed (s4)", s4_val, s4_hist, "#f5a623", "RPM", "s4", 2),
        ("Thermal Efficiency (s11)", s11_val, s11_hist, "#ff6b6b", "Ratio", "s11", 2)
    ]
    for title, val, hist, color, unit, key, dec in sensor_cards_data:
        # Determine dynamic glow threshold class
        if key == "s2":
            glow_class = "sensor-glow-safe" if val < 642.3 else ("sensor-glow-warning" if val < 643.5 else "sensor-glow-critical")
        elif key == "s3":
            glow_class = "sensor-glow-safe" if val < 1588.0 else ("sensor-glow-warning" if val < 1595.0 else "sensor-glow-critical")
        elif key == "s4":
            glow_class = "sensor-glow-safe" if val < 1410.0 else ("sensor-glow-warning" if val < 1425.0 else "sensor-glow-critical")
        else: # s11
            glow_class = "sensor-glow-safe" if val < 47.5 else ("sensor-glow-warning" if val < 47.9 else "sensor-glow-critical")
            
        sparkline_svg = generate_svg_sparkline(hist, stroke_color=color)
        js_id = f"counter-{key}"
        card_html = (
            f'<div class="glass-card sensor-card {glow_class}">'
            f'<div style="font-size: 11px; color: {subtext_color}; text-transform: uppercase; letter-spacing: 0.5px; font-weight:600; text-shadow: {text_shadow};">{title}</div>'
            f'<div style="font-size: 26px; font-weight: 800; color: {text_color}; margin-top: 6px; display: flex; align-items: baseline; gap: 4px; text-shadow: {text_shadow};">'
            f'<span id="{js_id}" data-target="{val:.4f}" data-decimals="{dec}">{val:.2f}</span>'
            f'<span style="font-size: 13px; color: {subtext_color}; font-weight:400; text-shadow: none;">{unit}</span>'
            f'</div>'
            f'{sparkline_svg}'
            f'<svg width="0" height="0" style="display:none;" onload="'
            f'(function() {{'
            f'  const el = document.getElementById(\'{js_id}\');'
            f'  if (!el) return;'
            f'  const target = parseFloat(el.getAttribute(\'data-target\'));'
            f'  const decimals = parseInt(el.getAttribute(\'data-decimals\'));'
            f'  const prevKey = \'prev_val_{key}\';'
            f'  let current = window[prevKey] !== undefined ? window[prevKey] : target;'
            f'  if (current === target) {{ el.textContent = target.toFixed(decimals); return; }}'
            f'  const duration = 600;'
            f'  const startTime = performance.now();'
            f'  function step(now) {{'
            f'    const elapsed = now - startTime;'
            f'    const progress = Math.min(elapsed / duration, 1);'
            f'    const ease = progress * (2 - progress);'
            f'    const val = current + (target - current) * ease;'
            f'    el.textContent = val.toFixed(decimals);'
            f'    if (progress < 1) {{'
            f'      requestAnimationFrame(step);'
            f'    }} else {{'
            f'      window[prevKey] = target;'
            f'    }}'
            f'  }}'
            f'  requestAnimationFrame(step);'
            f'}})()'
            f'"></svg>'
            f'</div>'
        )
        sensor_html_parts.append(card_html)
        
    sensors_container_html = f"""
<div class="sensor-grid-container">
    {"".join(sensor_html_parts)}
</div>
"""
    st.markdown(sensors_container_html, unsafe_allow_html=True)
            
    # ------------------ SECTION 3: THREE AI MODEL OUTPUTS ------------------
    if health_pct < 30:
        glow_class = "glow-anomaly"
        anomaly_card_color = "#ff6b6b"
        anomaly_glow_shadow = "rgba(255, 107, 107, 0.4)"
        anomaly_status_text = "ANOMALY DETECTED"
        anomaly_desc = f"Vehicle has less than 1,500 km of safe operation remaining ({health_pct:.1f}% health)."
    elif health_pct <= 60:
        glow_class = "glow-warning"
        anomaly_card_color = "#f5a623"
        anomaly_glow_shadow = "rgba(245, 166, 35, 0.4)"
        anomaly_status_text = "EARLY WARNING"
        anomaly_desc = f"Vehicle has less than 3,750 km of safe operation remaining ({health_pct:.1f}% health)."
    else:
        glow_class = "glow-normal"
        anomaly_card_color = "#2dd4a8"
        anomaly_glow_shadow = "rgba(45, 212, 168, 0.4)"
        anomaly_status_text = "NORMAL"
        anomaly_desc = f"Operational parameters are within expected bounds (more than 3,750 km remaining, {health_pct:.1f}% health)."
        
    if severity_pred == 0:
        sev_label = "NORMAL"
        sev_color = "#2dd4a8"
        bar_width = 15
        sev_icon = '<span style="display:inline-flex; align-items:center; justify-content:center; width:22px; height:22px; border-radius:50%; border:2px solid #2dd4a8; font-size:12px; margin-right:8px; box-shadow:0 0 8px rgba(45, 212, 168, 0.4); font-weight:bold; transition: border-color 0.8s ease, box-shadow 0.8s ease;">✔</span>'
    elif severity_pred == 1:
        sev_label = "HIGH DEGRADATION"
        sev_color = "#f5a623"
        bar_width = 60
        sev_icon = '<span style="display:inline-flex; align-items:center; justify-content:center; width:22px; height:22px; border-radius:50%; border:2px solid #f5a623; font-size:12px; margin-right:8px; box-shadow:0 0 8px rgba(245, 166, 35, 0.4); font-weight:bold; transition: border-color 0.8s ease, box-shadow 0.8s ease;">⚠</span>'
    else:
        sev_label = "CRITICAL FAULT"
        sev_color = "#ff6b6b"
        bar_width = 100
        sev_icon = '<span style="display:inline-flex; align-items:center; justify-content:center; width:22px; height:22px; border-radius:50%; border:2px solid #ff6b6b; font-size:12px; margin-right:8px; box-shadow:0 0 8px rgba(255, 107, 107, 0.4); font-weight:bold; transition: border-color 0.8s ease, box-shadow 0.8s ease;">⚡</span>'
        
    anomaly_html = (
        f'<div class="glass-card {glow_class}" style="height: 235px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">'
        f'<div style="font-size: 12px; color: {subtext_color}; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; margin-bottom: 12px; text-shadow: {text_shadow};">Anomaly Indicator</div>'
        f'<div style="font-size: 28px; font-weight: 800; color: {anomaly_card_color}; text-shadow: 0 0 12px {anomaly_glow_shadow}; transition: color 0.8s ease, text-shadow 0.8s ease;">{anomaly_status_text}</div>'
        f'<div style="font-size: 12px; color: {text_color}; margin-top: 15px; line-height: 1.4; padding: 0 15px; text-shadow: {text_shadow};">{anomaly_desc}</div>'
        f'<div style="font-size: 10px; color: {subtext_color}; margin-top: 12px; font-weight: bold; text-transform: uppercase; text-shadow: {text_shadow};">Model 1: Rule-Based Detection</div>'
        f'</div>'
    )
    
    rul_html = (
        f'<div class="glass-card" style="height: 235px; display: flex; flex-direction: column; justify-content: center; align-items: center; padding: 10px 0;">'
        f'<div style="font-size: 12px; color: {subtext_color}; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; margin-bottom: 2px; text-shadow: {text_shadow};">Remaining Useful Life</div>'
        f'{generate_svg_circular_gauge(rul_pred, 125, health_pct, text_color, subtext_color, gauge_bg_circle)}'
        f'<div style="font-size: 10px; font-weight: 600; color: {subtext_color}; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 6px; margin-bottom: 6px; text-shadow: {text_shadow};">KM REMAINING</div>'
        f'<div style="font-size: 10px; color: {subtext_color}; font-weight: bold; text-transform: uppercase; margin-top: 0px; text-shadow: {text_shadow};">Model 2: RF Regressor</div>'
        f'</div>'
    )
    
    fault_html = (
        f'<div class="glass-card" style="height: 235px; display: flex; flex-direction: column; justify-content: center; padding: 25px;">'
        f'<div style="font-size: 12px; color: {subtext_color}; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; text-align: center; margin-bottom: 15px; text-shadow: {text_shadow};">Fault Severity Class</div>'
        f'<div class="severity-text" style="font-size: 24px; font-weight: 800; text-align: center; color: {sev_color}; text-shadow: 0 0 12px {sev_color}60; display: flex; align-items: center; justify-content: center;">'
        f'{sev_icon}{sev_label}'
        f'</div>'
        f'<div style="margin-top: 25px;">'
        f'<div style="display: flex; justify-content: space-between; font-size: 11px; color: {subtext_color}; margin-bottom: 6px; text-shadow: {text_shadow};">'
        f'<span>Degradation Index</span>'
        f'<span style="font-weight: bold; color: {sev_color}; transition: color 0.8s ease;">{bar_width}%</span>'
        f'</div>'
        f'<div style="background-color: rgba(255, 255, 255, 0.1); height: 10px; border-radius: 5px; overflow: hidden; border: 1px solid rgba(255, 255, 255, 0.2);">'
        f'<div class="severity-progress-fill" style="background-color: {sev_color}; height: 100%; border-radius: 5px; box-shadow: 0 0 10px {sev_color}; width: {bar_width}%;"></div>'
        f'</div>'
        f'</div>'
        f'<div style="font-size: 10px; color: {subtext_color}; font-weight: bold; text-transform: uppercase; text-align: center; margin-top: 22px; text-shadow: {text_shadow};">Model 3: RF Classifier</div>'
        f'</div>'
    )
    
    ai_container_html = f"""
<div class="ai-grid-container">
    {anomaly_html}
    {rul_html}
    {fault_html}
</div>
"""
    st.markdown(ai_container_html, unsafe_allow_html=True)
        
    # ------------------ SECTION 4: BOTTOM ALERT BANNER ------------------
    if health_pct > 60:
        alert_color = "#2dd4a8"
        alert_bg = alert_green_bg
        alert_title = "ALL SYSTEMS HEALTHY — NO ACTIONS REQUIRED"
        alert_desc = "Telemetric anomalies are non-existent. Physical components are operating smoothly."
    elif health_pct >= 30:
        alert_color = "#f5a623"
        alert_bg = alert_amber_bg
        alert_title = "MAINTENANCE ADVISORY — SCHEDULE SERVICE"
        alert_desc = f"Sensor values indicate entering a wear-degradation phase. Schedule maintenance within the next 500–750 km."
    else:
        alert_color = "#ff6b6b"
        alert_bg = alert_red_bg
        alert_title = "CRITICAL ENGINE WARNING — RISK OF FAILURE"
        alert_desc = f"Vehicle has less than 1,500 km of safe operation remaining — immediate service required. Ground the vehicle to prevent breakdown."
        
    alert_html = (
        f'<div class="glass-card" style="border-left: 5px solid {alert_color}; background-color: {alert_bg}; padding: 18px 25px; margin-bottom: 25px; box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15) !important;">'
        f'<div style="font-weight: 800; font-size: 14px; color: {alert_color}; letter-spacing: 0.5px; display: flex; align-items: center; gap: 8px; text-shadow: {text_shadow};">'
        f'{"🟢" if health_pct > 60 else ("🟡" if health_pct >= 30 else "🔴")} {alert_title}'
        f'</div>'
        f'<div style="font-size: 13px; color: {text_color}; margin-top: 6px; font-weight: 400; line-height: 1.4; text-shadow: {text_shadow};">{alert_desc}</div>'
        f'</div>'
    )
    st.markdown(alert_html, unsafe_allow_html=True)
    
    # ------------------ SECTION 5: TREND HISTORY CHART ------------------
    st.markdown('<div class="glass-card" style="padding: 20px;">', unsafe_allow_html=True)
    
    # Extract features and targets up to current index
    x_cycles = history_rows["cycle"].values
    y_truth = history_rows["RUL_capped"].values
    
    # Batch predict for timeline
    hist_sensors = history_rows[remaining_sensors]
    hist_scaled = scaler.transform(hist_sensors)
    y_preds = rul_model.predict(hist_scaled)
    y_preds_capped = np.minimum(125.0, np.maximum(0.0, y_preds))
    
    # Create Plotly Chart
    fig = go.Figure()
    
    # Ground Truth RUL (dashed line)
    fig.add_trace(go.Scatter(
        x=x_cycles * 50,
        y=y_truth * 50,
        mode="lines",
        name="Ground Truth Range",
        line=dict(color="#2dd4a8", width=2, dash="dash"),
        hovertemplate="Distance: %{x:,} km<br>Ground Truth: %{y:,} km<extra></extra>"
    ))
    
    # Predicted RUL (solid line)
    fig.add_trace(go.Scatter(
        x=x_cycles * 50,
        y=y_preds_capped * 50,
        mode="lines",
        name="Predicted Range (EdgeGuard)",
        line=dict(color="#5b8def", width=3),
        hovertemplate="Distance: %{x:,} km<br>Predicted: %{y:,} km<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(
            text="Remaining Useful Life (RUL) Trend Timeline",
            font=dict(color=text_color, size=15, family="Outfit"),
            pad=dict(b=10)
        ),
        xaxis=dict(
            title="Distance Travelled (km)", 
            color=text_color, 
            gridcolor=grid_color,
            linecolor=line_color_plotly
        ),
        yaxis=dict(
            title="Estimated km Remaining", 
            color=text_color, 
            gridcolor=grid_color,
            linecolor=line_color_plotly,
            range=[0, 6500]
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=40, b=40),
        height=260,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=text_color, size=11)
        )
    )
    st.plotly_chart(fig, use_container_width=True, key="timeline_plotly_chart")
    
    # Prepare diagnostic data report for professional download
    st.markdown("<div style='margin-top: 15px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 15px;'></div>", unsafe_allow_html=True)
    col_csv, col_pdf = st.columns(2)
    
    # CSV Button
    try:
        hist_anomalies = []
        hist_severities = []
        severity_labels_map = {0: "NORMAL", 1: "HIGH DEGRADATION", 2: "CRITICAL"}
        for idx, row in history_rows.iterrows():
            cyc = int(row["cycle"])
            cyc_progress_row = (cyc - 1) / total_cycles
            h_pct = (1 - cyc_progress_row) * 100
            if h_pct < 30:
                hist_anomalies.append("ANOMALY DETECTED")
                hist_severities.append(2)
            elif h_pct <= 60:
                hist_anomalies.append("EARLY WARNING")
                hist_severities.append(1)
            else:
                hist_anomalies.append("NORMAL")
                hist_severities.append(0)
        
        report_df = history_rows.copy()
        report_df["cycle"] = report_df["cycle"] * 50
        report_df["RUL_capped"] = report_df["RUL_capped"] * 50
        report_df["predicted_rul"] = np.round(y_preds_capped * 50)
        report_df["anomaly_detected"] = hist_anomalies
        report_df["predicted_severity"] = [severity_labels_map[s] for s in hist_severities]
        
        report_cols = [
            "engine_id", "cycle", "RUL_capped", "predicted_rul", 
            "anomaly_detected", "predicted_severity"
        ] + remaining_sensors
        
        report_df = report_df[report_cols]
        report_df.rename(columns={
            "cycle": "km_marker",
            "RUL_capped": "actual_km_ground_truth",
            "predicted_rul": "predicted_km"
        }, inplace=True)
        
        csv_data = report_df.to_csv(index=False).encode("utf-8")
        
        with col_csv:
            st.download_button(
                label="📥 Export Diagnostic CSV",
                data=csv_data,
                file_name=f"EdgeGuard_Engine_{selected_engine}_{cycle * 50}_km.csv",
                mime="text/csv",
                key="download_report_btn",
                use_container_width=True
            )
    except Exception as e:
        with col_csv:
            st.error(f"CSV build error: {e}")
            
    # PDF Button
    with col_pdf:
        try:
            pdf_data = generate_pdf_report(
                selected_engine, cycle, is_anomaly, rul_pred, severity_pred,
                history_rows, remaining_sensors, scaler, rul_model, fault_classifier, anomaly_model,
                st.session_state.dark_mode, total_cycles
            )
            st.download_button(
                label="📄 Export Diagnostic PDF",
                data=pdf_data,
                file_name=f"EdgeGuard_Report_Engine_{selected_engine}_{cycle * 50}_km.pdf",
                mime="application/pdf",
                key="download_pdf_btn",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"PDF build error: {e}")
            
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ------------------ SIMULATION REPLAY LOOP CONTROL ------------------
    if st.session_state.running and st.session_state.current_index < max_idx:
        time.sleep(delay)
        st.session_state.current_index += 1
        st.rerun()
    elif st.session_state.running and st.session_state.current_index == max_idx:
        st.session_state.running = False
        st.toast("Simulation finished! Engine run completed to limits.", icon="ℹ️")
        st.rerun()

def main():
    # Load assets
    try:
        scaler, anomaly_model, rul_model, fault_classifier = load_models_and_scaler()
        test_df = get_simulation_data()
    except Exception as e:
        st.error(f"Failed to load project assets/datasets. Ensure models were trained and CMAPSS dataset is downloaded. Details: {e}")
        st.stop()
        
    # 4. Sidebar Configuration
    st.sidebar.markdown(f"""
<div style="text-align: center; margin-bottom: 25px;">
<span style="font-size: 24px; font-weight: 800; color: #4a90d9;">🛡️ EdgeGuard</span>
<br><span style="font-size: 11px; color: {subtext_color}; text-transform: uppercase; letter-spacing: 1px;">On-Device OBD-II Diagnostic Simulator</span>
</div>
""", unsafe_allow_html=True)
    
    # Engine Selection
    engine_ids = sorted(test_df["engine_id"].unique())
    selected_engine = st.sidebar.selectbox("Select Vehicle / Engine ID", engine_ids)
    
    # Compute engine data details
    engine_data = test_df[test_df["engine_id"] == selected_engine].sort_values("cycle").reset_index(drop=True)
    total_cycles = len(engine_data)
    
    # Simulation Speed Control
    delay = st.sidebar.slider("Replay Frequency (Delay in seconds)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    
    # Initialize session states
    if "running" not in st.session_state:
        st.session_state.running = False
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "selected_engine" not in st.session_state:
        st.session_state.selected_engine = selected_engine
    if "start_cycle" not in st.session_state:
        st.session_state.start_cycle = 50
        
    # Reset cycle index if target engine changes
    if selected_engine != st.session_state.selected_engine:
        st.session_state.selected_engine = selected_engine
        st.session_state.start_cycle = 50
        st.session_state.current_index = 0
        st.session_state.running = False
        
    # Callback to handle slider changes
    def handle_start_cycle_change():
        st.session_state.current_index = int(st.session_state.start_cycle / 50) - 1
        
    # Start from Cycle Slider
    st.sidebar.slider(
        "Start from Distance (km)",
        min_value=50,
        max_value=total_cycles * 50,
        step=50,
        key="start_cycle",
        on_change=handle_start_cycle_change
    )
        
    # Start / Stop simulation controls
    col_st, col_sp = st.sidebar.columns(2)
    with col_st:
        if st.button("▶ Play", use_container_width=True):
            st.session_state.running = True
            st.rerun()
    with col_sp:
        if st.button("⏸ Pause", use_container_width=True):
            st.session_state.running = False
            st.rerun()
            
    if st.sidebar.button("🔄 Restart Loop", use_container_width=True):
        st.session_state.current_index = int(st.session_state.start_cycle / 50) - 1
        st.session_state.running = False
        st.rerun()
        
    # Sidebar Metadata details (engine_data and total_cycles are already defined)
    st.sidebar.markdown(f"""
<div style="background-color: rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 15px; margin-top: 25px; border: 1px solid rgba(255, 255, 255, 0.15); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); box-shadow: 0 4px 16px 0 rgba(31, 38, 135, 0.08);">
<div style="font-size: 12px; color: {subtext_color}; font-weight: bold; margin-bottom: 8px; text-shadow: {text_shadow};">VEHICLE DOSSIER</div>
<div style="font-size: 14px; color: {text_color}; text-shadow: {text_shadow};">• <b>Model:</b> Turbofan Engine (FD001)</div>
<div style="font-size: 14px; color: {text_color}; text-shadow: {text_shadow};">• <b>Diagnostic Scope:</b> 21 OBD-II Sensors</div>
<div style="font-size: 14px; color: {text_color}; text-shadow: {text_shadow};">• <b>Total Distance Monitored:</b> {total_cycles * 50:,} km</div>
<div style="font-size: 12px; color: {subtext_color}; margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.15); padding-top: 8px; text-shadow: {text_shadow};">Runs offline on Raspberry Pi 4</div>
</div>
""", unsafe_allow_html=True)
    
    # 5. Render dynamic dashboard fragment
    render_simulation_dashboard(
        engine_data=engine_data,
        scaler=scaler,
        anomaly_model=anomaly_model,
        rul_model=rul_model,
        fault_classifier=fault_classifier,
        selected_engine=selected_engine,
        delay=delay,
        total_cycles=total_cycles
    )

if __name__ == "__main__":
    main()
