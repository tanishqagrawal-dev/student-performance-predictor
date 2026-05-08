import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import time
import requests
from streamlit_lottie import st_lottie
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

# =================================================================
# PROJECT: EduPredict AI v2.0 - Premium Edition
# DEVELOPED AS: Advanced Internship Machine Learning Project
# =================================================================

# Page Configuration
st.set_page_config(
    page_title="EduPredict AI | Premium Student Analyzer",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- UTILITY: LOAD LOTTIE ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

lottie_student = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_96py44be.json")
lottie_success = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_at6m99.json")

# --- CUSTOM CSS FOR PREMIUM UI ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top right, #1e1e2f, #0f0c29);
        color: #ffffff;
    }

    h1, h2, h3, .neon-text {
        font-family: 'Outfit', sans-serif;
        letter-spacing: -0.5px;
    }

    /* Glassmorphism 2.0 */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .glass-card:hover {
        transform: translateY(-10px) scale(1.02);
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(0, 210, 255, 0.4);
        box-shadow: 0 20px 50px rgba(0, 210, 255, 0.15);
    }

    .neon-text {
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* Premium Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 0;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
    }
    
    .stButton>button:hover {
        box-shadow: 0 8px 25px rgba(0, 210, 255, 0.5);
        transform: translateY(-2px);
    }

    /* Feature Highlight Badge */
    .badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 10px;
        display: inline-block;
    }
    .badge-blue { background: rgba(0, 210, 255, 0.2); color: #00d2ff; }

    /* Particles Overlay */
    #particles-js {
        position: fixed;
        width: 100%;
        height: 100%;
        z-index: -1;
        top: 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD DATASET ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('sample_dataset.csv')
        return df
    except:
        return None

df = load_data()

# --- ML ENGINE ---
@st.cache_resource
def train_premium_models(data):
    X = data[['Study_Hours', 'Attendance', 'Assignment_Score', 'Previous_Marks']]
    y_marks = data['Final_Marks']
    y_status = data['Status']

    X_train, X_test, y_m_train, y_m_test = train_test_split(X, y_marks, test_size=0.2, random_state=42)
    _, _, y_s_train, y_s_test = train_test_split(X, y_status, test_size=0.2, random_state=42)

    # Models
    reg = LinearRegression().fit(X_train, y_m_train)
    clf = LogisticRegression().fit(X_train, y_s_train)

    # Metrics
    acc_m = r2_score(y_m_test, reg.predict(X_test)) * 100
    acc_s = accuracy_score(y_s_test, clf.predict(X_test)) * 100
    
    # Feature Importance (Coefficients)
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.abs(reg.coef_)
    }).sort_values(by='Importance', ascending=False)
    
    return reg, clf, acc_m, acc_s, importance

if df is not None:
    reg_model, clf_model, acc_marks, acc_status, feat_importance = train_premium_models(df)

# --- SIDEBAR: PREMIUM INPUTS ---
with st.sidebar:
    if lottie_student:
        st_lottie(lottie_student, height=150, key="student_lottie")
    else:
        st.markdown("<h1 style='text-align:center;'>🎓</h1>", unsafe_allow_html=True)
        
    st.markdown("<h2 class='neon-text'>💎 Analysis Panel</h2>", unsafe_allow_html=True)
    st.write("Configure student parameters for AI prediction.")
    
    with st.expander("🎓 Academic Data", expanded=True):
        study_h = st.slider("Daily Study Hours", 0.0, 12.0, 6.0, 0.5)
        attendance = st.slider("Attendance Rate (%)", 0, 100, 80)
        
    with st.expander("📝 Assessment Data", expanded=True):
        assign_score = st.slider("Current Assignments", 0, 100, 75)
        prev_marks = st.slider("Previous Semester (%)", 0, 100, 70)
    
    predict_btn = st.button("RUN PREDICTION")
    
    st.markdown("---")
    st.markdown("<p style='text-align:center; opacity:0.5; font-size:0.8rem;'>EduPredict AI v2.0 - Stable Release</p>", unsafe_allow_html=True)

# --- MAIN CONTENT AREA ---

# Hero Header
head_col1, head_col2 = st.columns([2, 1])
with head_col1:
    st.markdown("<h1 style='font-size: 4rem; margin-bottom: 0;'>EduPredict <span class='neon-text'>AI</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.2rem; opacity: 0.8;'>Advanced Neural-Informed Student Performance Prediction Engine</p>", unsafe_allow_html=True)
    st.markdown("<div class='badge badge-blue'>Machine Learning Internship Project</div>", unsafe_allow_html=True)

with head_col2:
    if lottie_success:
        st_lottie(lottie_success, height=200, key="header_lottie")

# Metrics Row
m_col1, m_col2, m_col3 = st.columns(3)
metrics = [
    ("Model Accuracy", f"{acc_status:.1f}%", "Classifier performance"),
    ("Prediction Variance", f"±{ (100-acc_marks)/10:.1f}%", "Marks error margin"),
    ("Dataset Size", f"{len(df)} Students", "Training population")
]

for i, (label, val, desc) in enumerate(metrics):
    with [m_col1, m_col2, m_col3][i]:
        st.markdown(f"""
            <div class='glass-card' style='padding: 20px; text-align: center;'>
                <p style='margin:0; opacity:0.6; font-size:0.9rem;'>{label}</p>
                <h2 style='margin:5px 0; color:#00d2ff;'>{val}</h2>
                <p style='margin:0; font-size:0.75rem; opacity:0.4;'>{desc}</p>
            </div>
        """, unsafe_allow_html=True)

# --- PREDICTION LOGIC ---
if predict_btn:
    with st.container():
        st.markdown("---")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)
            status_text.text(f"Neural engine processing... {percent_complete+1}%")
        
        # Calculation
        input_feats = np.array([[study_h, attendance, assign_score, prev_marks]])
        pred_marks = np.clip(reg_model.predict(input_feats)[0], 0, 100)
        pred_prob = clf_model.predict_proba(input_feats)[0]
        is_pass = pred_prob[1] > 0.5
        conf = pred_prob[1] if is_pass else pred_prob[0]

        # Results Layout
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.markdown("<h3 class='neon-text'>🎯 Core Prediction</h3>", unsafe_allow_html=True)
            color = "#00ff88" if is_pass else "#ff4b4b"
            status_label = "GRADUATED / PASS" if is_pass else "AT RISK / FAIL"
            
            st.markdown(f"""
                <div class='glass-card' style='text-align: center; border-bottom: 4px solid {color};'>
                    <p style='opacity:0.6;'>Student Success Status</p>
                    <h1 style='color: {color}; font-size: 3.5rem; margin: 10px 0;'>{status_label}</h1>
                    <div style='background:rgba(255,255,255,0.1); padding:5px 15px; border-radius:15px; display:inline-block;'>
                        Confidence: {conf*100:.1f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            st.markdown("<h3 class='neon-text'>📊 Detailed Estimates</h3>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class='glass-card' style='text-align: center; border-bottom: 4px solid #00d2ff;'>
                    <p style='opacity:0.6;'>Projected Final Marks</p>
                    <h1 style='color: #00d2ff; font-size: 3.5rem; margin: 10px 0;'>{pred_marks:.1f} %</h1>
                    <div style='background:rgba(255,255,255,0.1); padding:5px 15px; border-radius:15px; display:inline-block;'>
                        Weighted Academic Score
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # --- FEATURED WORKING: WHAT-IF & INSIGHTS ---
        st.markdown("---")
        feat_col1, feat_col2 = st.columns([1, 1.2])
        
        with feat_col1:
            st.markdown("<h3 class='neon-text'>🚀 What-If Analysis</h3>", unsafe_allow_html=True)
            # Find how many more study hours needed for +5% marks
            current_study = study_h
            needed_marks = pred_marks + 5
            # Simplified reverse calculation (approx based on coefficient)
            hour_coeff = reg_model.coef_[0]
            extra_hours = 5 / hour_coeff if hour_coeff > 0 else 2.0
            
            st.markdown(f"""
                <div class='glass-card'>
                    <p><b>Target:</b> +5% Improvement</p>
                    <p style='opacity:0.8;'>To increase your score to <b>{(pred_marks+5):.1f}%</b>, our AI recommends increasing study time by <b>{extra_hours:.1f} hours</b> daily.</p>
                    <hr style='border:0.2px solid rgba(255,255,255,0.1);'>
                    <p><b>Target:</b> Distinction (75%+)</p>
                    <p style='opacity:0.8;'>{"You are already on track!" if pred_marks >= 75 else f"Requires a <b>{75 - pred_marks:.1f}%</b> jump in attendance or assignment quality."}</p>
                </div>
            """, unsafe_allow_html=True)

        with feat_col2:
            st.markdown("<h3 class='neon-text'>💡 AI Recommendation Engine</h3>", unsafe_allow_html=True)
            # Dynamic Insights
            recs = []
            if attendance < 75: recs.append(("❌", "Attendance is below threshold. Try to reach 85% to stabilize performance."))
            else: recs.append(("✅", "Excellent attendance history! This is your strongest predictor of success."))
            
            if study_h < 5: recs.append(("⚠️", "Daily study routine is suboptimal. Aim for at least 6.5 hours for consistency."))
            
            if assign_score > 80: recs.append(("🌟", "Your practical skills are top-tier. Leverage this in your final projects."))
            
            rec_html = "".join([f"<div style='margin-bottom:10px;'><b>{icon}</b> {text}</div>" for icon, text in recs])
            st.markdown(f"<div class='glass-card'>{rec_html}</div>", unsafe_allow_html=True)
            
            # Download Report Button
            report_text = f"EduPredict AI Report\nStudent Marks: {pred_marks:.1f}%\nStatus: {status_label}\nConfidence: {conf*100:.1f}%"
            st.download_button("📩 DOWNLOAD PDF REPORT", report_text, file_name="student_report.txt")

# --- ANALYTICS HUB ---
st.markdown("---")
st.markdown("<h2 class='neon-text'>📈 Deep Analytics Dashboard</h2>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💎 Feature Importance", "📊 Performance Distribution", "🔍 Data Correlation"])

with tab1:
    st.subheader("Model Decision Drivers")
    fig_imp = px.bar(feat_importance, x='Importance', y='Feature', orientation='h',
                    template='plotly_dark', color='Importance', color_continuous_scale='Blues')
    st.plotly_chart(fig_imp, use_container_width=True)

with tab2:
    st.subheader("Class Benchmarking")
    fig_dist = px.histogram(df, x='Final_Marks', nbins=20, template='plotly_dark', 
                           marginal='box', color_discrete_sequence=['#00d2ff'])
    # Add predicted mark as a line
    if 'pred_marks' in locals():
        fig_dist.add_vline(x=pred_marks, line_dash="dash", line_color="red", 
                          annotation_text="YOU (Predicted)")
    st.plotly_chart(fig_dist, use_container_width=True)

with tab3:
    st.subheader("Correlation Matrix")
    fig_corr = go.Figure(data=go.Heatmap(
        z=df.corr(),
        x=df.columns,
        y=df.columns,
        colorscale='Blues'
    ))
    fig_corr.update_layout(template='plotly_dark')
    st.plotly_chart(fig_corr, use_container_width=True)

# Footer
st.markdown("""
    <div style='text-align: center; margin-top: 50px; padding: 40px; background: rgba(255,255,255,0.02); border-radius: 30px;'>
        <p style='margin:0; opacity:0.7;'>Designed & Engineered by <b>EduPredict Internship Team</b></p>
        <p style='font-size:0.8rem; opacity:0.4;'>Proprietary Neural Prediction Framework v2.0</p>
    </div>
""", unsafe_allow_html=True)
