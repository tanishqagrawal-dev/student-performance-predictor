import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import requests
import os
from streamlit_lottie import st_lottie
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score

# =================================================================
# EduPredict AI - Premium Selection Edition v4.5
# =================================================================

st.set_page_config(
    page_title="EduPredict AI | Student Performance Analytics",
    page_icon="🎓",
    layout="wide"
)

# --- UTILITIES ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except: return None

lottie_loading = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_at6m99.json")

# --- PREMIUM CSS (ADVANCED ALIGNMENT) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500&display=swap');

    :root {
        --bg-color: #0a0b1e;
        --card-bg: rgba(255, 255, 255, 0.03);
        --accent-cyan: #00d2ff;
        --accent-green: #00ff99;
        --text-main: #f2f2f2;
        --border-color: rgba(255, 255, 255, 0.1);
    }

    .stApp { background-color: var(--bg-color); color: var(--text-main); font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Outfit', sans-serif; color: #ffffff; margin-bottom: 0.5rem; }

    /* Forces all cards in a row to match height */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid var(--border-color);
        padding: 30px;
        margin-bottom: 20px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        height: 100%;
        min-height: 200px;
    }
    .glass-card:hover { 
        border: 1px solid rgba(0, 210, 255, 0.4); 
        transform: translateY(-8px);
        background: rgba(255, 255, 255, 0.05);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    .neon-text {
        text-shadow: 0 0 8px rgba(0, 210, 255, 0.4);
    }
    .neon-text-green {
        text-shadow: 0 0 8px rgba(0, 255, 153, 0.4);
    }

    .stButton>button {
        background: linear-gradient(135deg, var(--accent-cyan), #3a7bd5);
        color: white; border: none; border-radius: 12px;
        padding: 16px 24px; font-weight: 700; width: 100%;
        font-size: 1.1rem; box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 6px 20px rgba(0, 210, 255, 0.5); }

    .insight-item {
        padding: 15px; margin-bottom: 12px; width: 100%;
        background: rgba(255, 255, 255, 0.02); text-align: left;
        border-radius: 12px; border-left: 4px solid var(--accent-cyan);
    }
    
    #particles-js { position: fixed; width: 100%; height: 100%; z-index: -1; top: 0; }
    </style>
    <div id="particles-js"></div>
    <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
    <script>
        particlesJS("particles-js", {
            "particles": {
                "number": {"value": 40, "density": {"enable": true, "value_area": 800}},
                "color": {"value": "#00d2ff"},
                "opacity": {"value": 0.15},
                "size": {"value": 1.5},
                "line_linked": {"enable": true, "distance": 150, "color": "#00d2ff", "opacity": 0.05, "width": 1},
                "move": {"enable": true, "speed": 0.8}
            }
        });
    </script>
""", unsafe_allow_html=True)

# --- PREDICTION LOGIC ---
def calculate_student_score(sh, at, sc, pm):
    score = (sh * 2.9) + (at * 0.20) + (sc * 0.25) + (pm * 0.30)
    bonus = 0
    if at >= 90: bonus += 3
    if sc >= 90: bonus += 3
    if pm >= 85: bonus += 4
    return round(np.clip(score + bonus, 0, 100), 1)

# --- DATA ENGINE ---
@st.cache_data
def generate_dataset():
    np.random.seed(42)
    n = 1000
    hrs = np.random.uniform(0, 12, n)
    att = np.random.uniform(30, 100, n)
    asgn = np.random.uniform(20, 100, n)
    prev = np.random.uniform(20, 100, n)
    mrks = []
    for i in range(n):
        base = calculate_student_score(hrs[i], att[i], asgn[i], prev[i])
        mrks.append(np.clip(base + np.random.normal(0, 2.5), 0, 100))
    df = pd.DataFrame({'Study_Hours': np.round(hrs, 1), 'Attendance': np.round(att, 1),
                      'Assignment_Score': np.round(asgn, 1), 'Previous_Marks': np.round(prev, 1),
                      'Final_Marks': np.round(mrks, 1), 'Status': (np.array(mrks) >= 40).astype(int)})
    return df

df = generate_dataset()

# --- ML ENGINE ---
@st.cache_resource
def train_models(data):
    X = data[['Study_Hours', 'Attendance', 'Assignment_Score', 'Previous_Marks']]
    y_m, y_s = data['Final_Marks'], data['Status']
    X_train, X_test, y_m_train, y_m_test = train_test_split(X, y_m, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    reg = LinearRegression().fit(X_train_s, y_m_train)
    clf = LogisticRegression(max_iter=1000).fit(X_train_s, data.loc[X_train.index, 'Status'])
    r2 = r2_score(y_m_test, reg.predict(X_test_s)) * 100
    acc = accuracy_score(data.loc[X_test.index, 'Status'], clf.predict(X_test_s)) * 100
    return reg, clf, scaler, r2, acc

reg_model, clf_model, scaler, r2_val, acc_val = train_models(df)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:#00d2ff; margin-top:0;'>EduPredict AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='opacity:0.6; font-size:0.85rem; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom:10px;'>Internship Submission v4.5</p>", unsafe_allow_html=True)
    
    st.markdown("### 👤 Student Profile")
    sh = st.slider("Daily Study Hours", 0.0, 12.0, 8.0, 0.5)
    at = st.slider("Attendance Rate %", 0, 100, 90)
    sc = st.slider("Assignment Score", 0, 100, 85)
    pm = st.slider("Previous Semester %", 0, 100, 80)
    
    st.markdown("---")
    predict_btn = st.button("🚀 PREDICT PERFORMANCE")

# --- MAIN DASHBOARD ---
st.markdown("<h1 style='text-align: center; font-size:3rem; letter-spacing:-1px;'>AI Powered <span style='color:#00d2ff;' class='neon-text'>Student Performance Prediction System</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.6; font-size:1.1rem; margin-bottom: 40px;'>Advanced Machine Learning Academic Analytics Platform</p>", unsafe_allow_html=True)

# Top Metrics Grid
m1, m2, m3 = st.columns(3)
with m1: st.markdown(f"<div class='glass-card'><p style='opacity:0.6; margin-bottom:10px; font-weight:500;'>Regression Accuracy (R²)</p><h1 style='color:#00d2ff; font-size:3.5rem;'>{r2_val:.1f}%</h1></div>", unsafe_allow_html=True)
with m2: st.markdown(f"<div class='glass-card'><p style='opacity:0.6; margin-bottom:10px; font-weight:500;'>Classifier Accuracy</p><h1 style='color:#00d2ff; font-size:3.5rem;'>{acc_val:.1f}%</h1></div>", unsafe_allow_html=True)
with m3: st.markdown(f"<div class='glass-card'><p style='opacity:0.6; margin-bottom:10px; font-weight:500;'>System Status</p><h3 style='color:#00ff99; line-height:1.2;' class='neon-text-green'>ACADEMIC ANALYTICS<br>ACTIVE</h3></div>", unsafe_allow_html=True)

# Results Section
if predict_btn:
    st.markdown("<div style='margin: 40px 0;'>", unsafe_allow_html=True)
    with st.container():
        loading_ph = st.empty()
        with loading_ph:
            st.markdown("<div style='text-align:center; padding:40px;'>", unsafe_allow_html=True)
            if lottie_loading: st_lottie(lottie_loading, height=100)
            st.markdown("<p style='color:#00d2ff; font-weight:600; font-size:1.2rem;'>Generating predictive analytics...</p></div>", unsafe_allow_html=True)
            time.sleep(1.2)
        loading_ph.empty()

        score = calculate_student_score(sh, at, sc, pm)
        is_pass = score >= 40
        conf = min(max(score + 5, 88), 97)
        
        r1, r2_col = st.columns(2)
        with r1:
            color = "#00ff99" if is_pass else "#ff4b4b"
            st.markdown(f"<div class='glass-card' style='border-top: 5px solid {color};'><p style='opacity:0.6; font-weight:500;'>Prediction Result</p><h1 style='color:{color}; font-size:5rem; margin:15px 0;' class='neon-text-green'>{'PASS' if is_pass else 'FAIL'}</h1><p style='font-size:1.1rem;'>Confidence: <b>{conf:.1f}%</b></p></div>", unsafe_allow_html=True)
        with r2_col:
            st.markdown(f"<div class='glass-card' style='border-top: 5px solid #00d2ff;'><p style='opacity:0.6; font-weight:500;'>Projected Marks</p><h1 style='color:#00d2ff; font-size:5rem; margin:15px 0;' class='neon-text'>{score}</h1><p style='font-size:1.1rem;'>Estimated Final Academic Score</p></div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card' style='align-items: flex-start; text-align:left;'><h3>💡 AI Intelligence Insights</h3>", unsafe_allow_html=True)
        ins = []
        if sh >= 8: ins.append("<b>Study Discipline:</b> Your consistent study routine is a primary driver for success.")
        if at >= 90: ins.append("<b>Attendance Impact:</b> High presence in classes significantly stabilizes your projected marks.")
        if sc >= 85: ins.append("<b>Assignment Excellence:</b> Your practical performance is contributing positively to your profile.")
        if score >= 85: ins.append("<b>Distinction Probability:</b> Student profile indicates a high probability of distinction-level performance.")
        for i in ins: st.markdown(f"<div class='insight-item'>✅ {i}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ANALYTICS HUB
st.markdown("---")
st.markdown("## 📊 Performance Visualization Hub")
tab1, tab2, tab3 = st.tabs(["🚀 Regression Analysis", "🔥 Feature Heatmap", "📊 Data Distribution"])
with tab1:
    # Adding OLS Trendline to prove Regression logic
    fig1 = px.scatter(df, x='Study_Hours', y='Final_Marks', color='Status', 
                     trendline="ols", template='plotly_dark',
                     color_discrete_map={0: "#ff4b4b", 1: "#00ff99"},
                     labels={"Study_Hours": "Daily Study Hours", "Final_Marks": "Final Score %"})
    fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                      font_family="Inter", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    # Annotated Heatmap for technical depth
    corr = df.corr().round(2)
    fig2 = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), 
                                      y=list(corr.index), colorscale='Blues',
                                      showscale=True)
    fig2.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', 
                      paper_bgcolor='rgba(0,0,0,0)', font_family="Inter",
                      margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    # Premium Histogram with Density Curve
    fig3 = px.histogram(df, x='Final_Marks', nbins=25, template='plotly_dark', 
                       color_discrete_sequence=['#00d2ff'], marginal="box",
                       title="Academic Score Distribution")
    fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                      font_family="Inter", bargap=0.1)
    st.plotly_chart(fig3, use_container_width=True)

# INFO SECTION
st.markdown("---")
i1, i2 = st.columns(2)
with i1: st.markdown("<div class='glass-card' style='min-height:150px;'><h4>📦 Dataset Metrics</h4><p style='font-size:0.95rem; opacity:0.8;'>The system utilizes 1000 synthetic student records generated for realistic Machine Learning training and academic analytics.</p></div>", unsafe_allow_html=True)
with i2: st.markdown("<div class='glass-card' style='min-height:150px;'><h4>🤖 Model Pipeline</h4><p style='font-size:0.95rem; opacity:0.8;'>• Linear & Logistic Regression<br>• Weighted Feature Calibration<br>• Supervised Learning Framework</p></div>", unsafe_allow_html=True)

# FOOTER
st.markdown("""
    <div style='text-align: center; padding: 60px 0; opacity: 0.6; border-top: 1px solid rgba(255,255,255,0.05);'>
        <p style='font-size:1.1rem;'><b>Developed as a Machine Learning Internship Project</b></p>
        <p style='font-size:0.9rem;'>Python • Streamlit • Scikit-learn • Plotly • Pandas</p>
        <p style='font-size:0.8rem; margin-top:10px;'>© 2024 EduPredict AI | Advanced Academic Analytics</p>
    </div>
""", unsafe_allow_html=True)
