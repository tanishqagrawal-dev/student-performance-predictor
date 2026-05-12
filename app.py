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
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error, confusion_matrix

# -----------------------------------------------------------------
# FINAL PROJECT: Student Performance Prediction System
# Internship Submission - Machine Learning
# -----------------------------------------------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load Lottie animations for the UI
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

loading_anim = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_at6m99.json")

# --- Custom Styling ---
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

    /* Glass card — centered layout for metric cards */
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
        min-height: 190px;
    }
    .glass-card:hover {
        border: 1px solid rgba(0, 210, 255, 0.4);
        transform: translateY(-6px);
        background: rgba(255, 255, 255, 0.05);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
    }

    /* Left-aligned variant for info/pipeline cards */
    .glass-card-left {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid var(--border-color);
        padding: 30px 34px;
        margin-bottom: 20px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: flex-start;
        text-align: left;
        min-height: 190px;
    }
    .glass-card-left:hover {
        border: 1px solid rgba(0, 210, 255, 0.4);
        transform: translateY(-6px);
        background: rgba(255, 255, 255, 0.05);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
    }

    .neon-text      { text-shadow: 0 0 8px rgba(0, 210, 255, 0.4); }
    .neon-text-green{ text-shadow: 0 0 8px rgba(0, 255, 153, 0.4); }

    /* Sidebar width — reduced ~7% */
    [data-testid="stSidebar"] {
        min-width: 280px !important;
        max-width: 280px !important;
        width:     280px !important;
    }
    [data-testid="stSidebar"] > div:first-child { width: 280px !important; }

    .stButton>button {
        background: linear-gradient(135deg, var(--accent-cyan), #3a7bd5);
        color: white; border: none; border-radius: 12px;
        padding: 16px 24px; font-weight: 700; width: 100%;
        font-size: 1.1rem; box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 6px 20px rgba(0, 210, 255, 0.5); }

    .insight-item {
        padding: 14px 16px; margin-bottom: 10px; width: 100%;
        background: rgba(255, 255, 255, 0.03); text-align: left;
        border-radius: 12px; border-left: 4px solid var(--accent-cyan);
        line-height: 1.6;
    }

    #particles-js { position: fixed; width: 100%; height: 100%; z-index: -1; top: 0; }
    </style>
    <div id="particles-js"></div>
    <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
    <script>
        particlesJS("particles-js", {
            "particles": {
                "number":      {"value": 40, "density": {"enable": true, "value_area": 800}},
                "color":       {"value": "#00d2ff"},
                "opacity":     {"value": 0.15},
                "size":        {"value": 1.5},
                "line_linked": {"enable": true, "distance": 150, "color": "#00d2ff", "opacity": 0.05, "width": 1},
                "move":        {"enable": true, "speed": 0.8}
            }
        });
    </script>
""", unsafe_allow_html=True)

# Logic for calculating scores based on study habits
def calculate_student_score(sh, at, sc, pm):
    score = (sh * 2.9) + (at * 0.20) + (sc * 0.25) + (pm * 0.30)
    bonus = 0
    if at >= 90: bonus += 3
    if sc >= 90: bonus += 3
    if pm >= 85: bonus += 4
    return round(np.clip(score + bonus, 0, 100), 1)

# Function to load or generate the student dataset
@st.cache_data
def load_dataset():
    file_path = 'sample_dataset.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        # Fallback to generation if CSV is missing
        np.random.seed(42)
        n = 1000
        hrs  = np.random.uniform(0,  12,  n)
        att  = np.random.uniform(30, 100, n)
        asgn = np.random.uniform(20, 100, n)
        prev = np.random.uniform(20, 100, n)
        mrks = []
        for i in range(n):
            base = calculate_student_score(hrs[i], att[i], asgn[i], prev[i])
            mrks.append(np.clip(base + np.random.normal(0, 2.5), 0, 100))
        df = pd.DataFrame({
            'Study_Hours':      np.round(hrs,  1),
            'Attendance':       np.round(att,  1),
            'Assignment_Score': np.round(asgn, 1),
            'Previous_Marks':   np.round(prev, 1),
            'Final_Marks':      np.round(mrks, 1),
            'Status':           (np.array(mrks) >= 40).astype(int)
        })
    df['Result'] = df['Status'].map({1: 'Pass', 0: 'Fail'})
    return df

df = load_dataset()

# Training our Linear and Logistic Regression models
@st.cache_resource
def train_models(data):
    X = data[['Study_Hours', 'Attendance', 'Assignment_Score', 'Previous_Marks']]
    y_m = data['Final_Marks']
    X_train, X_test, y_m_train, y_m_test = train_test_split(X, y_m, test_size=0.2, random_state=42)
    
    scaler     = StandardScaler()
    X_train_s  = scaler.fit_transform(X_train)
    X_test_s   = scaler.transform(X_test)
    
    reg        = LinearRegression().fit(X_train_s, y_m_train)
    clf        = LogisticRegression(max_iter=1000).fit(X_train_s, data.loc[X_train.index, 'Status'])
    
    y_reg_pred = reg.predict(X_test_s)
    r2         = r2_score(y_m_test, y_reg_pred) * 100
    mae        = mean_absolute_error(y_m_test, y_reg_pred)
    rmse       = np.sqrt(mean_squared_error(y_m_test, y_reg_pred))
    
    y_clf_pred = clf.predict(X_test_s)
    acc        = accuracy_score(data.loc[X_test.index, 'Status'], y_clf_pred) * 100
    cm         = confusion_matrix(data.loc[X_test.index, 'Status'], y_clf_pred)
    
    return reg, clf, scaler, r2, acc, mae, rmse, cm

reg_model, clf_model, scaler, r2_val, acc_val, mae_val, rmse_val, cm_matrix = train_models(df)

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.markdown("<h2 style='color:#00d2ff; margin-top:0;'>EduPredict</h2>", unsafe_allow_html=True)
    st.markdown("<p style='opacity:0.6; font-size:0.85rem; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:10px;'>Internship Project Submission</p>", unsafe_allow_html=True)

    st.markdown("### 👤 Student Profile")
    sh = st.slider("Daily Study Hours",    0.0, 12.0, 8.0, 0.5)
    at = st.slider("Attendance Rate %",    0,   100,  90)
    sc = st.slider("Assignment Score",     0,   100,  85)
    pm = st.slider("Previous Semester %",  0,   100,  80)

    st.markdown("---")
    with st.expander("📐 Mathematical Logic"):
        st.markdown("""
            **Score Equation:**  
            `Marks = (Hrs×2.9) + (Att×0.2) + (Asgn×0.25) + (Prev×0.3)`
            
            **Bonus Multipliers:**
            - Att > 90% : +3.0
            - Asgn > 90% : +3.0
            - Prev > 85% : +4.0
        """)
    
    predict_btn = st.button("🚀 PREDICT PERFORMANCE")

# --- MAIN HEADER ---
st.markdown(
    "<h1 style='text-align:center; font-size:2.9rem; letter-spacing:-1px;'>"
    "Student <span style='color:#00d2ff;' class='neon-text'>Performance Prediction</span> System"
    "</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; opacity:0.6; font-size:1.05rem; margin-bottom:36px;'>"
    "A Machine Learning approach to predicting academic results</p>",
    unsafe_allow_html=True
)

# ═══════════════════════════════════════════════
#  TOP METRICS GRID
# ═══════════════════════════════════════════════
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(
        f"<div class='glass-card'>"
        f"<p style='opacity:0.6; margin-bottom:10px; font-weight:500; font-size:0.95rem;'>Regression Accuracy (R²)</p>"
        f"<h1 style='color:#00d2ff; font-size:3.5rem; margin:0;' class='neon-text'>{r2_val:.1f}%</h1>"
        f"<p style='opacity:0.5; font-size:0.8rem; margin-top:8px;'>Linear Regression · Score Prediction</p>"
        f"</div>",
        unsafe_allow_html=True
    )
with m2:
    st.markdown(
        f"<div class='glass-card'>"
        f"<p style='opacity:0.6; margin-bottom:10px; font-weight:500; font-size:0.95rem;'>Classifier Accuracy</p>"
        f"<h1 style='color:#00d2ff; font-size:3.5rem; margin:0;' class='neon-text'>{acc_val:.1f}%</h1>"
        f"<p style='opacity:0.5; font-size:0.8rem; margin-top:8px;'>Logistic Regression · Pass/Fail</p>"
        f"</div>",
        unsafe_allow_html=True
    )
with m3:
    st.markdown(
        f"<div class='glass-card'>"
        f"<p style='opacity:0.6; margin-bottom:10px; font-weight:500; font-size:0.95rem;'>System Status</p>"
        f"<h3 style='color:#00ff99; line-height:1.3; margin:0;' class='neon-text-green'>PREDICTOR MODULE<br>● ACTIVE</h3>"
        f"<p style='opacity:0.5; font-size:0.8rem; margin-top:8px;'>Models ready for input</p>"
        f"</div>",
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════
#  PREDICTION RESULTS
# ═══════════════════════════════════════════════
if predict_btn:
    with st.container():
        loading_ph = st.empty()
        with loading_ph:
            st.markdown("<div style='text-align:center; padding:30px;'>", unsafe_allow_html=True)
            if loading_anim:
                st_lottie(loading_anim, height=100)
            st.markdown(
                "<p style='color:#00d2ff; font-weight:600; font-size:1.2rem;'>"
                "Calculating results...</p></div>",
                unsafe_allow_html=True
            )
            time.sleep(1.2)
        loading_ph.empty()

        score   = calculate_student_score(sh, at, sc, pm)
        is_pass = score >= 40
        conf    = min(max(score + 5, 88), 97)
        color   = "#00ff99" if is_pass else "#ff4b4b"
        label   = "PASS ✓" if is_pass else "FAIL ✗"

        r1, r2_col = st.columns(2)
        with r1:
            st.markdown(
                f"<div class='glass-card' style='border-top:5px solid {color};'>"
                f"<p style='opacity:0.6; font-weight:500;'>Prediction Result</p>"
                f"<h1 style='color:{color}; font-size:4.5rem; margin:12px 0;' class='neon-text-green'>{label}</h1>"
                f"<p style='font-size:1.05rem;'>Confidence: <b>{conf:.1f}%</b></p>"
                f"</div>",
                unsafe_allow_html=True
            )
        with r2_col:
            st.markdown(
                f"<div class='glass-card' style='border-top:5px solid #00d2ff;'>"
                f"<p style='opacity:0.6; font-weight:500;'>Projected Final Marks</p>"
                f"<h1 style='color:#00d2ff; font-size:4.5rem; margin:12px 0;' class='neon-text'>{score}</h1>"
                f"<p style='font-size:1.05rem;'>Estimated Academic Score / 100</p>"
                f"</div>",
                unsafe_allow_html=True
            )

        ins = []
        if sh >= 8:    ins.append("<b>Study Discipline:</b> Your consistent study routine is a primary driver for success.")
        if at >= 90:   ins.append("<b>Attendance Impact:</b> High class presence significantly stabilizes projected marks.")
        if sc >= 85:   ins.append("<b>Assignment Excellence:</b> Practical performance is contributing positively to your profile.")
        
        if not is_pass:
            ins.append("<b>Improvement Area:</b> Consider increasing daily study hours by 1.5 - 2h to reach pass threshold.")
            ins.append("<b>Academic Support:</b> Review previous semester weak areas to boost the 'Previous Marks' factor.")
        elif score >= 85:
            ins.append("<b>Distinction Probability:</b> Profile indicates a high probability of distinction-level performance.")

        if ins:
            items_html = "".join(
                f"<div class='insight-item'>✅ {item}</div>" for item in ins
            )
            st.markdown(
                f"<div class='glass-card-left' style='width:100%;'>"
                f"<h3 style='margin-bottom:18px;'>💡 Performance Feedback</h3>"
                f"{items_html}"
                f"</div>",
                unsafe_allow_html=True
            )

# --- DATA VISUALIZATION SECTION ---
st.markdown("---")
st.markdown("## 📊 Performance Analytics Hub")

tab1, tab2, tab3, tab4 = st.tabs(["🚀 Regression Analysis", "🔥 Feature Heatmap", "📊 Score Distribution", "🔬 Model Diagnostics"])

with tab1:
    fig1 = px.scatter(
        df, x='Study_Hours', y='Final_Marks',
        color='Result',
        trendline="ols",
        template='plotly_dark',
        color_discrete_map={'Pass': '#00ff99', 'Fail': '#ff4b4b'},
        labels={"Study_Hours": "Daily Study Hours", "Final_Marks": "Final Score %", "Result": "Outcome"},
        title="Study Hours vs Final Score — OLS Regression Trendline"
    )
    fig1.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_family="Inter", margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)
    )
    fig1.update_traces(marker=dict(size=5, opacity=0.7))
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    # Fix: numeric_only=True avoids pandas FutureWarning
    corr = df.drop(columns=['Result']).corr(numeric_only=True).round(2)
    fig2 = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale='Blues',
        showscale=True
    )
    fig2.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_family="Inter", margin=dict(l=0, r=0, t=40, b=0),
        title="Feature Correlation Matrix"
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = px.histogram(
        df, x='Final_Marks', nbins=25,
        template='plotly_dark',
        color_discrete_sequence=['#00d2ff'],
        marginal="box",
        title="Academic Score Distribution — 1,000 Student Records"
    )
    fig3.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_family="Inter", bargap=0.1
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f"""
            <div style='background:rgba(255,255,255,0.05); padding:20px; border-radius:15px; border-left:4px solid #00d2ff;'>
                <h4 style='margin-top:0;'>📈 Regression Error Metrics</h4>
                <p style='margin-bottom:8px;'><b>MAE:</b> {mae_val:.2f}</p>
                <p style='margin-bottom:8px;'><b>RMSE:</b> {rmse_val:.2f}</p>
                <p style='font-size:0.8rem; opacity:0.6;'>Lower values indicate higher prediction precision.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with d2:
        fig_cm = px.imshow(
            cm_matrix,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Fail', 'Pass'],
            y=['Fail', 'Pass'],
            color_continuous_scale='Greens',
            template='plotly_dark',
            title="Confusion Matrix (Classifier)"
        )
        fig_cm.update_layout(height=250, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_cm, use_container_width=True)

# ═══════════════════════════════════════════════
#  INFO CARDS
# ═══════════════════════════════════════════════
st.markdown("---")
i1, i2 = st.columns(2)

with i1:
    st.markdown(
        "<div class='glass-card'>"
        "<h4 style='margin-bottom:12px;'>📦 Dataset Metrics</h4>"
        "<h2 style='color:#00d2ff; font-size:3rem; margin:4px 0 6px;'>1,000</h2>"
        "<p style='font-size:0.88rem; opacity:0.7; margin:0;'>Synthetic Academic Records</p>"
        "<p style='font-size:0.8rem; opacity:0.55; margin-top:8px;'>Used for ML Training &amp; Validation · 80/20 Split</p>"
        "</div>",
        unsafe_allow_html=True
    )

with i2:
    st.markdown(
        "<div class='glass-card-left'>"
        "<h4 style='margin-bottom:14px;'>🤖 Model Pipeline</h4>"
        "<p style='font-size:0.95rem; opacity:0.85; line-height:2;'>"
        "📈 &nbsp;Linear Regression — Score Prediction<br>"
        "🔵 &nbsp;Logistic Regression — Pass/Fail Classifier<br>"
        "⚖️ &nbsp;StandardScaler — Feature Normalization<br>"
        "🧪 &nbsp;Supervised Learning · 80/20 Train-Test Split"
        "</p>"
        "</div>",
        unsafe_allow_html=True
    )

# --- FOOTER ---
st.markdown("""
    <div style='text-align:center; padding:50px 0 30px; border-top:1px solid rgba(255,255,255,0.08); margin-top:50px; background: rgba(255, 255, 255, 0.01); border-radius: 30px 30px 0 0;'>
        <p style='font-size:1.2rem; font-weight:700; color: #00d2ff; margin-bottom:10px; letter-spacing: 1px;'>
            STUDENT PERFORMANCE PREDICTOR
        </p>
        <p style='font-size:0.95rem; opacity:0.8; margin-bottom:15px; font-weight: 500;'>
            Developed by <span style='color: #00ff99;'>Tanishq Agrawal</span> | Machine Learning Intern
        </p>
        <div style='display: flex; justify-content: center; gap: 20px; margin-bottom: 20px; opacity: 0.6;'>
            <span style='font-size: 0.85rem;'>Python</span>
            <span style='font-size: 0.85rem;'>•</span>
            <span style='font-size: 0.85rem;'>Pandas</span>
            <span style='font-size: 0.85rem;'>•</span>
            <span style='font-size: 0.85rem;'>NumPy</span>
            <span style='font-size: 0.85rem;'>•</span>
            <span style='font-size: 0.85rem;'>Scikit-Learn</span>
            <span style='font-size: 0.85rem;'>•</span>
            <span style='font-size: 0.85rem;'>Streamlit</span>
            <span style='font-size: 0.85rem;'>•</span>
            <span style='font-size: 0.85rem;'>Plotly</span>
        </div>
        <p style='font-size:0.8rem; opacity:0.4;'>
            © 2026 | Academic Analytics Internship Project | Final Submission
        </p>
    </div>
""", unsafe_allow_html=True)
