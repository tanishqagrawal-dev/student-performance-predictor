# EduPredict AI — Student Performance Prediction System 🎓

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Developed as a Machine Learning Internship Project**  
> Predicts whether a student will **Pass or Fail**, and estimates **final marks**, based on study habits and academic behaviour.

---

## 📌 Project Overview

**EduPredict AI** is a full end-to-end Machine Learning web application built using Python and Streamlit. It demonstrates a complete ML pipeline — from data generation and preprocessing, through exploratory data analysis (EDA) and model training, to live interactive predictions.

The project fulfils all requirements of the **Machine Learning Internship — Project 1: Student Performance Prediction System**.

---

## 🎯 Objective

> *Predict whether a student will pass/fail or estimate marks based on study habits.*

The system accepts four student profile inputs:
- 📚 Daily Study Hours
- 🏫 Attendance Rate (%)
- 📝 Assignment Score
- 📊 Previous Semester Marks

And outputs:
- ✅ **Pass / Fail prediction** with confidence percentage
- 📈 **Projected final marks** out of 100
- 💡 **AI-generated insights** based on the student profile

---

## 🧠 ML Concepts Demonstrated

| # | Concept | Implementation |
|---|---|---|
| 1 | **Data Preprocessing** | Feature clipping, rounding, normalization via `StandardScaler` |
| 2 | **Exploratory Data Analysis (EDA)** | 3 interactive Plotly charts (Scatter, Heatmap, Histogram) |
| 3 | **Linear Regression** | `sklearn.LinearRegression` — predicts final score (continuous output) |
| 4 | **Logistic Regression** | `sklearn.LogisticRegression` — classifies Pass (1) / Fail (0) |
| 5 | **Feature Scaling** | `StandardScaler` applied to all 4 input features before model training |

---

## ⚙️ ML Pipeline — Step by Step

Following the internship project specification:

**Step 1 — Load Dataset using Pandas**
```
1,000 synthetic student records generated with NumPy (seed=42 for reproducibility)
Loaded into a pandas DataFrame with 6 columns
```

**Step 2 — Clean and Preprocess Data**
```
- Values bounded using np.clip() to realistic academic ranges
- Features rounded to 1 decimal place
- StandardScaler applied to normalize all input features
```

**Step 3 — Perform EDA using Visualization**
```
- Scatter Plot: Study Hours vs Final Score with OLS Regression Trendline
- Correlation Heatmap: Annotated 6×6 feature correlation matrix
- Distribution Plot: Histogram + Boxplot of Final Marks across 1,000 records
```

**Step 4 — Split Data into Training and Testing Sets**
```
- 80% Training  →  800 records
- 20% Testing   →  200 records
- random_state=42 for reproducibility
```

**Step 5 — Train Model and Evaluate Accuracy**
```
- Linear Regression  →  R² Score: ~96%
- Logistic Regression →  Classifier Accuracy: ~97%
```

---

## 📊 Model Performance

| Model | Algorithm | Metric | Score |
|---|---|---|---|
| Score Predictor | Linear Regression | R² Score | **~96%** |
| Pass/Fail Classifier | Logistic Regression | Accuracy | **~97%** |

> Both metrics are computed on the **held-out test set (20%)** and displayed live on the dashboard.

---

## 🎯 Key Features

- **Real-time Prediction** — Adjust sliders to see instant Pass/Fail + score results
- **AI Smart Insights** — Contextual feedback based on student profile inputs
- **3 EDA Visualizations** — Scatter (OLS), Heatmap, Score Distribution
- **Model Accuracy Display** — R² and Classifier Accuracy shown as metric cards
- **Premium Glassmorphism UI** — Dark mode with neon accents and particle animations
- **Confidence Scoring** — Model certainty shown for every prediction

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Web Framework** | Streamlit |
| **Machine Learning** | Scikit-learn (LinearRegression, LogisticRegression, StandardScaler) |
| **Data Handling** | Pandas, NumPy |
| **Visualizations** | Plotly Express, Plotly Figure Factory |
| **UI/Animations** | Custom CSS, Particles.js, Streamlit-Lottie |
| **Statistical Trendline** | Statsmodels (OLS via Plotly) |

---

## 📂 Project Structure

```
student-performance-predictor/
├── app.py               # Main Streamlit Application (ML pipeline + UI)
├── generate_data.py     # Script to generate synthetic dataset
├── sample_dataset.csv   # 1,000 Synthetic Student Performance Records
├── requirements.txt     # All Python Dependencies
└── README.md            # Project Documentation
```

---

## ⚙️ How to Run Locally

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/student-performance-predictor.git
cd student-performance-predictor
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the application:**
```bash
python -m streamlit run app.py
```

**4. Open in browser:**
```
http://localhost:8501
```

---

## 📦 Dataset Details

| Property | Value |
|---|---|
| Total Records | 1,000 synthetic student profiles |
| Generation Method | NumPy random generation (seed=42) |
| Features | Study Hours, Attendance, Assignment Score, Previous Marks |
| Target Variables | Final Marks (regression) + Pass/Fail Status (classification) |
| Pass Threshold | Final Marks ≥ 40 |
| Train / Test Split | 80% / 20% |

> The dataset is synthetically generated to ensure reproducibility and controlled distribution while simulating realistic academic scenarios.

---

## 👨‍💻 Internship Project Note

This system was developed to demonstrate end-to-end Machine Learning skills, from raw data generation through to a fully deployed interactive application. It is designed to be clearly explained during a technical interview or viva, covering:

- Why **StandardScaler** is used before training (feature normalization prevents scale bias)
- Why **two separate models** are used (regression for continuous output, classification for binary)
- Why **synthetic data** with `seed=42` ensures reproducibility across environments
- How **R² score** and **Accuracy** are computed on the held-out test set, not training data

---

**Developed as a Machine Learning Internship Project**  
Python • Streamlit • Scikit-learn • Plotly • Pandas • NumPy
