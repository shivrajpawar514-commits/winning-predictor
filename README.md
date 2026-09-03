# 🏏 IPL Match Predictor — Winning Team & Score Analytics Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning web application that predicts **real-time winning probabilities** and **first-innings projected scores** for Indian Premier League (IPL) matches. Built using **Streamlit** and powered by **Random Forest Ensemble Models** trained on extensive historical IPL match and delivery datasets.

---

## 🌟 Key Features

- **🏆 2nd Innings Win Probability (Classification)**:
  - Real-time winning chance calculation for the chasing batting team vs defending bowling team.
  - Dynamically updates as the match progresses ball-by-ball.
  - Displays dual-team probability bars and strategic situational breakdown.
- **📊 1st Innings Score Projection (Regression)**:
  - Predicts final projected total score by the end of 20 overs.
  - Calculates estimated safe target ranges and projected run rates.
- **⚡ Live Match Analytics Dashboard**:
  - Live **Current Run Rate (CRR)** & **Required Run Rate (RRR)** calculation.
  - Real-time remaining balls & runs required indicators.
- **🎨 Modern Dark UI & Fast Model Caching**:
  - Glassmorphic design with intuitive side-by-side match configuration.
  - In-memory model caching with `@st.cache_resource` for instantaneous predictions.

---

## 🛠️ Tech Stack & Architecture

- **Frontend**: [Streamlit](https://streamlit.io/) with custom CSS & dark theme
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (`RandomForestClassifier`, `RandomForestRegressor`)
- **Data Engineering**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Model Serialization**: [Joblib](https://joblib.readthedocs.io/)

### 🧠 ML Pipeline Overview
```
Deliveries Data + Matches Data
       │
       ▼
Feature Engineering (CRR, Wickets, Overs, Target, Teams)
       │
       ├──────────────────────────────┐
       ▼                              ▼
Random Forest Classifier       Random Forest Regressor
(2nd Innings Win Probability)  (1st Innings Total Score)
```

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/shivrajpawar514-commits/winning-predictor.git
cd winning-predictor
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Train / Re-train Models
```bash
python train_model.py
```

### 5. Launch the Web App
```bash
streamlit run app.py
```
Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Repository Structure

```
├── app.py                   # Streamlit web application & prediction UI
├── train_model.py           # ML training script for classifier & regressor
├── requirements.txt         # Pinned Python package dependencies
├── matches.csv              # Historical IPL match data
├── deliveries.csv           # Historical ball-by-ball delivery data
├── .streamlit/
│   └── config.toml          # Streamlit UI theme configuration
├── .gitignore               # Git ignore patterns for venv & cache
├── LICENSE                  # MIT License
└── README.md                # Project documentation
```

---

## 👨‍💻 Author

Created with ❤️ for IPL analytics by **Shivraj** ([@shivrajpawar514-commits](https://github.com/shivrajpawar514-commits)).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).