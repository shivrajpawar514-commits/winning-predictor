# 🏏 IPL Match Intelligence & Winning Probability Prediction Engine (v2.0)

[![CI Pipeline](https://github.com/shivrajpawar514-commits/winning-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/shivrajpawar514-commits/winning-predictor/actions)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-EB5424.svg)](https://xgboost.ai/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An **enterprise-grade, end-to-end Machine Learning & AI Engineering platform** for real-time Indian Premier League (IPL) match analytics. Features real-time **2nd innings win probability estimation**, **1st innings score projection**, an interactive **What-If Scenario Simulator**, **Explainable AI (XAI)** model interpretability, a production **FastAPI REST API**, and a modern **Streamlit Web Dashboard**.

---

## 🌟 Key Capabilities & System Features

### 1. 🔮 Real-Time Live Match Predictor
- **Second Innings Win Probability (Classification)**: Computes dynamic chasing vs defending win percentages ball-by-ball.
- **First Innings Total Score (Regression)**: Predicts final projected totals and confidence interval ranges based on match state.
- **Real-Time Cricket Metrics**: Computes **CRR** (Current Run Rate), **RRR** (Required Run Rate), balls remaining, runs needed, and wickets in hand.

### 2. ⚡ Interactive "What-If" Scenario Simulator
- Enables tactical match trajectory simulation across varying run rates (6.0 to 14.0 RPO) and projected wicket losses.
- Visualizes interactive multi-over win probability curves using Plotly.

### 3. 🧠 Explainable AI (XAI) & Interpretability
- Provides normalized feature importance distributions explaining the underlying drivers (target score, cumulative runs, wickets fallen, and overs completed).

### 4. 📊 Historical IPL Exploratory Data Analysis (EDA)
- Analyzes franchise win histories and toss decision advantages across historical IPL seasons.

### 5. 🔌 Production REST API (FastAPI)
- High-performance, fully documented API with Pydantic V2 validation, automated OpenAPI/Swagger docs (`/docs`), and health checks.

---

## 🏗️ System Architecture

```
                                  ┌───────────────────────────┐
                                  │   Raw IPL Match Datasets   │
                                  │ (deliveries.csv, matches) │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │    Data Ingestion & Clean │
                                  │      (src/data_loader)    │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │    Feature Engineering    │
                                  │ (CRR, RRR, Rolling Stats) │
                                  └─────────────┬─────────────┘
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     ▼                                                     ▼
        ┌─────────────────────────┐                           ┌─────────────────────────┐
        │ Random Forest + XGBoost │                           │ Random Forest + XGBoost │
        │ Classifier (Win Prob)   │                           │ Regressor (Score Proj)  │
        └────────────┬────────────┘                           └────────────┬────────────┘
                     │                                                     │
                     └──────────────────────────┬──────────────────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │ Production Inference &    │
                                  │ Explainability Engine     │
                                  │ (src/predictor, XAI)      │
                                  └─────────────┬─────────────┘
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     ▼                                                     ▼
        ┌─────────────────────────┐                           ┌─────────────────────────┐
        │  FastAPI REST Service   │                           │ Streamlit Web Dashboard │
        │       (api.py)          │                           │        (app.py)         │
        └─────────────────────────┘                           └─────────────────────────┘
```

---

## 📊 Model Performance Benchmarks & Calibration

### 1. Leak-Free Evaluation (Match-Level Split)
To eliminate data leakage caused by random row splitting (where deliveries from the same match appear in both training and test sets), the pipeline employs **GroupShuffleSplit on match ID** (873 train matches vs 219 test matches) ensuring zero match contamination.

| Task | Architecture | Accuracy / R² | ROC-AUC / RMSE | Brier Score | ECE (Calib Err) | Log Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Win Probability (2nd Innings)** | **Gradient Boosting Classifier** | **76.9%** | **0.8570** | **0.1532** | **0.0541** | **0.4656** |
| **Win Probability (2nd Innings)** | Random Forest Classifier | 74.0% | 0.8347 | 0.1660 | 0.0634 | 0.5000 |
| **Score Projection (1st Innings)**| **Gradient Boosting Regressor** | **0.5032 R²** | **21.02 RMSE** | — | — | MAE: 15.49 |
| **Score Projection (1st Innings)**| Random Forest Regressor | 0.4231 R² | 22.66 RMSE | — | — | MAE: 16.65 |

> **Calibration & Reliability**: Probability estimates are evaluated using **Brier Score Loss** (0.1532) and **Expected Calibration Error (ECE: 5.41%)** across 10 probability deciles. Calibration curves and reliability tables are exported to `model_metadata.json`.

### 2. Baseline Row-Level Evaluation (Optimistic Leakage Reference)
When trained with unconstrained row-level random sampling (test_size=0.2, random_state=42):
- **Win Probability (2nd Innings)**: Random Forest 82.4% Accuracy, 0.884 ROC-AUC | XGBoost 81.9% Accuracy, 0.879 ROC-AUC
- **Score Projection (1st Innings)**: Random Forest 0.781 R², 15.2 RMSE | XGBoost 0.765 R², 16.1 RMSE

---

## 🔬 Reproducing Reported Benchmarks

### Dataset Source
- **Primary Source**: Kaggle IPL Complete Dataset ([deliveries.csv](deliveries.csv) and [matches.csv](matches.csv)).
- **Volume**: 260,920 delivery records across 1,095 IPL matches spanning seasons 2008 through 2024.

### Preprocessing Steps
1. **Match ID Harmonization**: Standardize delivery identifier (`match_id` -> `id`).
2. **Team Alias Harmonization**: Canonicalize franchise rebrands (e.g. `Delhi Daredevils` -> `Delhi Capitals`, `Kings XI Punjab` -> `Punjab Kings`, `Royal Challengers Bangalore` -> `Royal Challengers Bengaluru`).
3. **Continuous Decimal Overs**: Convert 0-indexed over and delivery number to continuous decimal overs (`overs_decimal = over + ball / 6.0`).
4. **Cumulative Match Aggregation**: Compute cumulative runs and wickets per match and innings.
5. **Target Runs Extraction**: Extract 1st innings total runs as the target for 2nd innings chase records.
6. **Domain Metrics**: Derive CRR, balls bowled, balls remaining, wickets in hand, and runs needed.
7. **Label Encoding**: Fit consistent LabelEncoders on all participating franchises.

### Random Seeds & Evaluation Script
- **Random Seed**: `random_state = 42` across all splits, random forests, and boosting ensembles.
- **Evaluation Scripts**:
  ```bash
  # Leak-free match-level split (recommended)
  python -m src.train --split-strategy match --seed 42

  # Season-based temporal split (train <= 2022, test 2023-2024)
  python -m src.train --split-strategy season --seed 42

  # Legacy standalone script
  python train_model.py
  ```
All generated metrics, reliability tables, and calibration curve coordinates are recorded automatically in `model_metadata.json`.

---

## 🛡️ Security, Validation & Failure Handling

### 1. Robust CORS Configuration
`allow_origins=["*"]` combined with `allow_credentials=True` is an invalid configuration under W3C specifications that modern browsers reject. The API specifies trusted origins explicitly and supports dynamic configuration:
- Default trusted origins: `http://localhost:8501`, `http://127.0.0.1:8501`, `http://localhost:3000`, `http://localhost:8000`.
- Environment variable override: configure comma-separated origins via `ALLOWED_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"`.

### 2. Cricket Overs Validation
In cricket, overs notation represents completed overs and remaining balls (`O.B`):
- `14.2` means **14 overs and 2 balls** (total balls bowled = $14 \times 6 + 2 = 86$ balls; balls remaining = $120 - 86 = 34$; decimal overs = $86 / 6 = 14.333\dots$).
- Decimal inputs like `14.8` or `14.25` are strictly rejected with HTTP 422 / ValueError since an over only contains legal balls $0 \dots 5$.
- All rate calculations (CRR, RRR, balls remaining, DLS par score) convert overs to total legal balls first.

### 3. Comprehensive Failure Handling
- **Missing Model Files**: `/health` returns HTTP 503 (`status: "unhealthy", "models_loaded": false`) with actionable diagnostic instructions instead of crashing.
- **Malformed Requests**: Pydantic V2 validators enforce bounds on runs ($0-500$), wickets ($0-10$), overs ($0.0-20.0$), non-identical teams, and positive targets.
- **Unavailable Datasets**: `load_raw_data` verifies file existence and raises informative `FileNotFoundError` if data files are missing.
- **Model Initialization Errors**: Catches corrupted or incompatible binary artifacts and raises descriptive `RuntimeError`.

---

## 🚀 Quick Start & Installation

### Option A: Local Python Environment

```bash
# 1. Clone repository
git clone https://github.com/shivrajpawar514-commits/winning-predictor.git
cd winning-predictor

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Automated Test Suite
pytest -v

# 5. Launch Streamlit Dashboard
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

### Option B: Run the FastAPI REST API

```bash
uvicorn api:app --reload --port 8000
```
- Interactive Swagger UI: **[http://localhost:8000/docs](http://localhost:8000/docs)**
- Redoc Documentation: **[http://localhost:8000/redoc](http://localhost:8000/redoc)**

---

### Option C: Docker & Docker Compose

Launch both the Streamlit Dashboard (Port 8501) and FastAPI API (Port 8000) in isolated containers with a single command:

```bash
docker-compose up --build
```

---

## 🔌 API Endpoint Documentation

### 1. Predict Win Probability
`POST /predict/win-probability`

```bash
curl -X POST "http://localhost:8000/predict/win-probability" \
     -H "Content-Type: application/json" \
     -d '{
       "batting_team": "Chennai Super Kings",
       "bowling_team": "Mumbai Indians",
       "runs_scored": 130,
       "wickets_lost": 3,
       "overs_completed": 14.2,
       "target_runs": 185
     }'
```

**Response:**
```json
{
  "batting_team": "Chennai Super Kings",
  "bowling_team": "Mumbai Indians",
  "batting_win_probability": 0.684,
  "bowling_win_probability": 0.316,
  "current_run_rate": 9.07,
  "required_run_rate": 9.71,
  "runs_needed": 55,
  "balls_remaining": 34,
  "wickets_in_hand": 7
}
```

---

## 🧪 Automated Testing

The project includes unit and integration tests covering feature engineering, model inference bounds, and API endpoints:

```bash
pytest -v
```

---

## 📁 Repository Structure

```
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI automated pipeline
├── .streamlit/
│   └── config.toml              # Modern dark theme configuration
├── src/                         # Core Machine Learning Package
│   ├── __init__.py
│   ├── config.py                # Paths, constants, and feature definitions
│   ├── data_loader.py           # Ingestion and data cleaning
│   ├── feature_engineering.py   # Domain cricket feature transformations
│   ├── predictor.py             # Production inference engine
│   ├── explainability.py        # XAI & What-If scenario simulation engine
│   └── train.py                 # Multi-model training and benchmarking script
├── tests/                       # Automated Pytest suite
│   ├── test_api.py              # API endpoint integration tests
│   ├── test_feature_engineering.py # Feature transformation unit tests
│   └── test_predictor.py        # Model inference bounds tests
├── api.py                       # Enterprise FastAPI REST service
├── app.py                       # 5-Tab Streamlit Dashboard with Plotly
├── Dockerfile                   # Production Docker image configuration
├── docker-compose.yml           # Multi-container orchestration
├── pytest.ini                   # Pytest path and execution configuration
├── requirements.txt             # Pinned dependencies
├── LICENSE                      # MIT License
└── README.md                    # Enterprise portfolio documentation
```

---

## 👨‍💻 Author

Engineered by **Shivraj** ([@shivrajpawar514-commits](https://github.com/shivrajpawar514-commits)).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).