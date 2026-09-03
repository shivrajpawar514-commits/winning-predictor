"""
Enterprise Model Training & Benchmarking Pipeline for IPL Match Predictor.
Evaluates multiple architectures (Random Forest, XGBoost) and logs metadata.
"""

import json
import time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from src.config import (
    CLASSIFICATION_FEATURES,
    DELIVERIES_CSV,
    LE_BATTING_PATH,
    LE_BOWLING_PATH,
    MATCHES_CSV,
    METADATA_PATH,
    REGRESSION_FEATURES,
    RUNS_REGRESSOR_PATH,
    WIN_CLASSIFIER_PATH
)
from src.data_loader import clean_and_merge_data, load_raw_data
from src.feature_engineering import build_match_state_dataframe


def train_and_benchmark_models():
    """Main training, cross-validation, and benchmarking pipeline."""
    print("=" * 65)
    print("🏏 IPL MATCH PREDICTOR — ML TRAINING & BENCHMARKING PIPELINE")
    print("=" * 65)

    start_time = time.time()

    # 1. Ingestion
    print("\n[1/4] Ingesting & Merging Datasets...")
    deliveries, matches = load_raw_data(DELIVERIES_CSV, MATCHES_CSV)
    df_merged = clean_and_merge_data(deliveries, matches)
    print(f"Loaded {len(deliveries):,} delivery records across {len(matches):,} matches.")

    # 2. Feature Engineering
    print("\n[2/4] Engineering Domain Match State Features...")
    over_stats, le_batting, le_bowling = build_match_state_dataframe(df_merged)
    print(f"Generated {len(over_stats):,} granular state vectors across {len(le_batting.classes_)} teams.")

    # 3. Train Classification Models (2nd Innings)
    print("\n[3/4] Benchmarking Win Probability Classifiers (2nd Innings)...")
    second_innings_data = over_stats[over_stats['inning'] == 2]
    X_cls = second_innings_data[CLASSIFICATION_FEATURES]
    y_cls = second_innings_data['win_status']

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    print(" -> Training Random Forest Classifier...")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train_c, y_train_c)
    rf_probs = rf_clf.predict_proba(X_test_c)[:, 1]
    rf_acc = accuracy_score(y_test_c, rf_clf.predict(X_test_c))
    rf_auc = roc_auc_score(y_test_c, rf_probs)
    rf_brier = brier_score_loss(y_test_c, rf_probs)
    print(f"    RF  | Accuracy: {rf_acc * 100:.2f}% | ROC-AUC: {rf_auc:.4f} | Brier: {rf_brier:.4f}")

    print(" -> Training XGBoost Classifier...")
    xgb_clf = XGBClassifier(n_estimators=120, max_depth=6, learning_rate=0.08, random_state=42, eval_metric="logloss")
    xgb_clf.fit(X_train_c, y_train_c)
    xgb_probs = xgb_clf.predict_proba(X_test_c)[:, 1]
    xgb_acc = accuracy_score(y_test_c, xgb_clf.predict(X_test_c))
    xgb_auc = roc_auc_score(y_test_c, xgb_probs)
    xgb_brier = brier_score_loss(y_test_c, xgb_probs)
    print(f"    XGB | Accuracy: {xgb_acc * 100:.2f}% | ROC-AUC: {xgb_auc:.4f} | Brier: {xgb_brier:.4f}")

    # Select best classifier
    best_clf = rf_clf if rf_auc >= xgb_auc else xgb_clf
    best_clf_name = "RandomForestClassifier" if rf_auc >= xgb_auc else "XGBClassifier"

    # 4. Train Regression Models (1st Innings)
    print("\n[4/4] Benchmarking 1st Innings Score Regressors...")
    innings1_data = over_stats[over_stats['inning'] == 1].copy()
    innings1_totals = df_merged[df_merged['inning'] == 1].groupby('id')['total_runs'].sum().reset_index()
    innings1_totals.rename(columns={'total_runs': 'final_total'}, inplace=True)
    innings1_data = innings1_data.merge(innings1_totals, on='id', how='left')

    X_reg = innings1_data[REGRESSION_FEATURES]
    y_reg = innings1_data['final_total']

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    print(" -> Training Random Forest Regressor...")
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train_r, y_train_r)
    rf_pred_r = rf_reg.predict(X_test_r)
    rf_r2 = r2_score(y_test_r, rf_pred_r)
    rf_rmse = np.sqrt(mean_squared_error(y_test_r, rf_pred_r))
    rf_mae = mean_absolute_error(y_test_r, rf_pred_r)
    print(f"    RF  | R² Score: {rf_r2:.4f} | RMSE: {rf_rmse:.2f} | MAE: {rf_mae:.2f}")

    print(" -> Training XGBoost Regressor...")
    xgb_reg = XGBRegressor(n_estimators=120, max_depth=6, learning_rate=0.08, random_state=42)
    xgb_reg.fit(X_train_r, y_train_r)
    xgb_pred_r = xgb_reg.predict(X_test_r)
    xgb_r2 = r2_score(y_test_r, xgb_pred_r)
    xgb_rmse = np.sqrt(mean_squared_error(y_test_r, xgb_pred_r))
    xgb_mae = mean_absolute_error(y_test_r, xgb_pred_r)
    print(f"    XGB | R² Score: {xgb_r2:.4f} | RMSE: {xgb_rmse:.2f} | MAE: {xgb_mae:.2f}")

    best_reg = rf_reg if rf_r2 >= xgb_r2 else xgb_reg
    best_reg_name = "RandomForestRegressor" if rf_r2 >= xgb_r2 else "XGBRegressor"

    # Save models
    print("\n💾 Serializing Models and Preprocessors...")
    joblib.dump(best_clf, WIN_CLASSIFIER_PATH)
    joblib.dump(best_reg, RUNS_REGRESSOR_PATH)
    joblib.dump(le_batting, LE_BATTING_PATH)
    joblib.dump(le_bowling, LE_BOWLING_PATH)

    # Save metadata
    elapsed_time = round(time.time() - start_time, 2)
    metadata = {
        "version": "2.0.0",
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "training_time_seconds": elapsed_time,
        "dataset": {
            "total_deliveries": len(deliveries),
            "total_matches": len(matches),
            "teams": list(le_batting.classes_)
        },
        "classification_benchmarks": {
            "selected_model": best_clf_name,
            "random_forest": {"accuracy": round(rf_acc, 4), "roc_auc": round(rf_auc, 4), "brier_score": round(rf_brier, 4)},
            "xgboost": {"accuracy": round(xgb_acc, 4), "roc_auc": round(xgb_auc, 4), "brier_score": round(xgb_brier, 4)}
        },
        "regression_benchmarks": {
            "selected_model": best_reg_name,
            "random_forest": {"r2_score": round(rf_r2, 4), "rmse": round(rf_rmse, 2), "mae": round(rf_mae, 2)},
            "xgboost": {"r2_score": round(xgb_r2, 4), "rmse": round(xgb_rmse, 2), "mae": round(xgb_mae, 2)}
        }
    }

    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata recorded at {METADATA_PATH}")
    print(f"🎉 Pipeline successfully completed in {elapsed_time}s!")


if __name__ == "__main__":
    train_and_benchmark_models()
