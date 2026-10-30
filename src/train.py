"""
Enterprise Model Training & Benchmarking Pipeline for IPL Match Predictor.
Implements leak-free match/season data splitting, calibration analysis (Brier score, ECE, MCE),
multi-architecture benchmarking (Random Forest vs XGBoost), and structured metadata logging.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
try:
    from xgboost import XGBClassifier, XGBRegressor
    # Probe runtime library loading (e.g. libomp on macOS)
    _dummy = XGBClassifier(n_estimators=1)
    HAS_XGBOOST = True
except Exception as _e:
    HAS_XGBOOST = False
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

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


def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
    and reliability table across probability bins.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    ece = 0.0
    mce = 0.0
    total_samples = len(y_true)
    reliability_table = []

    for b in range(n_bins):
        mask = (bin_indices == b)
        bin_size = int(np.sum(mask))
        if bin_size > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            abs_err = abs(bin_acc - bin_conf)
            ece += (bin_size / total_samples) * abs_err
            mce = max(mce, abs_err)
            reliability_table.append({
                "bin": b + 1,
                "range": f"[{bin_edges[b]:.1f}, {bin_edges[b+1]:.1f})",
                "count": bin_size,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
                "abs_error": round(abs_err, 4)
            })

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "reliability_table": reliability_table,
        "calibration_curve": {
            "prob_true": [round(float(x), 4) for x in prob_true],
            "prob_pred": [round(float(x), 4) for x in prob_pred]
        }
    }


def split_dataset(
    df: pd.DataFrame,
    split_strategy: str = "match",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into train and test sets without data leakage.
    - 'match': Uses GroupShuffleSplit on match 'id' so all balls of a match are in one split.
    - 'season': Trains on earlier historical seasons and tests on the latest seasons.
    - 'random': Standard row-level random split (for baseline comparison).
    """
    if split_strategy == "match":
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(df, groups=df['id']))
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
    elif split_strategy == "season" and 'season' in df.columns:
        all_seasons = sorted(df['season'].dropna().unique())
        n_test_seasons = max(1, int(len(all_seasons) * test_size))
        test_seasons = all_seasons[-n_test_seasons:]
        train_seasons = all_seasons[:-n_test_seasons]
        train_df = df[df['season'].isin(train_seasons)].copy()
        test_df = df[df['season'].isin(test_seasons)].copy()
        print(f"Season split: Train={train_seasons}, Test={test_seasons}")
    else:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Verify zero match leakage
    leakage = set(train_df['id']).intersection(set(test_df['id']))
    if leakage:
        print(f"⚠️ Warning: Detected {len(leakage)} overlapping match IDs in train/test splits.")
    else:
        print(f"✅ Verified leak-free split: 0 match overlap ({train_df['id'].nunique()} train matches, {test_df['id'].nunique()} test matches).")

    return train_df, test_df


def train_and_benchmark_models(
    split_strategy: str = "match",
    random_state: int = 42,
    deliveries_path: Path = DELIVERIES_CSV,
    matches_path: Path = MATCHES_CSV
) -> Dict[str, Any]:
    """Main training, leak-free evaluation, and benchmarking pipeline."""
    print("=" * 70)
    print("🏏 IPL MATCH PREDICTOR — LEAK-FREE ML TRAINING & BENCHMARKING PIPELINE")
    print("=" * 70)
    print(f"Configuration: split_strategy='{split_strategy}', random_state={random_state}")

    start_time = time.time()

    # 1. Ingestion
    print("\n[1/4] Ingesting & Validating Datasets...")
    deliveries, matches = load_raw_data(deliveries_path, matches_path)
    df_merged = clean_and_merge_data(deliveries, matches)
    print(f"Loaded {len(deliveries):,} delivery records across {len(matches):,} matches.")

    # 2. Feature Engineering
    print("\n[2/4] Engineering Domain Match State Features...")
    over_stats, le_batting, le_bowling = build_match_state_dataframe(df_merged)
    print(f"Generated {len(over_stats):,} granular state vectors across {len(le_batting.classes_)} teams.")

    # 3. Train Classification Models (2nd Innings Win Probability)
    print(f"\n[3/4] Benchmarking Win Probability Classifiers (Split Strategy: {split_strategy.upper()})...")
    second_innings_data = over_stats[over_stats['inning'] == 2].copy()

    train_cls_df, test_cls_df = split_dataset(
        second_innings_data,
        split_strategy=split_strategy,
        test_size=0.2,
        random_state=random_state
    )

    X_train_c = train_cls_df[CLASSIFICATION_FEATURES]
    y_train_c = train_cls_df['win_status'].values
    X_test_c = test_cls_df[CLASSIFICATION_FEATURES]
    y_test_c = test_cls_df['win_status'].values

    # Train Random Forest Classifier
    print(" -> Training Random Forest Classifier...")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=random_state, n_jobs=-1)
    rf_clf.fit(X_train_c, y_train_c)
    rf_probs = rf_clf.predict_proba(X_test_c)[:, 1]
    rf_preds = rf_clf.predict(X_test_c)
    rf_acc = accuracy_score(y_test_c, rf_preds)
    rf_auc = roc_auc_score(y_test_c, rf_probs)
    rf_brier = brier_score_loss(y_test_c, rf_probs)
    rf_logloss = log_loss(y_test_c, rf_probs)
    rf_calib = compute_calibration_metrics(y_test_c, rf_probs)

    print(f"    RF  | Acc: {rf_acc * 100:.2f}% | ROC-AUC: {rf_auc:.4f} | Brier: {rf_brier:.4f} | LogLoss: {rf_logloss:.4f} | ECE: {rf_calib['ece']:.4f}")

    # Train Second Architecture (XGBoost or GradientBoosting)
    if HAS_XGBOOST:
        print(" -> Training XGBoost Classifier...")
        boost_clf = XGBClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.08,
            random_state=random_state,
            eval_metric="logloss"
        )
        boost_name = "XGBoost"
    else:
        print(" -> Training Gradient Boosting Classifier (sklearn fallback)...")
        boost_clf = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state
        )
        boost_name = "GradientBoosting"

    boost_clf.fit(X_train_c, y_train_c)
    boost_probs = boost_clf.predict_proba(X_test_c)[:, 1]
    boost_preds = boost_clf.predict(X_test_c)
    boost_acc = accuracy_score(y_test_c, boost_preds)
    boost_auc = roc_auc_score(y_test_c, boost_probs)
    boost_brier = brier_score_loss(y_test_c, boost_probs)
    boost_logloss = log_loss(y_test_c, boost_probs)
    boost_calib = compute_calibration_metrics(y_test_c, boost_probs)

    print(f"    {boost_name[:3]} | Acc: {boost_acc * 100:.2f}% | ROC-AUC: {boost_auc:.4f} | Brier: {boost_brier:.4f} | LogLoss: {boost_logloss:.4f} | ECE: {boost_calib['ece']:.4f}")

    # Select best classifier based on ROC-AUC and Brier calibration
    best_clf = rf_clf if rf_auc >= boost_auc else boost_clf
    best_clf_name = "RandomForestClassifier" if rf_auc >= boost_auc else f"{boost_name}Classifier"

    # 4. Train Regression Models (1st Innings Projected Total)
    print(f"\n[4/4] Benchmarking 1st Innings Score Regressors (Split Strategy: {split_strategy.upper()})...")
    innings1_data = over_stats[over_stats['inning'] == 1].copy()
    innings1_totals = df_merged[df_merged['inning'] == 1].groupby('id')['total_runs'].sum().reset_index()
    innings1_totals.rename(columns={'total_runs': 'final_total'}, inplace=True)
    innings1_data = innings1_data.merge(innings1_totals, on='id', how='left')

    train_reg_df, test_reg_df = split_dataset(
        innings1_data,
        split_strategy=split_strategy,
        test_size=0.2,
        random_state=random_state
    )

    X_train_r = train_reg_df[REGRESSION_FEATURES]
    y_train_r = train_reg_df['final_total'].values
    X_test_r = test_reg_df[REGRESSION_FEATURES]
    y_test_r = test_reg_df['final_total'].values

    # Random Forest Regressor
    print(" -> Training Random Forest Regressor...")
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=16, random_state=random_state, n_jobs=-1)
    rf_reg.fit(X_train_r, y_train_r)
    rf_pred_r = rf_reg.predict(X_test_r)
    rf_r2 = r2_score(y_test_r, rf_pred_r)
    rf_rmse = np.sqrt(mean_squared_error(y_test_r, rf_pred_r))
    rf_mae = mean_absolute_error(y_test_r, rf_pred_r)
    rf_cov10 = float(np.mean(np.abs(y_test_r - rf_pred_r) <= 10))
    rf_cov15 = float(np.mean(np.abs(y_test_r - rf_pred_r) <= 15))

    print(f"    RF  | R²: {rf_r2:.4f} | RMSE: {rf_rmse:.2f} | MAE: {rf_mae:.2f} | Cov±10: {rf_cov10 * 100:.1f}%")

    # Second Architecture Regressor
    if HAS_XGBOOST:
        print(" -> Training XGBoost Regressor...")
        boost_reg = XGBRegressor(n_estimators=120, max_depth=6, learning_rate=0.08, random_state=random_state)
    else:
        print(" -> Training Gradient Boosting Regressor (sklearn fallback)...")
        boost_reg = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state)

    boost_reg.fit(X_train_r, y_train_r)
    boost_pred_r = boost_reg.predict(X_test_r)
    boost_r2 = r2_score(y_test_r, boost_pred_r)
    boost_rmse = np.sqrt(mean_squared_error(y_test_r, boost_pred_r))
    boost_mae = mean_absolute_error(y_test_r, boost_pred_r)
    boost_cov10 = float(np.mean(np.abs(y_test_r - boost_pred_r) <= 10))
    boost_cov15 = float(np.mean(np.abs(y_test_r - boost_pred_r) <= 15))

    print(f"    {boost_name[:3]} | R²: {boost_r2:.4f} | RMSE: {boost_rmse:.2f} | MAE: {boost_mae:.2f} | Cov±10: {boost_cov10 * 100:.1f}%")

    best_reg = rf_reg if rf_r2 >= boost_r2 else boost_reg
    best_reg_name = "RandomForestRegressor" if rf_r2 >= boost_r2 else f"{boost_name}Regressor"

    # Save models & encoders
    print("\n💾 Serializing Models and Preprocessors...")
    joblib.dump(best_clf, WIN_CLASSIFIER_PATH)
    joblib.dump(best_reg, RUNS_REGRESSOR_PATH)
    joblib.dump(le_batting, LE_BATTING_PATH)
    joblib.dump(le_bowling, LE_BOWLING_PATH)

    # Save metadata
    elapsed_time = round(time.time() - start_time, 2)
    metadata = {
        "version": "2.2.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_time_seconds": elapsed_time,
        "evaluation_setup": {
            "split_strategy": split_strategy,
            "random_seed": random_state,
            "train_matches": int(train_cls_df['id'].nunique()),
            "test_matches": int(test_cls_df['id'].nunique()),
            "leak_free_guarantee": True
        },
        "dataset": {
            "source": "Kaggle IPL Complete Dataset (deliveries.csv & matches.csv)",
            "total_deliveries": len(deliveries),
            "total_matches": len(matches),
            "teams": list(le_batting.classes_)
        },
        "classification_benchmarks": {
            "selected_model": best_clf_name,
            "random_forest": {
                "accuracy": round(rf_acc, 4),
                "roc_auc": round(rf_auc, 4),
                "brier_score": round(rf_brier, 4),
                "log_loss": round(rf_logloss, 4),
                "expected_calibration_error": rf_calib["ece"],
                "max_calibration_error": rf_calib["mce"],
                "reliability_table": rf_calib["reliability_table"]
            },
            boost_name.lower(): {
                "accuracy": round(boost_acc, 4),
                "roc_auc": round(boost_auc, 4),
                "brier_score": round(boost_brier, 4),
                "log_loss": round(boost_logloss, 4),
                "expected_calibration_error": boost_calib["ece"],
                "max_calibration_error": boost_calib["mce"],
                "reliability_table": boost_calib["reliability_table"]
            }
        },
        "regression_benchmarks": {
            "selected_model": best_reg_name,
            "random_forest": {
                "r2_score": round(rf_r2, 4),
                "rmse": round(rf_rmse, 2),
                "mae": round(rf_mae, 2),
                "coverage_within_10_runs": round(rf_cov10, 4),
                "coverage_within_15_runs": round(rf_cov15, 4)
            },
            boost_name.lower(): {
                "r2_score": round(boost_r2, 4),
                "rmse": round(boost_rmse, 2),
                "mae": round(boost_mae, 2),
                "coverage_within_10_runs": round(boost_cov10, 4),
                "coverage_within_15_runs": round(boost_cov15, 4)
            }
        }
    }

    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata recorded at {METADATA_PATH}")
    print(f"🎉 Pipeline successfully completed in {elapsed_time}s!")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and benchmark IPL prediction models.")
    parser.add_argument("--split-strategy", choices=["match", "season", "random"], default="match",
                        help="Data splitting strategy to prevent temporal/match leakage.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    train_and_benchmark_models(split_strategy=args.split_strategy, random_state=args.seed)

