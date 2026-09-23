"""
Model training script.
Trains Linear Regression, Random Forest, and XGBoost.
Compares all three and saves the best performing model.
Usage: python train_model.py
"""

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config
from feature_engineering import add_engineered_features


def load_and_clean_data():
    """Load the dataset and handle missing values."""
    df = pd.read_csv(config.DATA_PATH)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"\nMissing values found:\n{missing}\n")

    # Fill numerical columns with median
    for col in config.NUMERICAL_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical columns with mode
    for col in config.CATEGORICAL_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Drop rows where the target is missing
    df = df.dropna(subset=[config.TARGET_COLUMN])

    print(f"After cleaning: {len(df)} rows")
    return df


def encode_categoricals(df):
    """Convert categorical columns to numbers using fixed mappings."""
    df = df.copy()
    for col, mapping in config.ENCODING_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)
    return df


def prepare_features(df):
    """Build the feature matrix and target vector."""
    # Step 1: engineer composite features (needs raw string values)
    df = add_engineered_features(df)

    # Step 2: encode categorical columns to integers
    df = encode_categoricals(df)

    # Combine all feature columns
    feature_cols = (
        config.NUMERICAL_COLUMNS
        + config.CATEGORICAL_COLUMNS
        + config.ENGINEERED_FEATURES
    )
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df[config.TARGET_COLUMN]

    return X, y, feature_cols


def train_and_compare(X_train, X_test, y_train, y_test):
    """Train three models, evaluate each, and return results."""
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=config.RANDOM_STATE,
            verbosity=0,
        ),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results[name] = {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
        }
        trained_models[name] = model
        print(f"  MAE: {mae:.4f}  |  RMSE: {rmse:.4f}  |  R²: {r2:.4f}")

    return results, trained_models


def save_best_model(results, trained_models, scaler, feature_names):
    """Pick the model with the highest R² and save everything."""
    best_name = max(results, key=lambda k: results[k]["R2"])
    best_model = trained_models[best_name]
    print(f"\n>>> Best model: {best_name} (R² = {results[best_name]['R2']})")

    with open(config.MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    with open(config.SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    with open(config.FEATURE_LIST_PATH, "wb") as f:
        pickle.dump(feature_names, f)

    report = {
        "best_model": best_name,
        "results": results,
        "feature_count": len(feature_names),
        "features": feature_names,
    }
    with open(config.TRAINING_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved model   -> {config.MODEL_PATH}")
    print(f"Saved scaler  -> {config.SCALER_PATH}")
    print(f"Saved report  -> {config.TRAINING_REPORT_PATH}")


def main():
    print("=" * 50)
    print("  Mental Wellbeing Check — Model Training")
    print("=" * 50)

    df = load_and_clean_data()
    X, y, feature_names = prepare_features(df)
    print(f"\nFeature count: {len(feature_names)}")
    print(f"Sample count:  {len(X)}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )

    # Save a small background sample for SHAP explainability
    background = X_train_scaled.sample(100, random_state=config.RANDOM_STATE)
    with open(config.SHAP_BACKGROUND_PATH, "wb") as f:
        pickle.dump(background, f)
    print(f"Saved SHAP background sample (100 rows)")

    # Train and compare models
    print("\n" + "-" * 50)
    results, trained_models = train_and_compare(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    # Save the best one
    print("\n" + "-" * 50)
    save_best_model(results, trained_models, scaler, feature_names)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
