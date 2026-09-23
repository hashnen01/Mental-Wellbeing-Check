"""
predict.py — Full prediction pipeline.

Flow:
  1. User form input (dict)
  2. Wrap in a single-row DataFrame
  3. Add engineered features (composite wellbeing dimension scores)
  4. Encode categorical columns to integers
  5. Select features in the exact order the model expects
  6. Scale with the fitted StandardScaler
  7. Predict with the trained Linear Regression model
  8. Compute SHAP values to explain which features drove the score
  9. Build human-readable explanations and recommendations
 10. Return score, risk level, dimensions, explanations, recommendations
"""
import numpy as np
import pandas as pd

import config
from model import get_model, get_scaler, get_feature_names
from feature_engineering import add_engineered_features
from utils import get_risk_level, format_score

# --- Human-readable explanations for each feature ---
FEATURE_EXPLANATIONS = {
    "sleep_hours": {
        "emoji": "😴", "negative": "Low sleep ({value:.0f} hours)",
        "positive": "Healthy sleep ({value:.0f} hours)",
        "tip": "Try to get 7-8 hours of sleep tonight.",
    },
    "sleep_consistency": {
        "emoji": "😴", "negative": "Inconsistent sleep schedule",
        "positive": "Consistent sleep schedule",
        "tip": "Go to bed and wake up at the same time every day.",
    },
    "night_awakenings": {
        "emoji": "😴", "negative": "Waking up often at night ({value:.0f} times)",
        "positive": "Few night awakenings",
        "tip": "Avoid screens and caffeine before bedtime.",
    },
    "physical_activity_mins": {
        "emoji": "🏃", "negative": "Very little physical activity ({value:.0f} mins)",
        "positive": "Good physical activity ({value:.0f} mins)",
        "tip": "Walk for at least 20 minutes today.",
    },
    "outdoor_time_mins": {
        "emoji": "🌳", "negative": "Very little time outdoors ({value:.0f} mins)",
        "positive": "Good outdoor time ({value:.0f} mins)",
        "tip": "Spend at least 30 minutes outdoors today.",
    },
    "sedentary_hours": {
        "emoji": "🪑", "negative": "Too much sitting ({value:.0f} hours)",
        "positive": "Healthy amount of movement",
        "tip": "Stand up and stretch every hour.",
    },
    "water_intake_L": {
        "emoji": "💧", "negative": "Low water intake ({value:.1f}L)",
        "positive": "Good hydration ({value:.1f}L)",
        "tip": "Drink at least 2 litres of water today.",
    },
    "meals_skipped": {
        "emoji": "🍽️", "negative": "Skipped {value:.0f} meal(s) today",
        "positive": "Regular meals throughout the day",
        "tip": "Try not to skip meals. Eat at regular times.",
    },
    "junk_food_days_per_week": {
        "emoji": "🍔", "negative": "Junk food {value:.0f} days per week",
        "positive": "Low junk food consumption",
        "tip": "Reduce junk food to 2 days a week or less.",
    },
    "meaningful_conversations_per_day": {
        "emoji": "👥", "negative": "Very few meaningful conversations",
        "positive": "Good social interactions",
        "tip": "Talk to a family member or friend today.",
    },
    "social_media_hours": {
        "emoji": "📱", "negative": "High screen time ({value:.1f} hours)",
        "positive": "Low screen time",
        "tip": "Reduce screen time by one hour.",
    },
    "social_isolation_score": {
        "emoji": "🏠", "negative": "Feeling socially isolated",
        "positive": "Good social connection",
        "tip": "Reach out to someone you trust today.",
    },
    "work_study_hours": {
        "emoji": "💼", "negative": "Work-life balance needs attention ({value:.0f} hours)",
        "positive": "Balanced work/study schedule",
        "tip": "Take regular breaks during work or study.",
    },
    "stressful_events_today": {
        "emoji": "😰", "negative": "Multiple stressful events ({value:.0f} today)",
        "positive": "Low stress today",
        "tip": "Try 5 minutes of deep breathing or a short walk.",
    },
    "mood_stability": {
        "emoji": "🧘", "negative": "Unstable mood patterns",
        "positive": "Stable mood",
        "tip": "Practice mindfulness or journaling for a few minutes.",
    },
    "concentration_difficulty": {
        "emoji": "🧠", "negative": "Difficulty concentrating",
        "positive": "Good concentration",
        "tip": "Try working in focused 25-minute sessions.",
    },
    "purpose_in_life": {
        "emoji": "🎯", "negative": "Low sense of purpose",
        "positive": "Strong sense of purpose",
        "tip": "Set one small meaningful goal for today.",
    },
    "hobby_time_mins": {
        "emoji": "🎨", "negative": "Very little hobby time ({value:.0f} mins)",
        "positive": "Good hobby time ({value:.0f} mins)",
        "tip": "Spend at least 30 minutes on something you enjoy.",
    },
    "days_since_last_vacation": {
        "emoji": "✈️", "negative": "Long time since last break ({value:.0f} days)",
        "positive": "Had a recent break",
        "tip": "Plan a short break or day off soon.",
    },
    "bmi": {
        "emoji": "⚖️", "negative": "BMI outside healthy range ({value:.1f})",
        "positive": "Healthy BMI ({value:.1f})",
        "tip": "Focus on balanced nutrition and regular exercise.",
    },
    "age": {
        "emoji": "📅", "negative": "Age-related factor",
        "positive": "Age-related factor", "tip": None,
    },
    "gender": {
        "emoji": "👤", "negative": "Demographic factor",
        "positive": "Demographic factor", "tip": None,
    },
    "occupation": {
        "emoji": "💼", "negative": "Occupation-related factor",
        "positive": "Occupation-related factor", "tip": None,
    },
    "smoking": {
        "emoji": "🚬", "negative": "Smoking is affecting your wellbeing",
        "positive": "Non-smoking habits helping wellbeing",
        "tip": "Consider reducing or quitting smoking.",
    },
    "alcohol_consumption": {
        "emoji": "🍷", "negative": "Alcohol consumption affecting wellbeing",
        "positive": "Low alcohol consumption",
        "tip": "Try to limit alcohol intake.",
    },
    # Engineered features
    "sleep_quality_score": {
        "emoji": "😴", "negative": "Overall sleep quality is low",
        "positive": "Good overall sleep quality",
        "tip": "Improve your sleep schedule and reduce night awakenings.",
    },
    "physical_activity_score": {
        "emoji": "🏃", "negative": "Low physical activity overall",
        "positive": "Good activity level overall",
        "tip": "Increase daily movement and outdoor time.",
    },
    "nutrition_score": {
        "emoji": "🍎", "negative": "Nutrition habits need improvement",
        "positive": "Good nutrition habits",
        "tip": "Eat regular meals, drink more water, reduce junk food.",
    },
    "social_connection_score": {
        "emoji": "👥", "negative": "Social connections need attention",
        "positive": "Strong social connections",
        "tip": "Have more meaningful conversations and reduce screen time.",
    },
    "daily_balance_score": {
        "emoji": "🧠", "negative": "Work-life balance needs attention",
        "positive": "Good daily balance",
        "tip": "Balance work with hobbies and take regular breaks.",
    },
}


def make_prediction(user_input):
    """
    Full prediction pipeline.

    Takes the validated form input (a plain dict) and returns a dict
    containing everything the result page template needs to render.

    Pipeline stages:
      1. Wrap in DataFrame       — one row per submission
      2. Feature engineering     — add five composite dimension scores
      3. Save dimension scores   — captured before encoding overwrites strings
      4. Encode categoricals     — convert text labels to integers
      5. Select & order features — match the exact column order the model expects
      6. Scale                   — StandardScaler: mean 0, std 1
      7. Predict                 — Linear Regression → raw score (0–100)
      8. SHAP + output           — explain the score and build recommendations
    """
    # Stage 1 — wrap the single user submission in a DataFrame row
    input_df = pd.DataFrame([user_input])

    # Stage 2 — add five composite scores (Sleep, Activity, Nutrition, Social, Balance).
    # This must happen before encoding because the formulas need the raw string values
    # (e.g. sleep_consistency == "Yes") not the encoded integers.
    input_df = add_engineered_features(input_df)

    # Stage 3 — read the dimension scores now, before encoding replaces the values
    dimensions = {
        "sleep":    float(input_df["sleep_quality_score"].iloc[0]),
        "activity": float(input_df["physical_activity_score"].iloc[0]),
        "nutrition": float(input_df["nutrition_score"].iloc[0]),
        "social":   float(input_df["social_connection_score"].iloc[0]),
        "balance":  float(input_df["daily_balance_score"].iloc[0]),
    }

    # Stage 4 — convert categorical text columns to integers using the fixed
    # mappings defined in config.py (e.g. "Male" → 0, "Never" → 0)
    for col, mapping in config.ENCODING_MAPS.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].map(mapping).fillna(0).astype(int)

    # Stage 5 — select only the features the model was trained on, in the
    # exact same column order that was saved during training
    feature_names = get_feature_names()
    input_features = input_df[feature_names]

    # Stage 6 — apply the same StandardScaler that was fitted on the training set
    scaler = get_scaler()
    input_scaled = pd.DataFrame(
        scaler.transform(input_features), columns=feature_names
    )

    # Stage 7 — run the trained Linear Regression model and clamp the output to 0–100
    model = get_model()
    raw_score = float(model.predict(input_scaled)[0])
    score = format_score(raw_score)

    # Stage 8 — compute SHAP values, convert to human-readable explanations,
    # generate recommendations, and determine the risk/wellbeing level
    shap_impacts = _get_shap_impacts(model, input_scaled, feature_names)
    explanations = _build_explanations(shap_impacts, user_input)
    recommendations = _build_recommendations(shap_impacts)
    risk = get_risk_level(score)

    # Build the dimension detail list (used by the radar/bar chart on the result page)
    dimension_details = [
        {"key": "sleep",    "score": dimensions["sleep"],    "emoji": "😴", "label": "Sleep"},
        {"key": "activity", "score": dimensions["activity"], "emoji": "🏃", "label": "Physical Activity"},
        {"key": "nutrition","score": dimensions["nutrition"],"emoji": "🍎", "label": "Nutrition"},
        {"key": "social",   "score": dimensions["social"],   "emoji": "👥", "label": "Social Connection"},
        {"key": "balance",  "score": dimensions["balance"],  "emoji": "🧠", "label": "Daily Balance"},
    ]

    return {
        "score": score,
        "risk_level": risk,
        "dimensions": dimensions,
        "dimension_details": dimension_details,
        "explanations": explanations,
        "recommendations": recommendations,
    }


def _get_shap_impacts(model, input_scaled, feature_names):
    """
    Compute SHAP values for a Linear Regression model and return feature
    impacts sorted from most to least influential (by absolute SHAP value).

    Why LinearExplainer?
    The saved model is confirmed to be a LinearRegression. SHAP's
    LinearExplainer is purpose-built for linear models — it computes each
    feature's contribution as: SHAP_i = coefficient_i × (x_i − E[x_i]).
    """
    try:
        import shap
        # Load the 100-row background sample saved during training.
        # SHAP uses it to estimate the average model output (the baseline).
        background = _load_shap_background()
        explainer = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(input_scaled)
        values = shap_values[0]  # single prediction → first (only) row
    except Exception:
        # Fallback if SHAP is not installed or fails.
        # For a linear model: impact ≈ coefficient × scaled feature value.
        # This is mathematically the same as a SHAP value for linear models.
        values = model.coef_ * input_scaled.values[0]

    impacts = []
    for i, fname in enumerate(feature_names):
        sv = float(values[i])
        impacts.append({
            "feature": fname,
            "shap_value": sv,
            "abs_value": abs(sv),
        })

    # Sort so the most influential features come first
    impacts.sort(key=lambda x: x["abs_value"], reverse=True)
    return impacts


def _load_shap_background():
    """Load the saved background sample for SHAP explainability."""
    import pickle
    with open(config.SHAP_BACKGROUND_PATH, "rb") as f:
        return pickle.load(f)


def _build_explanations(shap_impacts, raw_input):
    """Turn top SHAP impacts into separated positive and negative explanations."""
    positives = []
    negatives = []

    for impact in shap_impacts:
        fname = impact["feature"]
        shap_val = impact["shap_value"]

        info = FEATURE_EXPLANATIONS.get(fname)
        if not info:
            continue

        direction = "negative" if shap_val < 0 else "positive"
        template = info[direction]

        # Try to fill in the actual value
        raw_value = raw_input.get(fname, 0)
        try:
            text = template.format(value=float(raw_value))
        except (ValueError, TypeError, KeyError):
            text = template

        item = {
            "emoji": info["emoji"],
            "text": text,
            "direction": direction,
        }

        if direction == "positive" and len(positives) < 3:
            positives.append(item)
        elif direction == "negative" and len(negatives) < 4:
            negatives.append(item)

        if len(positives) >= 3 and len(negatives) >= 4:
            break

    return {
        "positives": positives,
        "negatives": negatives,
    }


def _build_recommendations(shap_impacts):
    """Generate prioritized actionable tips (Today's Top 3 + Additional)."""
    tips = []
    seen = set()

    for impact in shap_impacts:
        if impact["shap_value"] >= 0:
            continue

        info = FEATURE_EXPLANATIONS.get(impact["feature"])
        if not info or not info.get("tip"):
            continue

        tip = info["tip"]
        if tip not in seen:
            tips.append(tip)
            seen.add(tip)

        if len(tips) >= 6:
            break

    return {
        "top_3": tips[:3],
        "additional": tips[3:],
    }
