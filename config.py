import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- File Paths ---
DATA_PATH = os.path.join(BASE_DIR, "data", "mental_wellbeing_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "wellbeing_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
FEATURE_LIST_PATH = os.path.join(BASE_DIR, "model", "feature_names.pkl")
TRAINING_REPORT_PATH = os.path.join(BASE_DIR, "model", "training_report.json")
SHAP_BACKGROUND_PATH = os.path.join(BASE_DIR, "model", "shap_background.pkl")
DATABASE_PATH = os.path.join(BASE_DIR, "wellbeing.db")

# --- Dataset Columns ---
TARGET_COLUMN = "mental_wellbeing_score"

CATEGORICAL_COLUMNS = [
    "gender",
    "occupation",
    "sleep_consistency",
    "smoking",
    "alcohol_consumption",
]

NUMERICAL_COLUMNS = [
    "age",
    "sleep_hours",
    "night_awakenings",
    "physical_activity_mins",
    "outdoor_time_mins",
    "sedentary_hours",
    "water_intake_L",
    "meals_skipped",
    "junk_food_days_per_week",
    "meaningful_conversations_per_day",
    "social_media_hours",
    "social_isolation_score",
    "work_study_hours",
    "stressful_events_today",
    "mood_stability",
    "concentration_difficulty",
    "purpose_in_life",
    "hobby_time_mins",
    "days_since_last_vacation",
    "bmi",
]

# --- Engineered Feature Names ---
ENGINEERED_FEATURES = [
    "sleep_quality_score",
    "physical_activity_score",
    "nutrition_score",
    "social_connection_score",
    "daily_balance_score",
]

# --- Risk Level Thresholds ---
RISK_LEVELS = [
    {
        "min": 90, "max": 100, "label": "Excellent", "color": "success", "emoji": "🟢",
        "interpretation": "Your recent habits indicate an outstandingly healthy lifestyle. Maintaining your consistent sleep, regular activity, and strong social connections will help preserve your high wellbeing."
    },
    {
        "min": 75, "max": 89, "label": "Good", "color": "success", "emoji": "🟢",
        "interpretation": "Your recent habits indicate a generally healthy lifestyle. Maintaining consistent sleep, regular activity, and social interaction is contributing positively to your wellbeing."
    },
    {
        "min": 60, "max": 74, "label": "Fair", "color": "warning", "emoji": "🟡",
        "interpretation": "Your wellbeing is stable overall, but there are a few lifestyle areas that could be improved to help you feel better over time."
    },
    {
        "min": 40, "max": 59, "label": "Needs Attention", "color": "orange", "emoji": "🟠",
        "interpretation": "Several recent lifestyle habits may be reducing your overall wellbeing. Small, consistent improvements can help increase your score over time."
    },
    {
        "min": 0, "max": 39, "label": "Poor Wellbeing", "color": "danger", "emoji": "🔴",
        "interpretation": "Your daily lifestyle habits suggest your wellbeing currently needs extra care and attention. Focusing on small daily steps like restful sleep and light movement can make a meaningful difference."
    },
]

# --- Categorical Encoding Maps ---
ENCODING_MAPS = {
    "gender": {"Male": 0, "Female": 1, "Non-binary": 2},
    "occupation": {
        "Student": 0, "Employed_Full_Time": 1, "Employed_Part_Time": 2,
        "Self_Employed": 3, "Unemployed": 4, "Retired": 5, "Homemaker": 6,
    },
    "sleep_consistency": {"No": 0, "Yes": 1},
    "smoking": {"Never": 0, "Former": 1, "Occasional": 2, "Daily": 3},
    "alcohol_consumption": {"Never": 0, "Rarely": 1, "Moderate": 2, "Heavy": 3},
}

# --- Model Training Settings ---
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Flask Settings ---
# Set SECRET_KEY in your environment for production.
SECRET_KEY = os.environ.get("SECRET_KEY", "wellbeing-check-local-key")
DEBUG = os.environ.get("DEBUG", "true").lower() == "true"
