from config import RISK_LEVELS


def get_risk_level(score):
    """Return the risk level dict (label, color, emoji) for a given score."""
    for level in RISK_LEVELS:
        if level["min"] <= score <= level["max"]:
            return level
    return RISK_LEVELS[-1]


def format_score(score):
    """Clamp score to 0-100 range and round to 1 decimal place."""
    return round(max(0, min(100, score)), 1)


def validate_form_data(form_data):
    """
    Validate and clean the assessment form input.
    Returns (cleaned_data_dict, list_of_errors).
    """
    errors = []
    cleaned = {}

    numeric_fields = {
        "age": (10, 100, "Age"),
        "sleep_hours": (0, 24, "Sleep hours"),
        "night_awakenings": (0, 20, "Night awakenings"),
        "physical_activity_mins": (0, 500, "Physical activity"),
        "outdoor_time_mins": (0, 500, "Outdoor time"),
        "sedentary_hours": (0, 24, "Sedentary hours"),
        "water_intake_L": (0, 10, "Water intake"),
        "meals_skipped": (0, 3, "Meals skipped"),
        "junk_food_days_per_week": (0, 7, "Junk food days"),
        "meaningful_conversations_per_day": (0, 20, "Meaningful conversations"),
        "social_media_hours": (0, 24, "Social media hours"),
        "social_isolation_score": (1, 5, "Social isolation"),
        "work_study_hours": (0, 24, "Work/study hours"),
        "stressful_events_today": (0, 10, "Stressful events"),
        "mood_stability": (1, 5, "Mood stability"),
        "concentration_difficulty": (1, 5, "Concentration difficulty"),
        "purpose_in_life": (1, 5, "Purpose in life"),
        "hobby_time_mins": (0, 500, "Hobby time"),
        "days_since_last_vacation": (0, 1000, "Days since vacation"),
        "bmi": (10, 60, "BMI"),
    }

    for field, (min_val, max_val, label) in numeric_fields.items():
        value = form_data.get(field, "").strip()
        if not value:
            errors.append(f"{label} is required.")
            continue
        try:
            num = float(value)
            if num < min_val or num > max_val:
                errors.append(f"{label} must be between {min_val} and {max_val}.")
            else:
                cleaned[field] = num
        except ValueError:
            errors.append(f"{label} must be a valid number.")

    select_fields = {
        "gender": "Gender",
        "occupation": "Occupation",
        "sleep_consistency": "Sleep consistency",
        "smoking": "Smoking",
        "alcohol_consumption": "Alcohol consumption",
    }

    for field, label in select_fields.items():
        value = form_data.get(field, "").strip()
        if not value:
            errors.append(f"{label} is required.")
        else:
            cleaned[field] = value

    return cleaned, errors
