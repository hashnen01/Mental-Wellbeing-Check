import numpy as np
import pandas as pd


def add_engineered_features(df):
    """Add the five composite wellbeing dimension scores to the dataframe."""
    df = df.copy()
    df["sleep_quality_score"] = calc_sleep_quality(df)
    df["physical_activity_score"] = calc_physical_activity(df)
    df["nutrition_score"] = calc_nutrition(df)
    df["social_connection_score"] = calc_social_connection(df)
    df["daily_balance_score"] = calc_daily_balance(df)
    return df


def calc_sleep_quality(df):
    """
    Combines sleep_hours, sleep_consistency, night_awakenings.
    Optimal sleep is 7-9 hours, consistent schedule, few awakenings.
    Returns a score from 0 to 100.
    """
    hours = df["sleep_hours"].fillna(7.0)
    consistency = df["sleep_consistency"].fillna("No")
    awakenings = df["night_awakenings"].fillna(1.0)

    # Hours component (0-40): best at 7-9h, decent at 6-10h
    hours_pts = np.where(
        (hours >= 7) & (hours <= 9), 40,
        np.where((hours >= 6) & (hours <= 10), 25, 10)
    )

    # Consistency bonus (0-30)
    consistency_pts = np.where(consistency == "Yes", 30, 10)

    # Awakenings penalty (0-30): lose 6 points per awakening
    awakenings_pts = np.clip(30 - awakenings * 6, 0, 30)

    return np.round(hours_pts + consistency_pts + awakenings_pts, 1)


def calc_physical_activity(df):
    """
    Combines physical_activity_mins, outdoor_time_mins, sedentary_hours.
    More activity and outdoor time is better, less sitting is better.
    Returns a score from 0 to 100.
    """
    activity = df["physical_activity_mins"].fillna(30.0)
    outdoor = df["outdoor_time_mins"].fillna(30.0)
    sedentary = df["sedentary_hours"].fillna(8.0)

    # Activity (0-40): 60+ mins daily is excellent
    activity_pts = np.clip(activity / 60 * 40, 0, 40)

    # Outdoor time (0-30): 60+ mins is excellent
    outdoor_pts = np.clip(outdoor / 60 * 30, 0, 30)

    # Sedentary penalty (0-30): sitting beyond 4 hours hurts
    sedentary_pts = np.clip(30 - (sedentary - 4) * 4, 0, 30)

    return np.round(activity_pts + outdoor_pts + sedentary_pts, 1)


def calc_nutrition(df):
    """
    Combines water_intake_L, meals_skipped, junk_food_days_per_week.
    More water is better, fewer skipped meals and junk food days is better.
    Returns a score from 0 to 100.
    """
    water = df["water_intake_L"].fillna(2.0)
    meals_skipped = df["meals_skipped"].fillna(0.0)
    junk_food = df["junk_food_days_per_week"].fillna(2.0)

    # Water intake (0-40): 2.5L+ is excellent
    water_pts = np.clip(water / 2.5 * 40, 0, 40)

    # Meals penalty (0-30): each skipped meal costs 10 points
    meals_pts = np.clip(30 - meals_skipped * 10, 0, 30)

    # Junk food penalty (0-30): each day costs ~4.3 points
    junk_pts = np.clip(30 - junk_food * 4.3, 0, 30)

    return np.round(water_pts + meals_pts + junk_pts, 1)


def calc_social_connection(df):
    """
    Combines meaningful_conversations_per_day, social_media_hours, social_isolation_score.
    More conversations is better, less social media and isolation is better.
    Returns a score from 0 to 100.
    """
    conversations = df["meaningful_conversations_per_day"].fillna(2.0)
    social_media = df["social_media_hours"].fillna(3.0)
    isolation = df["social_isolation_score"].fillna(3.0)

    # Conversations (0-40): 4+ per day is excellent
    convo_pts = np.clip(conversations / 4 * 40, 0, 40)

    # Social media penalty (0-30): more than 1 hour starts hurting
    social_pts = np.clip(30 - (social_media - 1) * 5, 0, 30)

    # Isolation penalty (0-30): scale 1-5, higher = more isolated
    isolation_pts = np.clip(30 - (isolation - 1) * 7.5, 0, 30)

    return np.round(convo_pts + social_pts + isolation_pts, 1)


def calc_daily_balance(df):
    """
    Combines work_study_hours, hobby_time_mins, stressful_events_today, days_since_last_vacation.
    Balanced work, more hobbies, less stress, recent vacation = better.
    Returns a score from 0 to 100.
    """
    work = df["work_study_hours"].fillna(8.0)
    hobby = df["hobby_time_mins"].fillna(30.0)
    stress = df["stressful_events_today"].fillna(1.0)
    vacation = df["days_since_last_vacation"].fillna(60.0)

    # Work balance (0-30): 6-8 hours is optimal
    work_pts = np.where(
        (work >= 6) & (work <= 8), 30,
        np.where((work >= 4) & (work <= 10), 20, 10)
    )

    # Hobby time (0-25): 60+ mins is excellent
    hobby_pts = np.clip(hobby / 60 * 25, 0, 25)

    # Stress penalty (0-25): each event costs 5 points
    stress_pts = np.clip(25 - stress * 5, 0, 25)

    # Vacation recency (0-20): within 30 days = great
    vacation_pts = np.where(
        vacation <= 30, 20,
        np.where(vacation <= 90, 12, 5)
    )

    return np.round(work_pts + hobby_pts + stress_pts + vacation_pts, 1)
