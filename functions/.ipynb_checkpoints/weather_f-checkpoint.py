"""
weather_f.py
Data handling, statistics, and analysis functions for the Weather Tracker app.
"""

import os
from datetime import datetime, timedelta

import pandas as pd

CSV_FILE = "data/weather_data.csv"
COLUMNS = ["Date", "Temperature_C", "Condition", "Humidity_%", "Wind_Speed_kmh"]
DATE_FORMAT = "%d-%m-%Y"

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def load_data(csv_path=CSV_FILE):
    """Load weather observations from CSV, creating an empty file if none exists."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, dtype={"Date": str})
        # Fix any old-format dates (YYYY-MM-DD) → DD-MM-YYYY
        df["Date"] = df["Date"].apply(_normalise_date)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(csv_path, index=False)
    return df


def _normalise_date(date_str):
    """Accept DD-MM-YYYY or YYYY-MM-DD and always return DD-MM-YYYY."""
    if not isinstance(date_str, str):
        return date_str
    # Already DD-MM-YYYY
    try:
        datetime.strptime(date_str, "%d-%m-%Y")
        return date_str
    except ValueError:
        pass
    # Old format YYYY-MM-DD
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%d-%m-%Y")
    except ValueError:
        pass
    return date_str  # leave as-is if unparseable


def save_observation(date_str, temperature, condition, humidity, wind_speed,
                     csv_path=CSV_FILE):
    """Append a new observation to the CSV file and return the updated DataFrame."""
    df = load_data(csv_path)
    new_row = {
        "Date": date_str,
        "Temperature_C": temperature,
        "Condition": condition,
        "Humidity_%": humidity,
        "Wind_Speed_kmh": wind_speed,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return df


def validate_date(date_str):
    """Return a datetime object if date_str matches DD-MM-YYYY, else None."""
    try:
        return datetime.strptime(date_str, DATE_FORMAT)
    except (ValueError, TypeError):
        return None


def _with_datetime(df):
    """Return a copy of df with a parsed Date_dt column; bad rows dropped."""
    df = df.copy()
    df["Date_dt"] = pd.to_datetime(df["Date"], format=DATE_FORMAT, errors="coerce")
    return df.dropna(subset=["Date_dt"])


#  Core features 

def get_statistics(df):
    if df.empty:
        return None
    temps = pd.to_numeric(df["Temperature_C"], errors="coerce").dropna()
    if temps.empty:
        return None
    mode = df["Condition"].mode()
    return {
        "avg_temp": round(temps.mean(), 2),
        "min_temp": temps.min(),
        "max_temp": temps.max(),
        "most_common_condition": mode.iloc[0] if not mode.empty else None,
    }


def search_by_date(df, date_str):
    return df[df["Date"] == date_str]


#  Month / season filtering 

def get_season(month):
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def filter_by_month(df, month):
    dfd = _with_datetime(df)
    return dfd[dfd["Date_dt"].dt.month == month].drop(columns="Date_dt")


def filter_by_season(df, season):
    dfd = _with_datetime(df)
    dfd["Season"] = dfd["Date_dt"].dt.month.apply(get_season)
    return dfd[dfd["Season"] == season].drop(columns=["Date_dt", "Season"])


def get_available_years(df):
    dfd = _with_datetime(df)
    if dfd.empty:
        return []
    return sorted(dfd["Date_dt"].dt.year.unique().tolist())


#  Text-based trend graph 

def generate_text_trend(df, n=10):
    dfd = _with_datetime(df).sort_values("Date_dt")
    if dfd.empty:
        return "No data available."
    dfd = dfd.tail(n)
    temps = pd.to_numeric(dfd["Temperature_C"], errors="coerce")
    min_temp, max_temp = temps.min(), temps.max()
    span = max(max_temp - min_temp, 1)
    lines = []
    for date_str, temp in zip(dfd["Date"], temps):
        bar_len = int(((temp - min_temp) / span) * 30) + 1
        lines.append(f"{date_str} | {'█' * bar_len} {temp}°C")
    return "\n".join(lines)


#  Tomorrow's weather prediction

def predict_tomorrow(df):
    dfd = _with_datetime(df)
    if dfd.empty:
        return None
    tomorrow = datetime.now() + timedelta(days=1)
    same_month = dfd[dfd["Date_dt"].dt.month == tomorrow.month]
    source = same_month if not same_month.empty else dfd
    temps = pd.to_numeric(source["Temperature_C"], errors="coerce").dropna()
    mode = source["Condition"].mode()
    return {
        "date": tomorrow.strftime(DATE_FORMAT),
        "predicted_temp": round(temps.mean(), 1) if not temps.empty else None,
        "predicted_condition": mode.iloc[0] if not mode.empty else None,
        "based_on": "same-month history" if not same_month.empty else "overall history",
        "sample_size": len(source),
    }


#  Year-over-year comparison 

def compare_years(df, year1, year2):
    dfd = _with_datetime(df)
    dfd["Temperature_C"] = pd.to_numeric(dfd["Temperature_C"], errors="coerce")
    dfd["Year"] = dfd["Date_dt"].dt.year
    dfd["Month"] = dfd["Date_dt"].dt.month
    y1 = dfd[dfd["Year"] == year1].groupby("Month")["Temperature_C"].mean()
    y2 = dfd[dfd["Year"] == year2].groupby("Month")["Temperature_C"].mean()
    comparison = pd.DataFrame({str(year1): y1, str(year2): y2})
    comparison.index = [MONTH_NAMES[m - 1] for m in comparison.index]
    comparison.index.name = "Month"
    comparison["Difference"] = comparison[str(year1)] - comparison[str(year2)]
    return comparison.round(2)


#  Record-breaking weather 

def get_records(df):
    if df.empty:
        return None
    dfd = df.copy()
    dfd["Temperature_C"] = pd.to_numeric(dfd["Temperature_C"], errors="coerce")
    dfd["Humidity_%"] = pd.to_numeric(dfd["Humidity_%"], errors="coerce")
    dfd["Wind_Speed_kmh"] = pd.to_numeric(dfd["Wind_Speed_kmh"], errors="coerce")
    dfd = dfd.dropna(subset=["Temperature_C", "Humidity_%", "Wind_Speed_kmh"])
    if dfd.empty:
        return None
    hottest = dfd.loc[dfd["Temperature_C"].idxmax()]
    coldest = dfd.loc[dfd["Temperature_C"].idxmin()]
    most_humid = dfd.loc[dfd["Humidity_%"].idxmax()]
    windiest = dfd.loc[dfd["Wind_Speed_kmh"].idxmax()]
    return {
        "hottest": (hottest["Date"], hottest["Temperature_C"]),
        "coldest": (coldest["Date"], coldest["Temperature_C"]),
        "most_humid": (most_humid["Date"], most_humid["Humidity_%"]),
        "windiest": (windiest["Date"], windiest["Wind_Speed_kmh"]),
    }