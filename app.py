"""
app.py
Streamlit UI for the Weather Tracker app.
"""

import pandas as pd
import streamlit as st

import weather_f as wf

st.set_page_config(page_title="Weather Tracker", layout="centered")
st.title("🌦️ Weather Tracker")

if "df" not in st.session_state:
    st.session_state.df = wf.load_data()

df = st.session_state.df

menu = st.sidebar.radio(
    "Menu",
    [
        "Record a new observation",
        "View weather statistics",
        "Search observations by date",
        "View all observations",
        "Temperature trends",
        "Filter by month/season",
        "Predict tomorrow's weather",
        "Compare years",
        "Record-breaking weather",
    ],
)

# Record a new observation
if menu == "Record a new observation":
    st.subheader("Record a New Weather Observation")
    with st.form("new_obs"):
        date_str = st.text_input("Date (MM-DD-YYYY)")
        temperature = st.number_input("Temperature (°C)", format="%.1f")
        condition = st.selectbox(
            "Weather condition",
            ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy", "Foggy", "Stormy", "Other"],
        )
        humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, step=1)
        wind_speed = st.number_input("Wind speed (km/h)", min_value=0.0, format="%.1f")
        submitted = st.form_submit_button("Save Observation")

    if submitted:
        if wf.validate_date(date_str) is None:
            st.error("Invalid date. Please use MM-DD-YYYY format.")
        else:
            st.session_state.df = wf.save_observation(
                date_str, temperature, condition, humidity, wind_speed
            )
            st.success(f"Observation for {date_str} saved.")

# View weather statistics
elif menu == "View weather statistics":
    st.subheader("Weather Statistics")
    stats = wf.get_statistics(df)
    if stats is None:
        st.info("No observations recorded yet.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Temp", f"{stats['avg_temp']}°C")
        col2.metric("Min Temp", f"{stats['min_temp']}°C")
        col3.metric("Max Temp", f"{stats['max_temp']}°C")
        st.write(f"**Most common condition:** {stats['most_common_condition']}")

# Search observations by date
elif menu == "Search observations by date":
    st.subheader("Search Observations by Date")
    search_date = st.text_input("Date to search (MM-DD-YYYY)")
    if st.button("Search"):
        if wf.validate_date(search_date) is None:
            st.error("Invalid date. Please use MM-DD-YYYY format.")
        else:
            results = wf.search_by_date(df, search_date)
            if results.empty:
                st.info("No observations found for that date.")
            else:
                st.dataframe(results, use_container_width=True)

# View all observations
elif menu == "View all observations":
    st.subheader("All Weather Observations")
    if df.empty:
        st.info("No observations recorded yet.")
    else:
        st.dataframe(df, use_container_width=True)

# Stretch: temperature trend (text-based graph)
elif menu == "Temperature trends":
    st.subheader("Temperature Trend (Text-Based Graph)")
    if df.empty:
        st.info("No observations recorded yet.")
    else:
        n = st.slider("Number of recent observations", 5, 50, 10)
        st.code(wf.generate_text_trend(df, n))

# Stretch: filter by month/season
elif menu == "Filter by month/season":
    st.subheader("Filter Observations")
    filter_type = st.radio("Filter by", ["Month", "Season"], horizontal=True)
    if filter_type == "Month":
        month = st.selectbox(
            "Month",
            list(range(1, 13)),
            format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"),
        )
        result = wf.filter_by_month(df, month)
    else:
        season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
        result = wf.filter_by_season(df, season)

    if result.empty:
        st.info("No observations match this filter.")
    else:
        st.dataframe(result, use_container_width=True)

# Stretch: predict tomorrow's weather
elif menu == "Predict tomorrow's weather":
    st.subheader("Tomorrow's Weather Prediction")
    prediction = wf.predict_tomorrow(df)
    if prediction is None:
        st.info("Not enough data to make a prediction.")
    else:
        st.write(f"**Date:** {prediction['date']}")
        st.write(f"**Predicted temperature:** {prediction['predicted_temp']}°C")
        st.write(f"**Predicted condition:** {prediction['predicted_condition']}")
        st.caption(f"Based on {prediction['sample_size']} observation(s) — {prediction['based_on']}.")

# Stretch: compare years
elif menu == "Compare years":
    st.subheader("Compare Years")
    years = wf.get_available_years(df)
    if len(years) < 2:
        st.info("Need observations from at least two different years to compare.")
    else:
        col1, col2 = st.columns(2)
        year1 = col1.selectbox("Year 1", years, index=len(years) - 1)
        year2 = col2.selectbox("Year 2", years, index=0)
        comparison = wf.compare_years(df, year1, year2)
        st.dataframe(comparison, use_container_width=True)

# Stretch: record-breaking weather
elif menu == "Record-breaking weather":
    st.subheader("Record-Breaking Weather")
    records = wf.get_records(df)
    if records is None:
        st.info("No observations recorded yet.")
    else:
        st.write(f"🔥 **Hottest day:** {records['hottest'][0]} — {records['hottest'][1]}°C")
        st.write(f"❄️ **Coldest day:** {records['coldest'][0]} — {records['coldest'][1]}°C")
        st.write(f"💧 **Most humid day:** {records['most_humid'][0]} — {records['most_humid'][1]}%")
        st.write(f"💨 **Windiest day:** {records['windiest'][0]} — {records['windiest'][1]} km/h")