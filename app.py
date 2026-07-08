"""
app.py
Streamlit UI for the Weather Tracker app.
"""

import os
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from functions import sdr_f as sdr
from functions import weather_f as wf

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
        "Satellite Weather",
    ],
)


# Record a new observation

if menu == "Record a new observation":
    st.subheader("Record a New Weather Observation")
    with st.form("new_obs"):
        date_str = st.text_input("Date (DD-MM-YYYY)")
        temperature = st.number_input("Temperature (°C)", format="%.1f", step=0.5)
        condition = st.selectbox(
            "Weather condition",
            ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy", "Foggy", "Stormy", "Other"],
        )
        humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, step=1)
        wind_speed = st.number_input("Wind speed (km/h)", min_value=0.0, format="%.1f", step=0.5)
        submitted = st.form_submit_button("Save Observation")

    if submitted:
        if wf.validate_date(date_str) is None:
            st.error("Invalid date. Please use DD-MM-YYYY format.")
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
    search_date = st.text_input("Date to search (DD-MM-YYYY)")
    if st.button("Search"):
        if wf.validate_date(search_date) is None:
            st.error("Invalid date. Please use DD-MM-YYYY format.")
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


# Temperature trends

elif menu == "Temperature trends":
    st.subheader("Temperature Trend (Text-Based Graph)")
    if df.empty:
        st.info("No observations recorded yet.")
    else:
        n = st.slider("Number of recent observations", 5, 50, 10)
        st.code(wf.generate_text_trend(df, n))


# Filter by month/season

elif menu == "Filter by month/season":
    st.subheader("Filter Observations")

    if df.empty:
        st.info("No observations recorded yet.")
    else:
        tab_month, tab_season = st.tabs(["By Month", "By Season"])

        with tab_month:
            month_name = st.selectbox("Select month", wf.MONTH_NAMES)
            month_number = wf.MONTH_NAMES.index(month_name) + 1
            month_result = wf.filter_by_month(df, month_number)
            if month_result.empty:
                st.info("No observations found for this month.")
            else:
                st.dataframe(month_result, use_container_width=True)

        with tab_season:
            season = st.selectbox(
                "Select season",
                ["Winter", "Spring", "Summer", "Fall"],
            )
            season_result = wf.filter_by_season(df, season)
            if season_result.empty:
                st.info("No observations found for this season.")
            else:
                st.dataframe(season_result, use_container_width=True)


# Predict tomorrow's weather 

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


# Compare years

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


# Record-breaking weather

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


# Satellite Weather (PlutoSDR receiver)

elif menu == "Satellite Weather":
    st.subheader("📡 Satellite Weather — Receiver")
    st.caption(
        "Tune into radio and satellite frequencies with your SDR.  "
        "Listen live or record for decoding."
    )

    # persistent listener
    if "listener" not in st.session_state:
        st.session_state.listener = sdr.LiveListener()
    listener = st.session_state.listener

    # SDR connection check
    with st.expander("Check SDR connection", expanded=False):
        if st.button("Check SDR"):
            ok, info = sdr.check_sdr()
            if ok:
                st.success("SDR detected!")
                st.code(info)
            else:
                st.error("SDR not found.")
                st.code(info)

    st.divider()

    # Frequency selection
    preset_names = ["Manual entry"] + list(sdr.PRESETS.keys())
    chosen_preset = st.selectbox("Frequency preset", preset_names)

    if chosen_preset == "Manual entry":
        freq_mhz = st.number_input("Frequency (MHz)", value=137.1,
                                    format="%.4f")
        sample_rate_mhz = st.number_input("Sample rate (MHz)", value=2.5,
                                           format="%.1f")
        bandwidth_khz = st.number_input("Bandwidth (kHz)", value=40,
                                         min_value=1, step=1)
        mode = st.selectbox("Demodulation", ["FM", "AM", "Raw"])
    else:
        preset = sdr.PRESETS[chosen_preset]
        freq_mhz        = preset[0]
        sample_rate_mhz  = preset[1]
        bandwidth_khz    = preset[2]
        mode             = preset[3]
        st.info(f"ℹ️ {preset[4]}")
        st.write(
            f"**Frequency:** {freq_mhz} MHz  ·  "
            f"**Sample rate:** {sample_rate_mhz} MHz  ·  "
            f"**Bandwidth:** {bandwidth_khz} kHz  ·  "
            f"**Mode:** {mode}"
        )

    # Gain setting (single slider for AD9361)
    with st.expander("Gain settings", expanded=False):
        rx_gain = st.slider("RX gain (dB)", 0, 73, 40, step=1,
                            help="AD9361 hardware gain, 0–73 dB")

    # convert units
    freq_hz = freq_mhz * 1e6
    sr_hz   = sample_rate_mhz * 1e6
    bw_hz   = bandwidth_khz * 1e3

    # ── LISTEN 
    st.divider()
    st.subheader("🎧 Listen")
    st.caption(
        "Continuous playback through your speakers.  "
        "Works with FM and AM modes."
    )

    if mode == "Raw":
        st.warning(
            "Listening is not available in Raw mode (digital signals "
            "can't be played as audio).  Switch to FM or AM, or use "
            "Record below to capture an IQ file for SatDump."
        )

    if listener.is_running():
        st.success("🟢  Listening …")
        if st.button("⏹  Stop listening"):
            listener.stop()
            st.rerun()
    else:
        if listener.error:
            st.error(listener.error)
        if mode in ("FM", "AM"):
            if st.button("▶️  Start listening"):
                listener.start(
                    freq_hz, sr_hz, bw_hz, mode,
                    rx_gain=rx_gain,
                )
                import time; time.sleep(0.3)
                if listener.error:
                    st.error(listener.error)
                else:
                    st.rerun()

    # RECORD
    st.divider()
    st.subheader("🔴 Record")
    st.caption(
        "Capture a recording.  For a full NOAA satellite "
        "pass try ~900 s (15 min)."
    )

    record_sec = st.number_input(
        "Recording duration (seconds)",
        min_value=1, value=30, step=10, key="record_dur",
    )

    if st.button("🔴  Start recording", type="primary"):
        if listener.is_running():
            listener.stop()

        with st.spinner(
            f"Recording {record_sec}s at {freq_mhz} MHz …"
        ):
            samples, iq_path, error = sdr.capture_iq(
                freq_hz, sr_hz, record_sec, bw_hz,
                rx_gain=rx_gain,
            )

        if error:
            st.error(f"Capture failed: {error}")
        elif samples is None or len(samples) == 0:
            st.error("No samples captured.")
        else:
            st.success(
                f"Captured {len(samples):,} samples "
                f"({len(samples) / (sr_hz / max(1, int(sr_hz / bw_hz))):.1f}s)"
            )

            # spectrogram
            st.write("**Spectrogram (waterfall)**")
            spec, extent = sdr.make_spectrogram(samples, sr_hz, freq_hz)
            if spec is not None:
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.imshow(spec, aspect="auto", extent=extent,
                          cmap="viridis")
                ax.set_xlabel("Frequency (MHz)")
                ax.set_ylabel("Time (s)")
                st.pyplot(fig)
                plt.close(fig)

            # demodulated audio
            if mode in ("FM", "AM"):
                st.write(f"**{mode} demodulated audio**")
                if mode == "FM":
                    audio = sdr.demod_fm(samples, sr_hz, bw_hz)
                else:
                    audio = sdr.demod_am(samples, sr_hz, bw_hz)

                wav_path = os.path.join(tempfile.gettempdir(),
                                        "pluto_audio.wav")
                sdr.save_wav(audio, wav_path)
                st.audio(wav_path, format="audio/wav")

                with open(wav_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download WAV", f,
                        file_name=f"capture_{freq_mhz}MHz.wav",
                        mime="audio/wav", key="dl_wav_rec",
                    )
            elif mode == "Raw":
                st.caption(
                    "Raw mode — no audio demodulation.  "
                    "Download the IQ file and open it in SatDump."
                )

            # raw IQ download
            st.write("**Raw IQ file** (for SatDump)")
            st.caption(
                f"Format: int8 interleaved IQ (cs8)  ·  "
                f"Sample rate: {sample_rate_mhz} MHz  ·  "
                f"Center: {freq_mhz} MHz"
            )
            with open(iq_path, "rb") as f:
                st.download_button(
                    "⬇️ Download .iq (cs8)", f,
                    file_name=(f"capture_{freq_mhz}MHz_"
                               f"{sample_rate_mhz}Msps.iq"),
                    mime="application/octet-stream", key="dl_iq_rec",
                )