# 🌦️ Weather Tracker

A weather observation tracker built with Python and Streamlit. Log daily weather, view stats, spot trends, and even pull signals from weather satellites using an SDR receiver.

I built this while learning Python, so the code is straightforward and easy to follow.

## Features

* **Record observations** — date, temperature, humidity, wind speed, and condition
* **Statistics** — average, min, max temperature and most common condition
* **Search \& filter** — look up observations by date, month, or season
* **Temperature trends** — ASCII bar chart of recent readings
* **Year comparison** — side-by-side monthly averages between two years
* **Tomorrow's prediction** — simple forecast based on same-month history
* **Record-breaking weather** — hottest, coldest, most humid, windiest days
* **Satellite Weather (SDR)** — tune into weather satellites with a HackRF One, listen live or record IQ files for decoding in SatDump

## Requirements

* Python 3.9+
* Pluto SDR, or any equivalent (only for the SDR section)

## Setup

```bash
pip install streamlit pandas numpy scipy matplotlib sounddevice
```

## Usage

```bash
streamlit run app.py
```

The app saves observations to `weather\\\\\\\_data.csv` in the same folder. Dates use DD-MM-YYYY format.

## SDR Section

The Satellite Weather page lets you connect a HackRF One and:

* **Listen** continuously to FM/AM signals through your speakers (like SDR#)
* **Record** IQ captures for offline decoding

Comes with built-in presets for NOAA APT, Meteor LRPT, MetOp, GOES, FengYun, AIS, and more. Recordings are saved as int8 IQ (cs8 format) — open them directly in [SatDump](https://github.com/SatDump/SatDump).

## Files

|File|What it does|
|-|-|
|`app.py`|Streamlit UI — all the pages and layout|
|`weather\\\\\\\_f.py`|Weather data functions — stats, filters, predictions|
|`sdr\\\\\\\_f.py`|SDR receiver functions — capture, demodulation, live listener|

## Deployed App Link

https://weathertracker2002.streamlit.app/

## Overview Video

https://youtu.be/zbPYGdtrmVs

