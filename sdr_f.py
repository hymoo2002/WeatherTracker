"""
sdr_f.py
SDR receiver functions for the Weather Tracker app.

Requirements:
    pip install numpy scipy sounddevice
"""

import os
import subprocess
import tempfile
import threading
from datetime import datetime

import numpy as np
from scipy.io import wavfile
from scipy.signal import decimate

# ── Satellite / radio frequency presets ──────────────────────────
# Each entry: (freq_mhz, sample_rate_mhz, bandwidth_khz,
#              default_mode, short_description)
#
# Modes:  "FM"  = wideband FM demodulation
#         "AM"  = AM envelope detection
#         "Raw" = save IQ file only (for SatDump)

PRESETS = {
    # ── Testing ──────────────────────────────────────────────────
    "FM Radio (test ~98 MHz)":
        (98.0,    2.5, 200,  "FM",  "Local FM station – good first test"),

    # ── VHF weather satellites  (V-dipole / QFH antenna) ────────
    "NOAA-15  APT  137.620 MHz":
        (137.620,   2.5, 40,  "FM",  "APT weather image, ~40 kHz signal"),
    "NOAA-18  APT  137.9125 MHz":
        (137.9125,  2.5, 40,  "FM",  "APT weather image, ~40 kHz signal"),
    "NOAA-19  APT  137.100 MHz":
        (137.100,   2.5, 40,  "FM",  "APT weather image, ~40 kHz signal"),
    "Meteor M2-4  LRPT  137.100 MHz":
        (137.100,   2.5, 150, "Raw", "Digital LRPT – decode in SatDump"),

    # ── L-band satellites  (dish / helix antenna) ────────────────
    "NOAA-15  HRPT  1702.5 MHz":
        (1702.5,  3.0, 2000, "Raw", "High-res digital – needs dish"),
    "NOAA-18  HRPT  1707.0 MHz":
        (1707.0,  3.0, 2000, "Raw", "High-res digital – needs dish"),
    "NOAA-19  HRPT  1698.0 MHz":
        (1698.0,  3.0, 2000, "Raw", "High-res digital – needs dish"),
    "MetOp-B  AHRPT  1701.3 MHz":
        (1701.3,  3.0, 2000, "Raw", "EUMETSAT polar orbiter – needs dish"),
    "MetOp-C  AHRPT  1701.3 MHz":
        (1701.3,  3.0, 2000, "Raw", "EUMETSAT polar orbiter – needs dish"),
    "Meteor M2-4  HRPT  1700 MHz":
        (1700.0,  3.0, 2000, "Raw", "Russian polar orbiter – needs dish"),
    "FengYun-3E  AHRPT  1704.5 MHz":
        (1704.5,  3.0, 2000, "Raw", "Chinese polar orbiter – needs dish"),
    "GOES-16  HRIT  1694.1 MHz":
        (1694.1,  3.0, 1200, "Raw", "Geostationary weather – needs dish"),
    "GOES-18  HRIT  1694.1 MHz":
        (1694.1,  3.0, 1200, "Raw", "Geostationary weather – needs dish"),

    # ── Marine / other ──────────────────────────────────────────
    "AIS (ships)  161.975 MHz":
        (161.975, 2.5, 25,  "FM",  "Ship tracking channel 1"),
    "AIS (ships)  162.025 MHz":
        (162.025, 2.5, 25,  "FM",  "Ship tracking channel 2"),
    "Inmarsat STD-C  1539.9 MHz":
        (1539.9,  2.5, 50,  "Raw", "Maritime safety messages"),
}

AUDIO_RATE = 48000  # output WAV sample rate in Hz


# ── HackRF hardware check ───────────────────────────────────────

def check_hackrf():
    """
    Run hackrf_info and return (ok, message).
    ok   = True  if a HackRF was found
    ok   = False if not found or tools missing
    """
    try:
        result = subprocess.run(
            ["hackrf_info"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return True, result.stdout
        return False, result.stderr or "hackrf_info returned an error."
    except FileNotFoundError:
        return False, (
            "Radio_info not found.\n"
            "Install SDR tools first:\n"
            "  install via Zadig + PothosSDR bundle"
        )
    except subprocess.TimeoutExpired:
        return False, "Device_info timed out – is the device stuck?"
    except Exception as err:
        return False, str(err)


# ── IQ capture (for Record mode) ────────────────────────────────

def capture_iq(freq_hz, sample_rate_hz, duration_sec, bandwidth_hz,
               lna_gain=32, vga_gain=40, amp_on=False):
    """
    Capture IQ samples with hackrf_transfer (blocking, finite).

    Returns
    -------
    samples : numpy complex64 array   (None on failure)
    iq_path : str   path to the raw .iq file on disk
    error   : str   error message (empty on success)
    """
    num_samples = int(sample_rate_hz * duration_sec)
    iq_path = os.path.join(tempfile.gettempdir(), "hackrf_capture.iq")

    command = [
        "hackrf_transfer",
        "-r", iq_path,
        "-f", str(int(freq_hz)),
        "-s", str(int(sample_rate_hz)),
        "-n", str(num_samples),
        "-l", str(int(lna_gain)),
        "-g", str(int(vga_gain)),
        "-a", "1" if amp_on else "0",
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=duration_sec + 15,
        )
    except subprocess.TimeoutExpired:
        return None, iq_path, "Capture timed out."
    except FileNotFoundError:
        return None, iq_path, "hackrf_transfer not found."
    except Exception as err:
        return None, iq_path, str(err)

    if result.returncode != 0:
        return None, iq_path, result.stderr or "hackrf_transfer failed."

    samples = read_iq_file(iq_path)
    if samples is None:
        return None, iq_path, "Could not read the .iq file."

    # Apply software bandwidth filter
    samples = _apply_bw_filter(samples, sample_rate_hz, bandwidth_hz)

    return samples, iq_path, ""


def read_iq_file(path):
    """
    Read a hackrf_transfer .iq file (interleaved int8) into a
    numpy complex64 array scaled to -1 … +1.
    Returns None if the file is missing or empty.
    """
    if not os.path.isfile(path):
        return None
    raw = np.fromfile(path, dtype=np.int8)
    if len(raw) < 2:
        return None
    i_samples = raw[0::2].astype(np.float32)
    q_samples = raw[1::2].astype(np.float32)
    samples = (i_samples + 1j * q_samples) / 128.0
    return samples


# ── Bandwidth filter ─────────────────────────────────────────────

def _apply_bw_filter(samples, sample_rate_hz, bandwidth_hz):
    """
    Decimate the IQ signal so that the effective sample rate roughly
    matches the desired bandwidth.  Simple and fast — good enough for
    listening; SatDump should use the unfiltered .iq file anyway.
    """
    if bandwidth_hz >= sample_rate_hz:
        return samples
    factor = max(1, int(sample_rate_hz / bandwidth_hz))
    if factor <= 1:
        return samples
    # scipy decimate applies an anti-alias filter before downsampling
    real_dec = decimate(samples.real, factor)
    imag_dec = decimate(samples.imag, factor)
    return real_dec + 1j * imag_dec


# ── Demodulation ─────────────────────────────────────────────────

def demod_fm(samples, sample_rate_hz):
    """FM demodulation: derivative of unwrapped phase, then downsample."""
    phase = np.angle(samples)
    audio = np.diff(np.unwrap(phase))

    factor = max(1, int(sample_rate_hz / AUDIO_RATE))
    if factor > 1:
        audio = decimate(audio, factor)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9
    return audio.astype(np.float32)


def demod_am(samples, sample_rate_hz):
    """AM envelope detection: magnitude, remove DC, then downsample."""
    audio = np.abs(samples).astype(np.float64)
    audio = audio - np.mean(audio)

    factor = max(1, int(sample_rate_hz / AUDIO_RATE))
    if factor > 1:
        audio = decimate(audio, factor)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9
    return audio.astype(np.float32)


# ── Save helpers ─────────────────────────────────────────────────

def save_wav(audio, filepath):
    """Save a float32 audio array as a 16-bit WAV at AUDIO_RATE."""
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(filepath, AUDIO_RATE, audio_int16)
    return filepath


def save_raw_iq(samples, filepath):
    """Save complex samples to interleaved int8 .iq (cs8 for SatDump)."""
    iq = np.empty(len(samples) * 2, dtype=np.int8)
    iq[0::2] = (samples.real * 128).clip(-128, 127).astype(np.int8)
    iq[1::2] = (samples.imag * 128).clip(-128, 127).astype(np.int8)
    iq.tofile(filepath)
    return filepath


# ── Spectrogram for display ──────────────────────────────────────

def make_spectrogram(samples, sample_rate_hz, center_freq_hz,
                     fft_size=1024):
    """
    Build a waterfall array from IQ samples.

    Returns
    -------
    spec_db : 2-D array  (rows=time, cols=frequency)
    extent  : [freq_min_MHz, freq_max_MHz, time_max_s, 0]
    """
    num_rows = len(samples) // fft_size
    if num_rows == 0:
        return None, None

    spec = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        chunk = samples[i * fft_size : (i + 1) * fft_size]
        fft_vals = np.fft.fftshift(np.fft.fft(chunk))
        spec[i, :] = 10 * np.log10(np.abs(fft_vals) ** 2 + 1e-12)

    freq_min = (center_freq_hz - sample_rate_hz / 2) / 1e6
    freq_max = (center_freq_hz + sample_rate_hz / 2) / 1e6
    time_max = len(samples) / sample_rate_hz
    extent = [freq_min, freq_max, time_max, 0]

    return spec, extent


# =====================================================================
#  Live Listener  — continuous audio through the speakers
# =====================================================================

class LiveListener:
    """
    Streams IQ from HackRF via `hackrf_transfer -r -` (stdout pipe),
    demodulates FM or AM in real-time, and plays audio through the
    system speakers using the `sounddevice` library.

    Usage from Streamlit (store in session_state):
        listener = LiveListener()
        listener.start(freq, sr, bw, "FM", lna, vga, amp)
        ...
        listener.stop()
    """

    def __init__(self):
        self.running = False
        self.process = None
        self.thread = None
        self.error = ""           # last error message (shown in UI)
        self._prev_sample = 0+0j  # keeps FM demod continuous

    # ── public API ───────────────────────────────────────────────

    def is_running(self):
        return self.running and self.thread is not None and self.thread.is_alive()

    def start(self, freq_hz, sample_rate_hz, bandwidth_hz, mode,
              lna_gain=32, vga_gain=40, amp_on=False):
        """Launch hackrf_transfer and start the audio thread."""
        if self.is_running():
            self.stop()

        self.running = True
        self.error = ""
        self._prev_sample = 0+0j

        cmd = [
            "hackrf_transfer",
            "-r", "-",                       # stream to stdout
            "-f", str(int(freq_hz)),
            "-s", str(int(sample_rate_hz)),
            "-l", str(int(lna_gain)),
            "-g", str(int(vga_gain)),
            "-a", "1" if amp_on else "0",
        ]

        try:
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            self.error = "hackrf_transfer not found."
            self.running = False
            return
        except Exception as err:
            self.error = str(err)
            self.running = False
            return

        self.thread = threading.Thread(
            target=self._audio_loop,
            args=(sample_rate_hz, bandwidth_hz, mode),
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        """Kill the hackrf_transfer process and wait for the thread."""
        self.running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None

    # ── internal audio loop (runs in background thread) ──────────

    def _audio_loop(self, sample_rate_hz, bandwidth_hz, mode):
        """Read IQ → decimate → demodulate → speakers."""
        try:
            import sounddevice as sd
        except ImportError:
            self.error = (
                "sounddevice not installed.\n"
                "Run:  pip install sounddevice"
            )
            self.running = False
            return

        # Decimate IQ to roughly the desired bandwidth
        iq_dec = max(1, int(sample_rate_hz / bandwidth_hz))
        effective_sr = sample_rate_hz / iq_dec

        # Decimate demodulated audio to 48 kHz
        audio_dec = max(1, int(effective_sr / AUDIO_RATE))
        actual_audio_rate = effective_sr / audio_dec

        # How many IQ bytes to read per chunk (~100 ms of full-rate data)
        chunk_samples = int(sample_rate_hz * 0.1)
        chunk_bytes = chunk_samples * 2          # I + Q, each 1 byte

        try:
            stream = sd.OutputStream(
                samplerate=actual_audio_rate,
                channels=1,
                dtype="float32",
                blocksize=2048,
            )
            stream.start()
        except Exception as err:
            self.error = f"Audio output error: {err}"
            self.running = False
            return

        try:
            while self.running:
                raw = self.process.stdout.read(chunk_bytes)
                if not raw:
                    break

                # ── parse IQ ──
                data = np.frombuffer(raw, dtype=np.int8)
                if len(data) < 2:
                    continue
                iq = (data[0::2].astype(np.float32)
                      + 1j * data[1::2].astype(np.float32)) / 128.0

                # ── decimate IQ to bandwidth ──
                if iq_dec > 1:
                    iq = iq[::iq_dec]

                # ── demodulate ──
                if mode == "FM":
                    full = np.concatenate(([self._prev_sample], iq))
                    self._prev_sample = iq[-1]
                    audio = np.diff(np.unwrap(np.angle(full)))
                elif mode == "AM":
                    audio = np.abs(iq)
                    audio = audio - np.mean(audio)
                else:
                    continue     # Raw mode has nothing to play

                # ── decimate to audio rate ──
                if audio_dec > 1:
                    audio = audio[::audio_dec]

                # ── normalise ──
                audio = audio.astype(np.float32)
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio / peak * 0.7

                stream.write(audio)

        except Exception as err:
            self.error = f"Listener error: {err}"
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
            self.running = False
