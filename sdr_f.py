"""
sdr_f.py
SDR receiver functions for the Weather Tracker app.
Uses pyadi-iio for PlutoSDR / AD9361-compatible devices (RX only).

Requirements:
    pip install pyadi-iio numpy scipy sounddevice
"""

import os
import tempfile
import threading

import numpy as np
from scipy.io import wavfile
from scipy.signal import decimate, lfilter

try:
    import adi
    ADI_AVAILABLE = True
except ImportError:
    ADI_AVAILABLE = False

# ── Device config ────────────────────────────────────────────────
SDR_URI = "ip:192.168.2.1"
AUDIO_RATE = 48000

# ── Frequency presets ────────────────────────────────────────────
# (freq_mhz, sample_rate_mhz, bandwidth_khz, mode, description)

PRESETS = {
    # Testing
    "FM Radio (test ~98 MHz)":
        (98.0, 2.5, 200, "FM", "Local FM station – good first test"),

    # VHF weather satellites
    "NOAA-15  APT  137.620 MHz":
        (137.620, 2.5, 40, "FM", "APT weather image, ~40 kHz signal"),
    "NOAA-18  APT  137.9125 MHz":
        (137.9125, 2.5, 40, "FM", "APT weather image, ~40 kHz signal"),
    "NOAA-19  APT  137.100 MHz":
        (137.100, 2.5, 40, "FM", "APT weather image, ~40 kHz signal"),
    "Meteor M2-4  LRPT  137.100 MHz":
        (137.100, 2.5, 150, "Raw", "Digital LRPT – decode in SatDump"),

    # L-band satellites
    "NOAA-15  HRPT  1702.5 MHz":
        (1702.5, 3.0, 2000, "Raw", "High-res digital – needs dish"),
    "NOAA-18  HRPT  1707.0 MHz":
        (1707.0, 3.0, 2000, "Raw", "High-res digital – needs dish"),
    "NOAA-19  HRPT  1698.0 MHz":
        (1698.0, 3.0, 2000, "Raw", "High-res digital – needs dish"),
    "MetOp-B  AHRPT  1701.3 MHz":
        (1701.3, 3.0, 2000, "Raw", "EUMETSAT polar orbiter – needs dish"),
    "MetOp-C  AHRPT  1701.3 MHz":
        (1701.3, 3.0, 2000, "Raw", "EUMETSAT polar orbiter – needs dish"),
    "Meteor M2-4  HRPT  1700 MHz":
        (1700.0, 3.0, 2000, "Raw", "Russian polar orbiter – needs dish"),
    "FengYun-3E  AHRPT  1704.5 MHz":
        (1704.5, 3.0, 2000, "Raw", "Chinese polar orbiter – needs dish"),
    "GOES-16  HRIT  1694.1 MHz":
        (1694.1, 3.0, 1200, "Raw", "Geostationary weather – needs dish"),
    "GOES-18  HRIT  1694.1 MHz":
        (1694.1, 3.0, 1200, "Raw", "Geostationary weather – needs dish"),

    # Marine
    "AIS (ships)  161.975 MHz":
        (161.975, 2.5, 25, "FM", "Ship tracking channel 1"),
    "AIS (ships)  162.025 MHz":
        (162.025, 2.5, 25, "FM", "Ship tracking channel 2"),
    "Inmarsat STD-C  1539.9 MHz":
        (1539.9, 2.5, 50, "Raw", "Maritime safety messages"),
}


# ═════════════════════════════════════════════════════════════════
#  DSP helpers
# ═════════════════════════════════════════════════════════════════

def _safe_decimate(x, factor):
    """Decimate a real signal in stages of max 12x each."""
    if factor <= 1:
        return x
    while factor > 1:
        step = min(factor, 12)
        x = decimate(x.astype(np.float64), step)
        factor //= step
    return x


def _safe_decimate_iq(iq, factor):
    """Decimate a complex IQ signal (filter both I and Q)."""
    if factor <= 1:
        return iq
    real = _safe_decimate(iq.real, factor)
    imag = _safe_decimate(iq.imag, factor)
    return (real + 1j * imag).astype(np.complex64)


def _deemphasis(audio, sample_rate, tau=50e-6):
    """FM broadcast de-emphasis filter (50 µs for most of world)."""
    alpha = 1.0 / (sample_rate * tau + 1.0)
    return lfilter([alpha], [1.0, -(1.0 - alpha)], audio)


# ═════════════════════════════════════════════════════════════════
#  SDR connection
# ═════════════════════════════════════════════════════════════════

def check_sdr():
    if not ADI_AVAILABLE:
        return False, (
            "pyadi-iio is not installed.\n"
            "Run:  pip install pyadi-iio"
        )
    try:
        radio = adi.ad9361(uri=SDR_URI)
        info = (
            f"SDR connected at {SDR_URI}\n"
            f"Sample rate : {radio.sample_rate} Hz\n"
            f"RX LO       : {radio.rx_lo} Hz\n"
        )
        del radio
        return True, info
    except Exception as e:
        return False, f"Cannot connect to SDR at {SDR_URI}:\n{e}"


def _make_radio(freq_hz, sample_rate_hz, rx_gain=40, buf_size=2**16):
    """Create an ad9361 configured for RX-only on channel 0."""
    radio = adi.ad9361(uri=SDR_URI)
    radio.rx_enabled_channels = [0]
    radio.sample_rate = int(sample_rate_hz)
    # Set analog filter to full sample rate — narrow filtering is done
    # in software, just like SDR# does.
    radio.rx_rf_bandwidth = int(sample_rate_hz)
    radio.rx_lo = int(freq_hz)
    radio.gain_control_mode_chan0 = "manual"
    radio.rx_hardwaregain_chan0 = int(rx_gain)
    radio.rx_buffer_size = int(buf_size)
    radio._rxadc.set_kernel_buffers_count(1)
    return radio


# ═════════════════════════════════════════════════════════════════
#  IQ capture  (Record mode)
# ═════════════════════════════════════════════════════════════════

def capture_iq(freq_hz, sample_rate_hz, duration_sec, bandwidth_hz,
               rx_gain=40):
    """
    Capture IQ samples from the PlutoSDR.

    Returns  (samples, iq_path, error)
    -  samples are at the ORIGINAL sample rate (not decimated)
    -  iq_path points to the raw cs8 file for SatDump
    """
    iq_path = os.path.join(tempfile.gettempdir(), "pluto_capture.iq")

    if not ADI_AVAILABLE:
        return None, iq_path, "pyadi-iio is not installed."

    try:
        radio = _make_radio(freq_hz, sample_rate_hz, rx_gain)
    except Exception as e:
        return None, iq_path, str(e)

    total_samples = int(sample_rate_hz * duration_sec)
    chunks = []
    collected = 0

    try:
        for _ in range(10):
            radio.rx()

        while collected < total_samples:
            data = radio.rx()
            chunks.append(data.copy())
            collected += len(data)
    except Exception as e:
        return None, iq_path, str(e)
    finally:
        del radio

    samples = np.concatenate(chunks)[:total_samples]

    # remove DC offset (LO leakage)
    samples = samples - np.mean(samples)

    # normalise to −1…+1
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples = (samples / peak).astype(np.complex64)

    # save raw IQ as cs8 for SatDump
    save_raw_iq(samples, iq_path)

    return samples, iq_path, ""


# ═════════════════════════════════════════════════════════════════
#  Demodulation
# ═════════════════════════════════════════════════════════════════

def demod_fm(samples, sample_rate_hz, bandwidth_hz=None):
    """
    FM demodulation with proper filtering.

    1.  Low-pass filter + decimate IQ to the signal bandwidth
    2.  FM discriminator  (diff of unwrapped phase)
    3.  Decimate audio to 48 kHz
    4.  De-emphasis for wideband FM  (>100 kHz bandwidth)
    5.  Normalise
    """
    # ── step 1: narrow to signal bandwidth ──
    if bandwidth_hz and bandwidth_hz < sample_rate_hz:
        iq_dec = int(sample_rate_hz / bandwidth_hz)
        if iq_dec > 1:
            samples = _safe_decimate_iq(samples, iq_dec)
            sample_rate_hz = sample_rate_hz / iq_dec

    # ── step 2: FM discriminator ──
    phase = np.angle(samples)
    audio = np.diff(np.unwrap(phase))

    # ── step 3: decimate to audio rate ──
    audio_dec = max(1, int(sample_rate_hz / AUDIO_RATE))
    if audio_dec > 1:
        audio = _safe_decimate(audio, audio_dec)
        sample_rate_hz = sample_rate_hz / audio_dec

    # ── step 4: de-emphasis for broadcast FM ──
    if bandwidth_hz and bandwidth_hz >= 100_000:
        audio = _deemphasis(audio, sample_rate_hz, tau=50e-6)

    # ── step 5: normalise ──
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9
    return audio.astype(np.float32)


def demod_am(samples, sample_rate_hz, bandwidth_hz=None):
    """AM envelope detection with proper filtering."""
    # narrow to signal bandwidth
    if bandwidth_hz and bandwidth_hz < sample_rate_hz:
        iq_dec = int(sample_rate_hz / bandwidth_hz)
        if iq_dec > 1:
            samples = _safe_decimate_iq(samples, iq_dec)
            sample_rate_hz = sample_rate_hz / iq_dec

    # envelope
    audio = np.abs(samples).astype(np.float64)
    audio = audio - np.mean(audio)

    # decimate to audio rate
    audio_dec = max(1, int(sample_rate_hz / AUDIO_RATE))
    if audio_dec > 1:
        audio = _safe_decimate(audio, audio_dec)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9
    return audio.astype(np.float32)


# ═════════════════════════════════════════════════════════════════
#  Save / load helpers
# ═════════════════════════════════════════════════════════════════

def save_wav(audio, filepath):
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(filepath, AUDIO_RATE, audio_int16)
    return filepath


def save_raw_iq(samples, filepath):
    """Save normalised (−1…+1) complex samples as interleaved int8 cs8."""
    iq = np.empty(len(samples) * 2, dtype=np.int8)
    iq[0::2] = (samples.real * 127).clip(-128, 127).astype(np.int8)
    iq[1::2] = (samples.imag * 127).clip(-128, 127).astype(np.int8)
    iq.tofile(filepath)
    return filepath


def read_iq_file(path):
    if not os.path.isfile(path):
        return None
    raw = np.fromfile(path, dtype=np.int8)
    if len(raw) < 2:
        return None
    i = raw[0::2].astype(np.float32)
    q = raw[1::2].astype(np.float32)
    return ((i + 1j * q) / 128.0).astype(np.complex64)


# ═════════════════════════════════════════════════════════════════
#  Spectrogram
# ═════════════════════════════════════════════════════════════════

def make_spectrogram(samples, sample_rate_hz, center_freq_hz, fft_size=1024):
    num_rows = len(samples) // fft_size
    if num_rows == 0:
        return None, None

    spec = np.zeros((num_rows, fft_size))
    win = np.hamming(fft_size)
    for i in range(num_rows):
        chunk = samples[i * fft_size:(i + 1) * fft_size]
        fft_vals = np.fft.fftshift(np.fft.fft(chunk * win))
        spec[i, :] = 10 * np.log10(np.abs(fft_vals) ** 2 + 1e-12)

    freq_min = (center_freq_hz - sample_rate_hz / 2) / 1e6
    freq_max = (center_freq_hz + sample_rate_hz / 2) / 1e6
    time_max = len(samples) / sample_rate_hz
    return spec, [freq_min, freq_max, time_max, 0]


# ═════════════════════════════════════════════════════════════════
#  Live Listener
# ═════════════════════════════════════════════════════════════════

class LiveListener:
    """
    Continuously reads IQ from PlutoSDR, demodulates FM/AM, and plays
    audio through the speakers using sounddevice.
    """

    def __init__(self):
        self.running = False
        self._radio = None
        self.thread = None
        self.error = ""
        self._prev_sample = 0 + 0j

    def is_running(self):
        return self.running and self.thread is not None and self.thread.is_alive()

    def start(self, freq_hz, sample_rate_hz, bandwidth_hz, mode,
              rx_gain=40):
        if self.is_running():
            self.stop()

        self.running = True
        self.error = ""
        self._prev_sample = 0 + 0j

        if not ADI_AVAILABLE:
            self.error = "pyadi-iio is not installed."
            self.running = False
            return

        try:
            self._radio = _make_radio(
                freq_hz, sample_rate_hz, rx_gain, buf_size=2**16,
            )
            for _ in range(10):
                self._radio.rx()
        except Exception as e:
            self.error = str(e)
            self.running = False
            return

        self.thread = threading.Thread(
            target=self._audio_loop,
            args=(sample_rate_hz, bandwidth_hz, mode),
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None
        if self._radio:
            try:
                del self._radio
            except Exception:
                pass
            self._radio = None

    # ── background thread ────────────────────────────────────────

    def _audio_loop(self, sample_rate_hz, bandwidth_hz, mode):
        try:
            import sounddevice as sd
        except ImportError:
            self.error = (
                "sounddevice not installed.\n"
                "Run:  pip install sounddevice"
            )
            self.running = False
            return

        # ── compute decimation plan ──
        iq_dec = max(1, int(sample_rate_hz / bandwidth_hz))
        effective_sr = sample_rate_hz / iq_dec
        audio_dec = max(1, int(effective_sr / AUDIO_RATE))
        actual_audio_rate = effective_sr / audio_dec

        # wideband FM gets de-emphasis
        do_deemph = (mode == "FM" and bandwidth_hz >= 100_000)

        try:
            stream = sd.OutputStream(
                samplerate=actual_audio_rate,
                channels=1,
                dtype="float32",
                blocksize=4096,
            )
            stream.start()
        except Exception as e:
            self.error = f"Audio output error: {e}"
            self.running = False
            return

        try:
            while self.running:
                iq = self._radio.rx()

                # remove DC offset
                iq = iq - np.mean(iq)

                # ── fast IQ decimation (reshape + mean) ──
                # Averages groups of iq_dec samples then keeps one
                # value per group.  Acts as a boxcar anti-alias filter
                # + downsample in a single vectorised numpy op — orders
                # of magnitude faster than scipy.decimate.
                if iq_dec > 1:
                    n = (len(iq) // iq_dec) * iq_dec
                    iq = iq[:n].reshape(-1, iq_dec).mean(axis=1)

                # ── demodulate ──
                if mode == "FM":
                    full = np.concatenate(([self._prev_sample], iq))
                    self._prev_sample = iq[-1]
                    audio = np.diff(np.unwrap(np.angle(full)))
                elif mode == "AM":
                    audio = np.abs(iq).astype(np.float64)
                    audio -= np.mean(audio)
                else:
                    continue

                # ── fast audio decimation ──
                if audio_dec > 1:
                    n = (len(audio) // audio_dec) * audio_dec
                    audio = audio[:n].reshape(-1, audio_dec).mean(axis=1)

                # ── de-emphasis ──
                if do_deemph:
                    audio = _deemphasis(audio, actual_audio_rate, tau=50e-6)

                # ── normalise ──
                audio = audio.astype(np.float32)
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio / peak * 0.7
                stream.write(audio)

        except Exception as e:
            self.error = f"Listener error: {e}"
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
            self.running = False