"""
Alpha paradigm starter: read Muse LSL EEG, compute alpha/delta relaxation metric,
optionally toggle an Arduino LED via serial, and save every update to a CSV file.

Prerequisites:
  - BlueMuse + LSL bridge running so an EEG stream exists before you start this script.
  - Optional: Arduino sketch listening for b'1' (on) and b'0' (off) on the serial port.

Beginner notes:
  - Each CSV row is one "decision tick" after your buffer advances (see SHIFT_LENGTH).
  - The metric alpha_log / delta_log is the same idea as muse-lsl/examples/neurofeedback.py.
  - Tune THRESH_ON / THRESH_OFF after watching printed values for eyes open vs eyes closed.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, lfilter, lfilter_zi

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

# -----------------------------------------------------------------------------
# Minimal EEG helpers (adapted from muse-lsl/examples/utils.py)
# -----------------------------------------------------------------------------

NOTCH_B, NOTCH_A = butter(4, np.array([55, 65]) / (256 / 2), btype="bandstop")


def nextpow2(i: int) -> int:
    n = 1
    while n < i:
        n *= 2
    return n


def compute_band_powers(eegdata: np.ndarray, fs: float) -> np.ndarray:
    """Returns log10 band powers: [delta, theta, alpha, beta] per channel, concatenated."""
    win_sample_length, nb_ch = eegdata.shape
    w = np.hamming(win_sample_length)
    data_win_centered = eegdata - np.mean(eegdata, axis=0)
    data_win_centered_ham = (data_win_centered.T * w).T

    nfft = nextpow2(win_sample_length)
    y = np.fft.fft(data_win_centered_ham, n=nfft, axis=0) / win_sample_length
    psd = 2 * np.abs(y[0 : int(nfft / 2), :])
    f = fs / 2 * np.linspace(0, 1, int(nfft / 2))

    ind_delta, = np.where(f < 4)
    mean_delta = np.mean(psd[ind_delta, :], axis=0)
    ind_theta, = np.where((f >= 4) & (f <= 8))
    mean_theta = np.mean(psd[ind_theta, :], axis=0)
    ind_alpha, = np.where((f >= 8) & (f <= 12))
    mean_alpha = np.mean(psd[ind_alpha, :], axis=0)
    ind_beta, = np.where((f >= 12) & (f < 30))
    mean_beta = np.mean(psd[ind_beta, :], axis=0)

    feature_vector = np.concatenate(
        (mean_delta, mean_theta, mean_alpha, mean_beta), axis=0
    )
    return np.log10(feature_vector)


def update_buffer(data_buffer, new_data, notch=False, filter_state=None):
    if new_data.ndim == 1:
        new_data = new_data.reshape(-1, data_buffer.shape[1])

    if notch:
        if filter_state is None:
            filter_state = np.tile(
                lfilter_zi(NOTCH_B, NOTCH_A), (data_buffer.shape[1], 1)
            ).T
        new_data, filter_state = lfilter(
            NOTCH_B, NOTCH_A, new_data, axis=0, zi=filter_state
        )

    new_buffer = np.concatenate((data_buffer, new_data), axis=0)
    new_buffer = new_buffer[new_data.shape[0] :, :]
    return new_buffer, filter_state


def get_last_data(data_buffer, newest_samples: int) -> np.ndarray:
    return data_buffer[(data_buffer.shape[0] - newest_samples) :, :]


# -----------------------------------------------------------------------------
# Paradigm logic
# -----------------------------------------------------------------------------

BUFFER_LENGTH = 5.0
EPOCH_LENGTH = 1.0
OVERLAP_LENGTH = 0.8
# Muse channel order in many LSL streams: 0 TP9, 1 AF7, 2 AF8, 3 TP10
DEFAULT_CHANNELS = [0]


@dataclass
class BandIndices:
    n_ch: int

    def alpha_sl(self, ch: int) -> int:
        return 2 * self.n_ch + ch

    def delta_sl(self, ch: int) -> int:
        return ch


def smooth_vector(band_buffer: np.ndarray) -> np.ndarray:
    return np.mean(band_buffer, axis=0)


def parse_channels(spec: str) -> list[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha paradigm LSL + CSV + Arduino")
    parser.add_argument(
        "--serial-port",
        default="",
        help="COM port, e.g. COM3 on Windows. Empty = no Arduino, CSV only.",
    )
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--channels",
        default="0",
        help="Comma-separated 0-based channel indices to average (e.g. 0 or 1,2).",
    )
    parser.add_argument(
        "--thresh-on",
        type=float,
        default=None,
        help="LED turn-on threshold (meaning depends on --mode).",
    )
    parser.add_argument(
        "--thresh-off",
        type=float,
        default=None,
        help="LED turn-off threshold (meaning depends on --mode).",
    )
    parser.add_argument(
        "--mode",
        choices=["relax", "focus"],
        default="relax",
        help=(
            "relax: higher alpha_metric turns LED on. "
            "focus: lower alpha_metric turns LED on."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=["alpha_minus_delta", "beta_minus_alpha"],
        default="alpha_minus_delta",
        help=(
            "Signal used for thresholding. "
            "alpha_minus_delta is good for relax, beta_minus_alpha often better for focus."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("paradigm_logs"),
        help="Folder for CSV files (created if missing).",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Show a live matplotlib plot of metric and thresholds.",
    )
    parser.add_argument(
        "--dimming",
        action="store_true",
        help="Send PWM brightness (0..255) to Arduino on ~9 via 'L###' messages.",
    )
    parser.add_argument(
        "--activation-smoothing",
        type=float,
        default=0.15,
        help="EMA smoothing for activation [0..1]. Lower = smoother, slower response.",
    )
    parser.add_argument(
        "--max-brightness-step",
        type=int,
        default=4,
        help="Max PWM change (0..255) per update. Lower = smoother fades.",
    )
    parser.add_argument(
        "--brightness-gamma",
        type=float,
        default=2.2,
        help="Gamma curve for perceived brightness. >1 makes low levels smoother (common: 2.0-3.0).",
    )
    parser.add_argument(
        "--metric-smoothing",
        type=float,
        default=1.0,
        help=(
            "EMA smoothing for the alpha_metric used for threshold decisions. "
            "1.0 = no smoothing (raw). 0.2-0.4 usually reduces flicker."
        ),
    )
    args = parser.parse_args()

    channels = parse_channels(args.channels)
    if not channels:
        sys.exit("Provide at least one channel index.")

    use_led = args.thresh_on is not None and args.thresh_off is not None
    if use_led:
        if args.mode == "relax" and args.thresh_off >= args.thresh_on:
            sys.exit("In relax mode, thresh-off must be < thresh-on.")
        if args.mode == "focus":
            if args.metric == "beta_minus_alpha" and args.thresh_off >= args.thresh_on:
                sys.exit(
                    "In focus mode with beta_minus_alpha, thresh-off must be < thresh-on."
                )
            if args.metric == "alpha_minus_delta" and args.thresh_off <= args.thresh_on:
                sys.exit(
                    "In focus mode with alpha_minus_delta, thresh-off must be > thresh-on."
                )
    if args.activation_smoothing <= 0 or args.activation_smoothing > 1:
        sys.exit("--activation-smoothing must be in (0, 1].")
    if args.max_brightness_step < 1:
        sys.exit("--max-brightness-step must be >= 1.")
    if args.brightness_gamma <= 0:
        sys.exit("--brightness-gamma must be > 0.")
    if args.metric_smoothing <= 0 or args.metric_smoothing > 1.0:
        sys.exit("--metric-smoothing must be in (0, 1].")
    serial_out = None
    if args.serial_port:
        try:
            import serial
        except ImportError:
            sys.exit("Install pyserial: pip install pyserial")
        serial_out = serial.Serial(args.serial_port, args.baud, timeout=0.1)
        time.sleep(2)
        print(f"Opened {args.serial_port} at {args.baud} baud.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.out_dir / f"alpha_paradigm_{stamp}.csv"

    print("Looking for an EEG stream (type=EEG)...")
    streams = resolve_byprop("type", "EEG", timeout=5)
    if len(streams) == 0:
        sys.exit("No EEG stream found. Start BlueMuse + LSL bridge first.")

    inlet = StreamInlet(streams[0], max_chunklen=12)
    inlet.time_correction()
    info = inlet.info()
    fs = int(info.nominal_srate())
    n_lsl_ch = info.channel_count()

    for c in channels:
        if c < 0 or c >= n_lsl_ch:
            sys.exit(f"Channel {c} out of range (stream has {n_lsl_ch} channels).")

    shift_length = EPOCH_LENGTH - OVERLAP_LENGTH
    n_win = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) / shift_length + 1))

    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), len(channels)))
    filter_state = None
    band_buffer = np.zeros((n_win, 4 * len(channels)))
    bi = BandIndices(len(channels))

    fieldnames = [
        "unix_time",
        "alpha_metric",
        "led_on",
        "led_pwm",
    ]
    for ci, ch in enumerate(channels):
        fieldnames += [
            f"delta_ch{ch}",
            f"theta_ch{ch}",
            f"alpha_ch{ch}",
            f"beta_ch{ch}",
        ]

    led_state = False
    activation_ema = 0.0
    metric_ema = None
    smooth_brightness = 0
    if use_led and serial_out:
        serial_out.write(b"0")
        led_state = False

    plt = None
    line_metric = None
    line_on = None
    line_off = None
    ax = None
    history = deque(maxlen=300)
    if args.dashboard:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            sys.exit("Install matplotlib for --dashboard: pip install matplotlib")
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 4))
        (line_metric,) = ax.plot([], [], label="metric")
        (line_on,) = ax.plot([], [], "--", label="thresh_on")
        (line_off,) = ax.plot([], [], "--", label="thresh_off")
        ax.set_title("EEG live metric")
        ax.set_xlabel("samples")
        ax.set_ylabel("value")
        ax.legend(loc="upper right")

    print(f"Writing: {csv_path.resolve()}")
    print("Ctrl+C to stop.\n")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        try:
            while True:
                eeg_data, _ = inlet.pull_chunk(
                    timeout=1, max_samples=int(shift_length * fs)
                )
                if not eeg_data:
                    continue

                raw = np.array(eeg_data)[:, channels]
                eeg_buffer, filter_state = update_buffer(
                    eeg_buffer, raw, notch=True, filter_state=filter_state
                )

                data_epoch = get_last_data(eeg_buffer, int(EPOCH_LENGTH * fs))
                band_powers = compute_band_powers(data_epoch, fs)
                band_buffer, _ = update_buffer(
                    band_buffer,
                    np.asarray([band_powers]),
                    notch=False,
                )
                smooth = smooth_vector(band_buffer)

                # Build a stable per-channel metric from log band powers.
                # Using differences is less likely to explode than dividing by tiny values.
                metrics = []
                row = {"unix_time": time.time()}
                for i, ch in enumerate(channels):
                    d = smooth[bi.delta_sl(i)]
                    a = smooth[bi.alpha_sl(i)]
                    b = smooth[3 * len(channels) + i]
                    if args.metric == "alpha_minus_delta":
                        m = a - d
                    else:
                        m = b - a
                    metrics.append(m if np.isfinite(m) else np.nan)
                    row[f"delta_ch{ch}"] = float(d)
                    row[f"theta_ch{ch}"] = float(smooth[1 * len(channels) + i])
                    row[f"alpha_ch{ch}"] = float(a)
                    row[f"beta_ch{ch}"] = float(b)

                alpha_metric = float(np.nanmean(metrics))
                row["alpha_metric"] = alpha_metric

                # Smooth the metric so threshold decisions are less jittery.
                if metric_ema is None:
                    metric_ema = alpha_metric
                else:
                    metric_ema = (1.0 - args.metric_smoothing) * metric_ema + args.metric_smoothing * alpha_metric
                metric_used = float(metric_ema)

                # Map metric into activation [0..1] inside your two thresholds.
                # This creates the "in-between brightness" zone you described.
                activation = 0.0
                if use_led:
                    if args.mode == "relax":
                        if not led_state and metric_used >= args.thresh_on:
                            led_state = True
                            if serial_out:
                                serial_out.write(b"1")
                        elif led_state and metric_used <= args.thresh_off:
                            led_state = False
                            if serial_out:
                                serial_out.write(b"0")
                    else:  # focus mode: invert decision direction
                        # For focus, threshold direction depends on the chosen metric.
                        # beta_minus_alpha: higher values usually indicate focus.
                        # alpha_minus_delta: lower values usually indicate focus.
                        if args.metric == "beta_minus_alpha":
                            if not led_state and metric_used >= args.thresh_on:
                                led_state = True
                                if serial_out:
                                    serial_out.write(b"1")
                            elif led_state and metric_used <= args.thresh_off:
                                led_state = False
                                if serial_out:
                                    serial_out.write(b"0")
                        else:
                            if not led_state and metric_used <= args.thresh_on:
                                led_state = True
                                if serial_out:
                                    serial_out.write(b"1")
                            elif led_state and metric_used >= args.thresh_off:
                                led_state = False
                                if serial_out:
                                    serial_out.write(b"0")

                    # Continuous activation for dynamic dimming (same thresholds)
                    if args.mode == "relax":
                        if metric_used <= args.thresh_off:
                            activation = 0.0
                        elif metric_used >= args.thresh_on:
                            activation = 1.0
                        else:
                            activation = (metric_used - args.thresh_off) / (
                                args.thresh_on - args.thresh_off
                            )
                    elif args.metric == "beta_minus_alpha":
                        # focus + beta_minus_alpha: higher => more activation
                        if metric_used <= args.thresh_off:
                            activation = 0.0
                        elif metric_used >= args.thresh_on:
                            activation = 1.0
                        else:
                            activation = (metric_used - args.thresh_off) / (
                                args.thresh_on - args.thresh_off
                            )
                    else:
                        # focus + alpha_minus_delta: lower => more activation
                        if metric_used >= args.thresh_off:
                            activation = 0.0
                        elif metric_used <= args.thresh_on:
                            activation = 1.0
                        else:
                            activation = (args.thresh_off - metric_used) / (
                                args.thresh_off - args.thresh_on
                            )
                    activation = clamp01(float(activation))
                row["led_on"] = int(led_state)

                # Smooth + rate-limit brightness so it doesn't flicker.
                activation_ema = (
                    (1.0 - args.activation_smoothing) * activation_ema
                    + args.activation_smoothing * activation
                )
                activation_ema = clamp01(float(activation_ema))
                shaped = pow(activation_ema, args.brightness_gamma)
                target_brightness = int(round(shaped * 255))
                delta = target_brightness - smooth_brightness
                step = min(abs(delta), args.max_brightness_step)
                if delta > 0:
                    smooth_brightness += step
                elif delta < 0:
                    smooth_brightness -= step
                brightness = int(smooth_brightness)
                row["led_pwm"] = brightness

                if use_led and args.dimming and serial_out:
                    serial_out.write(f"L{brightness}\n".encode("ascii"))

                writer.writerow(row)
                f.flush()

                if args.dashboard and plt is not None and line_metric is not None and ax is not None:
                    history.append(alpha_metric)
                    y = np.asarray(history, dtype=float)
                    x = np.arange(len(y))
                    line_metric.set_data(x, y)
                    if use_led and line_on is not None and line_off is not None:
                        line_on.set_data(x, np.full_like(y, args.thresh_on, dtype=float))
                        line_off.set_data(x, np.full_like(y, args.thresh_off, dtype=float))
                    y_min = np.nanmin(y) if y.size else -1.0
                    y_max = np.nanmax(y) if y.size else 1.0
                    if use_led:
                        y_min = min(y_min, args.thresh_on, args.thresh_off)
                        y_max = max(y_max, args.thresh_on, args.thresh_off)
                    pad = max(0.1, (y_max - y_min) * 0.2)
                    ax.set_xlim(0, max(50, len(y)))
                    ax.set_ylim(y_min - pad, y_max + pad)
                    plt.pause(0.001)

                print(
                    f"alpha_metric={alpha_metric:.4f}  led={led_state}  "
                    f"pwm={brightness:3d} "
                    f"(mode={args.mode}, metric={args.metric}, on={args.thresh_on}, off={args.thresh_off})"
                    if use_led
                    else f"alpha_metric={alpha_metric:.4f}  (set thresholds to drive LED)"
                )

        except KeyboardInterrupt:
            print("\nStopped. CSV saved.")

    if serial_out:
        serial_out.close()


if __name__ == "__main__":
    main()
