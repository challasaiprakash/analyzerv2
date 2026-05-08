import io
import re
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.fft import rfft, rfftfreq
from scipy.io import wavfile
from scipy.signal import fftconvolve, find_peaks, get_window, hilbert, spectrogram


st.set_page_config(page_title="Audio Analyzer", layout="wide")

PLOT_COLOR = "#00E6FF"
SECONDARY_COLOR = "#7DD3FC"
ENVELOPE_COLOR = "#F59E0B"
COMPARISON_COLORS = [
    "#00E6FF",
    "#F59E0B",
    "#22C55E",
    "#A855F7",
    "#EF4444",
    "#F97316",
    "#14B8A6",
    "#EAB308",
    "#EC4899",
    "#84CC16",
    "#38BDF8",
    "#F43F5E",
    "#8B5CF6",
    "#10B981",
    "#FACC15",
    "#FB7185",
    "#2DD4BF",
    "#C084FC",
    "#A3E635",
    "#60A5FA",
]
MAX_PLOT_SAMPLES = 30000
MAX_ANALYSIS_SAMPLES = 65536
MAX_IMAGE_SAMPLES = 16384
IMAGE_GRAPH_TYPES = {"Spectrogram", "Compressed spectrogram", "Scalogram"}
SINGLE_GRAPH_OPTIONS = [
    "Amplitude envelope",
    "Frequency domain",
    "Cepstrum",
    "Autocorrelation",
    "Spectrogram",
    "Compressed spectrogram",
    "Scalogram",
]
MULTI_GRAPH_OPTIONS = [
    "Amplitude envelope",
    "Waveform",
    "Frequency domain",
]


def downsample_signal(signal: np.ndarray, sample_rate: float, max_samples: int):
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        return signal, sample_rate
    step = max(1, int(np.ceil(signal.size / float(max_samples))))
    return signal[::step], sample_rate / step


def normalize_audio(signal: np.ndarray):
    signal = np.asarray(signal)
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    if np.issubdtype(signal.dtype, np.integer):
        max_val = max(abs(np.iinfo(signal.dtype).min), np.iinfo(signal.dtype).max)
        signal = signal.astype(np.float32) / float(max_val)
    else:
        signal = signal.astype(np.float32)
    return np.clip(signal, -1.0, 1.0)


def compute_envelope(signal: np.ndarray):
    if signal.size == 0:
        return signal
    try:
        return np.abs(hilbert(signal))
    except Exception:
        return np.abs(signal)


def segment_metrics(signal: np.ndarray, sample_rate: float):
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        return None

    peaks, _ = find_peaks(signal, distance=max(1, int(sample_rate * 0.01)))
    env = compute_envelope(signal)
    zc = np.sum(np.abs(np.diff(np.signbit(signal).astype(np.int8))))
    abs_max = float(np.max(np.abs(signal))) if signal.size else 0.0
    clip_frac = float(np.sum(np.abs(signal) >= 0.99 * abs_max)) / float(signal.size) if abs_max > 0 else 0.0

    return {
        "Duration (s)": float(signal.size) / float(sample_rate),
        "Samples": int(signal.size),
        "Max": float(np.max(signal)),
        "Min": float(np.min(signal)),
        "Average": float(np.mean(signal)),
        "RMS": float(np.sqrt(np.mean(signal**2))),
        "Peak-to-peak": float(np.max(signal) - np.min(signal)),
        "Peaks": int(len(peaks)),
        "Zero-crossing rate": float(zc) * (sample_rate / max(1.0, float(signal.size))),
        "Clipping fraction": clip_frac,
        "Envelope mean": float(np.mean(env)),
        "Envelope std": float(np.std(env)),
    }


def peak_frequency(signal: np.ndarray, sample_rate: float):
    if signal.size < 1024:
        return None
    nfft = min(65536, max(1024, signal.size))
    win = get_window("hann", nfft)
    sig_trim = signal[:nfft] if signal.size >= nfft else np.pad(signal, (0, nfft - signal.size))
    fft_vals = np.abs(rfft(sig_trim * win))
    freqs = rfftfreq(nfft, 1.0 / sample_rate)
    if fft_vals.size <= 1:
        return None
    idx = np.argmax(fft_vals[1:]) + 1
    return float(freqs[idx])


def snr_db(signal: np.ndarray):
    if signal.size == 0:
        return None
    sig_pow = np.var(signal)
    max_abs = np.max(np.abs(signal))
    if max_abs <= 0:
        return None
    noise_mask = np.abs(signal) < (0.1 * max_abs)
    if np.any(noise_mask):
        noise_pow = np.var(signal[noise_mask])
    else:
        mad = np.median(np.abs(signal - np.median(signal)))
        noise_pow = (mad**2) + 1e-12
    noise_pow = max(noise_pow, 1e-12)
    return float(10.0 * np.log10(max(sig_pow, 1e-12) / noise_pow))


def base_plotly_layout(title: str, height: int = 360):
    return dict(
        title=title,
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="#050509",
        plot_bgcolor="#020617",
        font=dict(color="#E5E7EB"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
        ),
    )


def comparison_color(index: int):
    if index < len(COMPARISON_COLORS):
        return COMPARISON_COLORS[index]
    hue = (index * 137.508) % 360.0
    return f"hsl({hue:.1f}, 80%, 58%)"


def safe_filename(value: str):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "graph"


def plotly_figure_bytes(fig):
    return fig.to_html(include_plotlyjs=True, full_html=True).encode("utf-8")


def png_buffer_bytes(buffer: io.BytesIO):
    return buffer.getvalue()


def graph_artifact(filename: str, data: bytes):
    return {"filename": filename, "data": data}


def graph_zip_bytes(artifacts):
    buffer = io.BytesIO()
    used_names = set()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for artifact in artifacts:
            filename = artifact["filename"]
            stem, dot, suffix = filename.rpartition(".")
            if not dot:
                stem, suffix = filename, ""
            candidate = filename
            counter = 2
            while candidate in used_names:
                candidate = f"{stem}_{counter}.{suffix}" if suffix else f"{stem}_{counter}"
                counter += 1
            used_names.add(candidate)
            archive.writestr(candidate, artifact["data"])
    buffer.seek(0)
    return buffer.getvalue()


def render_download_button(label: str, artifact, key: str):
    file_ext = artifact["filename"].rsplit(".", 1)[-1].lower()
    mime = "image/png" if file_ext == "png" else "text/html"
    st.download_button(
        label,
        data=artifact["data"],
        file_name=artifact["filename"],
        mime=mime,
        key=key,
        use_container_width=True,
    )


def waveform_figure(signal: np.ndarray, sample_rate: float, title: str, color: str = PLOT_COLOR):
    plot_signal, plot_sr = downsample_signal(signal, sample_rate, MAX_PLOT_SAMPLES)
    t = np.arange(plot_signal.size, dtype=np.float64) / float(plot_sr)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=t, y=plot_signal, mode="lines", line=dict(color=color, width=1)))
    fig.update_layout(**base_plotly_layout(title))
    fig.update_xaxes(title="Time (s)", gridcolor="#1F2937")
    fig.update_yaxes(title="Amplitude", gridcolor="#1F2937")
    return fig


def frequency_figure(signal: np.ndarray, sample_rate: float):
    plot_signal, plot_sr = downsample_signal(signal, sample_rate, MAX_ANALYSIS_SAMPLES)
    n = plot_signal.size
    win = get_window("hann", n)
    fft_vals = np.abs(rfft(plot_signal * win))
    freqs = rfftfreq(n, 1.0 / plot_sr)
    y = 20.0 * np.log10(fft_vals + 1e-12)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=freqs[1:], y=y[1:], mode="lines", line=dict(color=PLOT_COLOR, width=1)))
    fig.update_layout(**base_plotly_layout("Frequency Domain (dB)"))
    fig.update_xaxes(title="Frequency (Hz)", gridcolor="#1F2937", type="log")
    fig.update_yaxes(title="Magnitude (dB)", gridcolor="#1F2937")
    return fig


def cepstrum_figure(signal: np.ndarray, sample_rate: float):
    plot_signal, plot_sr = downsample_signal(signal, sample_rate, MAX_ANALYSIS_SAMPLES)
    win = get_window("hann", plot_signal.size)
    segw = plot_signal * win
    cepstrum = np.fft.irfft(np.log(np.abs(rfft(segw)) + 1e-12))
    q = np.arange(len(cepstrum), dtype=np.float64) / float(plot_sr)
    max_q = min(len(cepstrum), max(32, int(0.05 * plot_sr)))
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=q[:max_q], y=cepstrum[:max_q], mode="lines", line=dict(color=SECONDARY_COLOR, width=1)))
    fig.update_layout(**base_plotly_layout("Real Cepstrum"))
    fig.update_xaxes(title="Quefrency (s)", gridcolor="#1F2937")
    fig.update_yaxes(title="Amplitude", gridcolor="#1F2937")
    return fig


def autocorr_figure(signal: np.ndarray, sample_rate: float):
    plot_signal, plot_sr = downsample_signal(signal, sample_rate, MAX_ANALYSIS_SAMPLES)
    centered = plot_signal - np.mean(plot_signal)
    autocorr = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    norm = autocorr[0] if autocorr.size > 0 and autocorr[0] > 0 else 1.0
    autocorr = autocorr / norm
    lag_t = np.arange(len(autocorr), dtype=np.float64) / float(plot_sr)
    max_lag = min(len(autocorr), max(64, int(0.5 * plot_sr)))
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=lag_t[:max_lag], y=autocorr[:max_lag], mode="lines", line=dict(color=SECONDARY_COLOR, width=1)))
    fig.update_layout(**base_plotly_layout("Autocorrelation"))
    fig.update_xaxes(title="Lag (s)", gridcolor="#1F2937")
    fig.update_yaxes(title="Correlation", gridcolor="#1F2937")
    return fig


def envelope_figure(signal: np.ndarray, sample_rate: float, title: str = "Amplitude Envelope", color: str = ENVELOPE_COLOR):
    plot_signal, plot_sr = downsample_signal(signal, sample_rate, MAX_PLOT_SAMPLES)
    envelope = compute_envelope(plot_signal)
    t = np.arange(plot_signal.size, dtype=np.float64) / float(plot_sr)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=t, y=envelope, mode="lines", line=dict(color=color, width=1.2)))
    fig.update_layout(**base_plotly_layout(title))
    fig.update_xaxes(title="Time (s)", gridcolor="#1F2937")
    fig.update_yaxes(title="Amplitude", gridcolor="#1F2937")
    return fig


def comparison_figure(items, graph_type: str):
    fig = go.Figure()
    for idx, item in enumerate(items):
        signal = item["signal"]
        sample_rate = item["sample_rate"]
        color = item.get("color", comparison_color(idx))
        label = item["name"]

        if graph_type == "Amplitude envelope":
            plot_signal, plot_sr = downsample_signal(signal, sample_rate, MAX_PLOT_SAMPLES)
            y = compute_envelope(plot_signal)
            x = np.arange(plot_signal.size, dtype=np.float64) / float(plot_sr)
            y_title = "Amplitude"
            title = "Envelope Comparison"
            x_type = "linear"
        elif graph_type == "Waveform":
            plot_signal, plot_sr = downsample_signal(signal, sample_rate, MAX_PLOT_SAMPLES)
            y = plot_signal
            x = np.arange(plot_signal.size, dtype=np.float64) / float(plot_sr)
            y_title = "Amplitude"
            title = "Waveform Comparison"
            x_type = "linear"
        else:
            plot_signal, plot_sr = downsample_signal(signal, sample_rate, MAX_ANALYSIS_SAMPLES)
            win = get_window("hann", plot_signal.size)
            fft_vals = np.abs(rfft(plot_signal * win))
            x = rfftfreq(plot_signal.size, 1.0 / plot_sr)[1:]
            y = 20.0 * np.log10(fft_vals + 1e-12)[1:]
            y_title = "Magnitude (dB)"
            title = "Frequency Comparison"
            x_type = "log"

        fig.add_trace(go.Scattergl(x=x, y=y, mode="lines", name=label, line=dict(color=color, width=1.2)))

    fig.update_layout(**base_plotly_layout(title, height=420))
    fig.update_xaxes(title="Time (s)" if graph_type != "Frequency domain" else "Frequency (Hz)", gridcolor="#1F2937", type=x_type)
    fig.update_yaxes(title=y_title, gridcolor="#1F2937")
    return fig


def metrics_overview_cards(item):
    metrics = item["metrics"] or {}
    cols = st.columns(4)
    cols[0].metric("Min", f"{metrics.get('Min', 0.0):.4f}")
    cols[1].metric("Max", f"{metrics.get('Max', 0.0):.4f}")
    cols[2].metric("Average", f"{metrics.get('Average', 0.0):.4f}")
    cols[3].metric("Peak freq", f"{item['peak_frequency']:.2f} Hz" if item["peak_frequency"] is not None else "N/A")


def figure_to_png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return buffer


@st.cache_data(show_spinner=False)
def load_wav_bytes(audio_bytes: bytes):
    sample_rate, signal = wavfile.read(io.BytesIO(audio_bytes))
    return normalize_audio(signal), int(sample_rate)


@st.cache_data(show_spinner=False)
def cached_metrics(signal: np.ndarray, sample_rate: int):
    metrics = segment_metrics(signal, sample_rate)
    return metrics, peak_frequency(signal, sample_rate), snr_db(signal)


@st.cache_data(show_spinner=False)
def spectrogram_png(signal: np.ndarray, sample_rate: int):
    plot_signal, plot_sr = downsample_signal(signal, sample_rate, MAX_IMAGE_SAMPLES)
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#050509")
    f, t_spec, sxx = spectrogram(plot_signal, fs=plot_sr, window="hann", nperseg=1024, noverlap=768, mode="magnitude")
    mesh = ax.pcolormesh(t_spec, f, 20.0 * np.log10(sxx + 1e-12), shading="auto", cmap="inferno")
    ax.set_title("Spectrogram", color="#E5E7EB")
    ax.set_xlabel("Time (s)", color="#E5E7EB")
    ax.set_ylabel("Frequency (Hz)", color="#E5E7EB")
    ax.set_facecolor("#020617")
    ax.tick_params(colors="#E5E7EB")
    fig.colorbar(mesh, ax=ax)
    return figure_to_png(fig)


@st.cache_data(show_spinner=False)
def pseudo_mel_png(signal: np.ndarray, sample_rate: int):
    plot_signal, plot_sr = downsample_signal(signal, sample_rate, MAX_IMAGE_SAMPLES)
    f, t_spec, sxx = spectrogram(plot_signal, fs=plot_sr, window="hann", nperseg=1024, noverlap=768, mode="magnitude")
    mel_like = np.log1p(sxx)
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#050509")
    mesh = ax.pcolormesh(t_spec, np.log10(np.maximum(f, 1.0)), 20.0 * np.log10(mel_like + 1e-12), shading="auto", cmap="inferno")
    ax.set_title("Compressed Spectrogram", color="#E5E7EB")
    ax.set_xlabel("Time (s)", color="#E5E7EB")
    ax.set_ylabel("log10(Frequency)", color="#E5E7EB")
    ax.set_facecolor("#020617")
    ax.tick_params(colors="#E5E7EB")
    fig.colorbar(mesh, ax=ax)
    return figure_to_png(fig)


@st.cache_data(show_spinner=False)
def scalogram_png(signal: np.ndarray, sample_rate: int):
    plot_signal, plot_sr = downsample_signal(signal, sample_rate, MAX_IMAGE_SAMPLES)
    widths = np.unique(np.geomspace(2, max(4, min(128, len(plot_signal) // 8)), num=32).astype(int))
    widths = widths[widths >= 2]
    coeffs = []
    for width in widths:
        half_span = max(8, int(8 * width))
        x = np.arange(-half_span, half_span + 1, dtype=np.float64)
        sigma2 = float(width) ** 2
        wavelet = (1.0 - (x**2) / sigma2) * np.exp(-(x**2) / (2.0 * sigma2))
        wavelet -= np.mean(wavelet)
        norm = np.sqrt(np.sum(wavelet**2))
        if norm > 0:
            wavelet /= norm
        conv = fftconvolve(plot_signal, wavelet[::-1], mode="same")
        coeffs.append(np.abs(conv))
    coeffs = np.asarray(coeffs, dtype=np.float32)
    pseudo_freq = plot_sr / np.maximum(2.0 * np.pi * widths.astype(np.float64), 1.0)
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#050509")
    mesh = ax.pcolormesh(
        np.arange(coeffs.shape[1], dtype=np.float64) / float(plot_sr),
        pseudo_freq,
        np.abs(coeffs),
        shading="auto",
        cmap="inferno",
    )
    ax.set_title("Scalogram", color="#E5E7EB")
    ax.set_xlabel("Time (s)", color="#E5E7EB")
    ax.set_ylabel("Pseudo-frequency (Hz)", color="#E5E7EB")
    ax.set_facecolor("#020617")
    ax.tick_params(colors="#E5E7EB")
    fig.colorbar(mesh, ax=ax)
    return figure_to_png(fig)


def metrics_dataframe(metrics: dict, peak: float | None, snr: float | None):
    rows = [{"Metric": key, "Value": value} for key, value in metrics.items()]
    if peak is not None:
        rows.append({"Metric": "Peak frequency (Hz)", "Value": peak})
    if snr is not None:
        rows.append({"Metric": "SNR (dB)", "Value": snr})
    frame = pd.DataFrame(rows)
    frame["Value"] = frame["Value"].map(lambda value: f"{value:.4f}" if isinstance(value, float) else value)
    return frame


def summary_statistics_table(metrics_frame: pd.DataFrame):
    numeric = metrics_frame.select_dtypes(include=[np.number])
    summary = pd.DataFrame(
        {
            "Mean": numeric.mean(axis=0),
            "Avg": numeric.mean(axis=0),
            "Min": numeric.min(axis=0),
            "Max": numeric.max(axis=0),
        }
    )
    summary.index.name = "Metric"
    return summary.reset_index()


def selected_segment(signal: np.ndarray, sample_rate: int, start_s: float, end_s: float):
    i0 = max(0, int(start_s * sample_rate))
    i1 = min(signal.size, int(end_s * sample_rate))
    segment = signal[i0:i1] if i1 > i0 else signal[: min(signal.size, 2048)]
    return np.asarray(segment, dtype=np.float32)


def graph_filename(prefix: str, graph_type: str, extension: str):
    return f"{safe_filename(prefix)}_{safe_filename(graph_type.lower())}.{extension}"


def selected_graph_artifact(graph_type: str, signal: np.ndarray, sample_rate: int, filename_prefix: str):
    if graph_type == "Amplitude envelope":
        fig = envelope_figure(signal, sample_rate)
        return fig, graph_artifact(graph_filename(filename_prefix, graph_type, "html"), plotly_figure_bytes(fig))
    if graph_type == "Waveform":
        fig = waveform_figure(signal, sample_rate, "Waveform", color=SECONDARY_COLOR)
        return fig, graph_artifact(graph_filename(filename_prefix, graph_type, "html"), plotly_figure_bytes(fig))
    if graph_type == "Frequency domain":
        fig = frequency_figure(signal, sample_rate)
        return fig, graph_artifact(graph_filename(filename_prefix, graph_type, "html"), plotly_figure_bytes(fig))
    if graph_type == "Cepstrum":
        fig = cepstrum_figure(signal, sample_rate)
        return fig, graph_artifact(graph_filename(filename_prefix, graph_type, "html"), plotly_figure_bytes(fig))
    if graph_type == "Autocorrelation":
        fig = autocorr_figure(signal, sample_rate)
        return fig, graph_artifact(graph_filename(filename_prefix, graph_type, "html"), plotly_figure_bytes(fig))
    if graph_type == "Spectrogram":
        buffer = spectrogram_png(signal, sample_rate)
        return buffer, graph_artifact(graph_filename(filename_prefix, graph_type, "png"), png_buffer_bytes(buffer))
    if graph_type == "Compressed spectrogram":
        buffer = pseudo_mel_png(signal, sample_rate)
        return buffer, graph_artifact(graph_filename(filename_prefix, graph_type, "png"), png_buffer_bytes(buffer))
    if graph_type == "Scalogram":
        buffer = scalogram_png(signal, sample_rate)
        return buffer, graph_artifact(graph_filename(filename_prefix, graph_type, "png"), png_buffer_bytes(buffer))
    return None, None


def render_plotly_graph(fig, artifact, key: str):
    render_download_button("Download graph", artifact, key=f"{key}_download")
    st.plotly_chart(fig, use_container_width=True)


def render_image_graph(buffer, artifact, key: str):
    render_download_button("Download graph", artifact, key=f"{key}_download")
    st.image(buffer, use_container_width=True)


def render_selected_graph(graph_type: str, signal: np.ndarray, sample_rate: int, key_prefix: str, filename_prefix: str):
    graph, artifact = selected_graph_artifact(graph_type, signal, sample_rate, filename_prefix)
    if graph is None or artifact is None:
        return []
    if graph_type in IMAGE_GRAPH_TYPES:
        render_image_graph(graph, artifact, key_prefix)
    else:
        render_plotly_graph(graph, artifact, key_prefix)
    return [artifact]


def render_waveform_graph(signal: np.ndarray, sample_rate: int, title: str, color: str, key_prefix: str, filename_prefix: str):
    fig = waveform_figure(signal, sample_rate, title, color=color)
    artifact = graph_artifact(graph_filename(filename_prefix, title, "html"), plotly_figure_bytes(fig))
    render_plotly_graph(fig, artifact, key_prefix)
    return artifact


def render_comparison_graph(graph_type: str, graph_items, key_prefix: str, filename_prefix: str):
    fig = comparison_figure(graph_items, graph_type)
    artifact = graph_artifact(graph_filename(filename_prefix, graph_type, "html"), plotly_figure_bytes(fig))
    render_plotly_graph(fig, artifact, key_prefix)
    return [artifact]


def render_multi_graph_output(graph_type: str, graph_items, key_prefix: str, filename_prefix: str):
    if graph_type in {"Amplitude envelope", "Waveform", "Frequency domain"}:
        return render_comparison_graph(graph_type, graph_items, key_prefix, filename_prefix)

    artifacts = []
    if len(graph_items) > 1:
        image_columns = st.columns(min(2, len(graph_items)), gap="large")
        for index, item in enumerate(graph_items):
            with image_columns[index % len(image_columns)]:
                st.caption(item["name"])
                artifacts.extend(
                    render_selected_graph(
                        graph_type,
                        item["signal"],
                        item["sample_rate"],
                        key_prefix=f"{key_prefix}_{index}",
                        filename_prefix=f"{filename_prefix}_{item['name']}",
                    )
                )
    else:
        item = graph_items[0]
        artifacts.extend(
            render_selected_graph(
                graph_type,
                item["signal"],
                item["sample_rate"],
                key_prefix=key_prefix,
                filename_prefix=f"{filename_prefix}_{item['name']}",
            )
        )
    return artifacts


def render_single_audio_tab():
    uploaded = st.file_uploader("Browse one WAV audio file", type=["wav"], key="single_audio")
    if uploaded is None:
        st.info("Upload one WAV file to start single-audio analysis.")
        return

    audio_bytes = uploaded.getvalue()
    with st.spinner("Loading WAV audio..."):
        signal, sample_rate = load_wav_bytes(audio_bytes)

    duration = float(signal.size) / float(sample_rate) if sample_rate else 0.0

    top_left, top_right = st.columns([1.2, 1.8], gap="large")
    with top_left:
        st.audio(audio_bytes, format="audio/wav")
        mode = st.radio("Analysis range", ["Selected segment", "Full audio"], horizontal=True)
        if mode == "Full audio" or duration <= 0:
            start_s = 0.0
            end_s = duration
        else:
            default_end = min(duration, max(0.3, duration * 0.15))
            start_s, end_s = st.slider(
                "Segment range (seconds)",
                min_value=0.0,
                max_value=float(duration),
                value=(0.0, float(default_end)),
            )
        graph_type = st.selectbox("Graph type", SINGLE_GRAPH_OPTIONS, key="single_graph_type")
        single_download_slot = st.empty()

    segment = selected_segment(signal, sample_rate, start_s, end_s)

    with st.spinner("Computing statistics..."):
        whole_metrics, whole_peak, whole_snr = cached_metrics(signal, sample_rate)
        segment_metrics_data, segment_peak, segment_snr = cached_metrics(segment, sample_rate)

    with top_right:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Duration", f"{duration:.2f} s")
        metric_cols[1].metric("Sample rate", f"{sample_rate} Hz")
        metric_cols[2].metric("Peak frequency", f"{whole_peak:.2f} Hz" if whole_peak is not None else "N/A")
        metric_cols[3].metric("SNR", f"{whole_snr:.2f} dB" if whole_snr is not None else "N/A")

        table_left, table_right = st.columns(2, gap="large")
        with table_left:
            st.subheader("Whole audio statistics")
            st.dataframe(metrics_dataframe(whole_metrics, whole_peak, whole_snr), use_container_width=True, hide_index=True)
        with table_right:
            st.subheader("Selected range statistics")
            st.dataframe(metrics_dataframe(segment_metrics_data, segment_peak, segment_snr), use_container_width=True, hide_index=True)

    waveform_left, waveform_right = st.columns(2, gap="large")
    current_graph_artifacts = []
    with waveform_left:
        current_graph_artifacts.append(
            render_waveform_graph(
                signal,
                sample_rate,
                "Full Waveform",
                PLOT_COLOR,
                "single_full_waveform",
                f"{uploaded.name}_full",
            )
        )
    with waveform_right:
        current_graph_artifacts.append(
            render_waveform_graph(
                segment,
                sample_rate,
                "Selected Segment",
                SECONDARY_COLOR,
                "single_selected_segment",
                f"{uploaded.name}_selected",
            )
        )

    current_graph_artifacts.extend(
        render_selected_graph(graph_type, segment, sample_rate, "single_selected_graph", f"{uploaded.name}_analysis")
    )
    single_download_slot.download_button(
        "Download all current graphs",
        data=graph_zip_bytes(current_graph_artifacts),
        file_name=f"{safe_filename(uploaded.name)}_current_graphs.zip",
        mime="application/zip",
        key="single_download_all_current_graphs",
        use_container_width=True,
    )


def render_multi_audio_tab():
    uploaded_files = st.file_uploader("Browse multiple WAV audio files", type=["wav"], accept_multiple_files=True, key="multi_audio")
    if not uploaded_files:
        st.info("Upload two or more WAV files to compare envelope, waveform, frequency, and statistics.")
        return

    with st.spinner("Loading WAV audio files..."):
        audio_items = []
        for index, uploaded in enumerate(uploaded_files):
            audio_bytes = uploaded.getvalue()
            signal, sample_rate = load_wav_bytes(audio_bytes)
            metrics, peak, snr = cached_metrics(signal, sample_rate)
            duration = float(signal.size) / float(sample_rate) if sample_rate else 0.0
            audio_items.append(
                {
                    "name": uploaded.name,
                    "audio_bytes": audio_bytes,
                    "signal": signal,
                    "sample_rate": sample_rate,
                    "metrics": metrics,
                    "peak_frequency": peak,
                    "snr": snr,
                    "duration": duration,
                    "color": comparison_color(index),
                }
            )

    comparison_rows = []
    for item in audio_items:
        metrics = item["metrics"] or {}
        comparison_rows.append(
            {
                "File": item["name"],
                "Sample rate (Hz)": item["sample_rate"],
                "Duration (s)": item["duration"],
                "Average": metrics.get("Average"),
                "Min": metrics.get("Min"),
                "Max": metrics.get("Max"),
                "RMS": metrics.get("RMS"),
                "Envelope mean": metrics.get("Envelope mean"),
                "Peak frequency (Hz)": item["peak_frequency"],
                "SNR (dB)": item["snr"],
            }
        )

    comparison_frame = pd.DataFrame(comparison_rows)

    stat_metrics = st.columns(4)
    stat_metrics[0].metric("Files", str(len(audio_items)))
    stat_metrics[1].metric("Longest", f"{comparison_frame['Duration (s)'].max():.2f} s")
    stat_metrics[2].metric("Highest peak", f"{comparison_frame['Peak frequency (Hz)'].dropna().max():.2f} Hz" if comparison_frame["Peak frequency (Hz)"].notna().any() else "N/A")
    stat_metrics[3].metric("Best SNR", f"{comparison_frame['SNR (dB)'].dropna().max():.2f} dB" if comparison_frame["SNR (dB)"].notna().any() else "N/A")

    browser_col, stats_col = st.columns([1.1, 1.9], gap="large")
    with browser_col:
        st.subheader("Browsed audio files")
        for index, item in enumerate(audio_items):
            with st.expander(f"{index + 1}. {item['name']}", expanded=index == 0):
                st.audio(item["audio_bytes"], format="audio/wav")
                st.caption(f"Duration: {item['duration']:.2f} s | Sample rate: {item['sample_rate']} Hz")
                metrics_overview_cards(item)
    with stats_col:
        table_left, table_right = st.columns(2, gap="large")
        with table_left:
            st.subheader("Per-file statistics")
            display_frame = comparison_frame.copy()
            numeric_cols = display_frame.select_dtypes(include=[np.number]).columns
            display_frame[numeric_cols] = display_frame[numeric_cols].round(4)
            st.dataframe(display_frame, use_container_width=True, hide_index=True)
        with table_right:
            st.subheader("Summary statistics")
            summary_frame = summary_statistics_table(comparison_frame.drop(columns=["File"]))
            summary_numeric = summary_frame.select_dtypes(include=[np.number]).columns
            summary_frame[summary_numeric] = summary_frame[summary_numeric].round(4)
            st.dataframe(summary_frame, use_container_width=True, hide_index=True)

    state_key = "multi_graph_panels"
    signature_key = "multi_graph_signature"
    current_signature = tuple(item["name"] for item in audio_items)
    if st.session_state.get(signature_key) != current_signature or state_key not in st.session_state:
        st.session_state[signature_key] = current_signature
        st.session_state[state_key] = [{"id": 1}]

    st.subheader("Comparison graphs")
    panels = st.session_state[state_key]
    all_names = [item["name"] for item in audio_items]
    current_graph_artifacts = []
    all_graph_download_slot = None

    for panel_index, panel in enumerate(panels):
        st.markdown(f"**Graph {panel_index + 1}**")

        is_first_panel = panel_index == 0
        if is_first_panel:
            graph_type = "Waveform"
            selected_names = all_names
            st.caption("Fixed graph: waveform comparison of all uploaded audio files.")
        else:
            graph_type = st.selectbox(
                "Graph type",
                ["Waveform"] + SINGLE_GRAPH_OPTIONS,
                key=f"panel_graph_type_{panel['id']}",
            )
            selected_names = st.multiselect(
                "Files for comparison",
                all_names,
                default=all_names,
                key=f"panel_files_{panel['id']}",
            )

        if not selected_names:
            st.info("Select at least one file for this graph.")
            continue

        selected_items = [item for item in audio_items if item["name"] in selected_names]
        range_mode = st.radio(
            "Analysis range",
            ["Full audio", "Selected segment"],
            horizontal=True,
            key=f"panel_range_{panel['id']}",
        )
        if all_graph_download_slot is None:
            all_graph_download_slot = st.empty()

        max_shared_duration = min(item["duration"] for item in selected_items)
        if range_mode == "Full audio" or max_shared_duration <= 0:
            start_s = 0.0
            end_s = max_shared_duration
        else:
            default_end = min(max_shared_duration, max(0.3, max_shared_duration * 0.15))
            start_s, end_s = st.slider(
                "Seconds",
                min_value=0.0,
                max_value=float(max_shared_duration),
                value=(0.0, float(default_end)),
                key=f"panel_slider_{panel['id']}",
            )

        graph_items = []
        panel_rows = []
        for selected_item in selected_items:
            plot_signal = (
                selected_item["signal"]
                if range_mode == "Full audio"
                else selected_segment(selected_item["signal"], selected_item["sample_rate"], start_s, end_s)
            )
            graph_items.append(
                {
                    "name": selected_item["name"],
                    "signal": plot_signal,
                    "sample_rate": selected_item["sample_rate"],
                    "color": selected_item["color"],
                }
            )
            panel_metrics, panel_peak, panel_snr = cached_metrics(plot_signal, selected_item["sample_rate"])
            panel_rows.append(
                {
                    "File": selected_item["name"],
                    "Start (s)": start_s,
                    "End (s)": end_s,
                    "Min": panel_metrics.get("Min"),
                    "Max": panel_metrics.get("Max"),
                    "Average": panel_metrics.get("Average"),
                    "Peak frequency (Hz)": panel_peak,
                    "SNR (dB)": panel_snr,
                }
            )

        current_graph_artifacts.extend(
            render_multi_graph_output(
                graph_type,
                graph_items,
                key_prefix=f"multi_graph_{panel['id']}",
                filename_prefix=f"graph_{panel_index + 1}",
            )
        )
        panel_stats = pd.DataFrame(panel_rows).round(4)
        st.dataframe(panel_stats, use_container_width=True, hide_index=True)

        if st.button("+ Add graph", key=f"add_multi_graph_{panel['id']}", use_container_width=True):
            next_id = max(existing_panel["id"] for existing_panel in st.session_state[state_key]) + 1
            st.session_state[state_key].append({"id": next_id})

        if panel_index < len(panels) - 1:
            st.divider()

    if all_graph_download_slot is not None and current_graph_artifacts:
        all_graph_download_slot.download_button(
            "Download all current graphs",
            data=graph_zip_bytes(current_graph_artifacts),
            file_name="current_comparison_graphs.zip",
            mime="application/zip",
            key="multi_download_all_current_graphs",
            use_container_width=True,
        )


st.title("Audio Analyzer")
st.caption("Single-file inspection and multi-file comparison for WAV audio.")

single_tab, multi_tab = st.tabs(["Single Audio", "Multiple Audio"])

with single_tab:
    render_single_audio_tab()

with multi_tab:
    render_multi_audio_tab()
