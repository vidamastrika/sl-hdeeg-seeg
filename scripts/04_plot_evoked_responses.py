"""Plot averaged EEG and SEEG cortico-cortical evoked responses."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "ccepcoreg"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

SUBJECT = "sub-05"
TASK = "ccepcoreg"
RUN = "run-03"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_modality(
    data_root: Path,
    subject: str,
    task: str,
    run: str,
    modality: str,
) -> dict:
    """Load one EEG or SEEG run and its metadata."""

    modality_dir = (
        data_root
        / "derivatives"
        / "epochs"
        / subject
        / modality
    )

    basename = f"{subject}_task-{task}_{run}"

    data_file = modality_dir / f"{basename}_epochs.npy"
    channels_file = modality_dir / f"{basename}_channels.tsv"
    events_file = modality_dir / f"{basename}_epochs.tsv"
    description_file = modality_dir / f"{basename}_epochs.json"

    required_files = [
        data_file,
        channels_file,
        events_file,
        description_file,
    ]

    missing_files = [
        path for path in required_files if not path.exists()
    ]

    if missing_files:
        missing_text = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            f"Missing files for {modality}:\n{missing_text}"
        )

    data = np.load(data_file)
    channels = pd.read_csv(channels_file, sep="\t")
    events = pd.read_csv(events_file, sep="\t")

    with description_file.open(encoding="utf-8") as file:
        description = json.load(file)

    if data.ndim != 3:
        raise ValueError(
            f"Expected three dimensions, found {data.shape}"
        )

    n_epochs, n_channels, n_times = data.shape

    if n_channels != len(channels):
        raise ValueError(
            f"{data_file.name} has {n_channels} channels, "
            f"but channels.tsv has {len(channels)} rows."
        )

    if n_epochs != len(events):
        raise ValueError(
            f"{data_file.name} has {n_epochs} epochs, "
            f"but epochs.tsv has {len(events)} rows."
        )

    sampling_rates = (
        channels["sampling_frequency"]
        .astype(float)
        .unique()
    )

    if len(sampling_rates) != 1:
        raise ValueError(
            f"Multiple sampling frequencies: {sampling_rates}"
        )

    sampling_rate = float(sampling_rates[0])

    if "zero_time" in events.columns:
        zero_times = (
            events["zero_time"]
            .astype(float)
            .unique()
        )

        if len(zero_times) != 1:
            raise ValueError(
                f"Multiple zero-time values: {zero_times}"
            )

        tmin = -float(zero_times[0])
    else:
        tmin = -0.3

    times = (
        np.arange(n_times, dtype=float) / sampling_rate
        + tmin
    )

    if "status" in channels.columns:
        status = (
            channels["status"]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        good_mask = status != "bad"
    else:
        good_mask = np.ones(n_channels, dtype=bool)

    good_mask = np.asarray(good_mask, dtype=bool)

    channel_names = (
        channels["name"]
        .astype(str)
        .to_numpy()
    )

    return {
        "data": data,
        "channels": channels,
        "events": events,
        "description": description,
        "times": times,
        "sampling_rate": sampling_rate,
        "good_mask": good_mask,
        "channel_names": channel_names,
    }


eeg = load_modality(
    DATA_ROOT,
    SUBJECT,
    TASK,
    RUN,
    modality="eeg",
)

seeg = load_modality(
    DATA_ROOT,
    SUBJECT,
    TASK,
    RUN,
    modality="ieeg",
)


# ---------------------------------------------------------------------
# Unit handling
# ---------------------------------------------------------------------

def get_display_scale(channels: pd.DataFrame) -> tuple[float, str, str]:
    """Choose a display scale using channels.tsv units."""

    unit_column = None

    for candidate in ("units", "unit"):
        if candidate in channels.columns:
            unit_column = candidate
            break

    if unit_column is None:
        return 1.0, "stored units", "not specified"

    stored_units = (
        channels[unit_column]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    if len(stored_units) != 1:
        return 1.0, "stored units", ", ".join(stored_units)

    stored_unit = stored_units[0]
    normalized_unit = stored_unit.lower()

    if normalized_unit in {"v", "volt", "volts"}:
        return 1e6, "µV", stored_unit

    if normalized_unit in {"mv", "millivolt", "millivolts"}:
        return 1e3, "µV", stored_unit

    if normalized_unit in {
        "uv",
        "µv",
        "μv",
        "microvolt",
        "microvolts",
    }:
        return 1.0, "µV", stored_unit

    return 1.0, stored_unit, stored_unit


eeg_scale, eeg_display_unit, eeg_stored_unit = get_display_scale(
    eeg["channels"]
)

seeg_scale, seeg_display_unit, seeg_stored_unit = get_display_scale(
    seeg["channels"]
)


# ---------------------------------------------------------------------
# Compute averaged responses
# ---------------------------------------------------------------------

eeg_good_data = eeg["data"][:, eeg["good_mask"], :]
seeg_good_data = seeg["data"][:, seeg["good_mask"], :]

eeg_good_names = eeg["channel_names"][eeg["good_mask"]]
seeg_good_names = seeg["channel_names"][seeg["good_mask"]]

eeg_evoked = eeg_good_data.mean(axis=0)
seeg_evoked = seeg_good_data.mean(axis=0)

eeg_evoked_display = eeg_evoked * eeg_scale
seeg_evoked_display = seeg_evoked * seeg_scale

eeg_spatial_rms = np.sqrt(
    np.mean(eeg_evoked_display**2, axis=0)
)

seeg_spatial_rms = np.sqrt(
    np.mean(seeg_evoked_display**2, axis=0)
)


# ---------------------------------------------------------------------
# Baseline and post-stimulation masks
# ---------------------------------------------------------------------

eeg_baseline_mask = (
    (eeg["times"] >= -0.3)
    & (eeg["times"] <= -0.01)
)

seeg_baseline_mask = (
    (seeg["times"] >= -0.3)
    & (seeg["times"] <= -0.01)
)

eeg_post_mask = (
    (eeg["times"] >= 0.01)
    & (eeg["times"] <= 0.30)
)

seeg_post_mask = (
    (seeg["times"] >= 0.01)
    & (seeg["times"] <= 0.30)
)


# ---------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------

def plot_butterfly(
    times: np.ndarray,
    evoked: np.ndarray,
    spatial_rms: np.ndarray,
    title: str,
    display_unit: str,
    output_file: Path,
) -> None:
    """Plot channel-level evoked responses and spatial RMS."""

    figure, axis = plt.subplots(
        figsize=(12, 7),
        constrained_layout=True,
    )

    axis.plot(
        times,
        evoked.T,
        color="#64748b",
        linewidth=0.5,
        alpha=0.35,
    )

    axis.plot(
        times,
        spatial_rms,
        color="#dc2626",
        linewidth=2.0,
        label="Spatial RMS",
    )

    axis.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Stimulation",
    )

    axis.axvspan(
        -0.005,
        0.010,
        color="#fbbf24",
        alpha=0.20,
        label="Potential stimulation artefact",
    )

    axis.set(
        title=title,
        xlabel="Time relative to stimulation (s)",
        ylabel=f"Amplitude ({display_unit})",
        xlim=(times[0], times[-1]),
    )

    axis.grid(
        alpha=0.20,
        linewidth=0.5,
    )

    axis.legend(
        loc="upper right",
        frameon=False,
    )

    figure.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


# ---------------------------------------------------------------------
# Figure 1: EEG averaged response
# ---------------------------------------------------------------------

eeg_figure_file = (
    FIGURE_DIR
    / f"{SUBJECT}_{RUN}_eeg_evoked.png"
)

plot_butterfly(
    times=eeg["times"],
    evoked=eeg_evoked_display,
    spatial_rms=eeg_spatial_rms,
    title=(
        f"{SUBJECT} {RUN}: averaged scalp EEG response "
        f"({len(eeg['data'])} epochs, "
        f"{len(eeg_good_names)} good channels)"
    ),
    display_unit=eeg_display_unit,
    output_file=eeg_figure_file,
)


# ---------------------------------------------------------------------
# Figure 2: SEEG averaged response
# ---------------------------------------------------------------------

seeg_figure_file = (
    FIGURE_DIR
    / f"{SUBJECT}_{RUN}_seeg_evoked.png"
)

plot_butterfly(
    times=seeg["times"],
    evoked=seeg_evoked_display,
    spatial_rms=seeg_spatial_rms,
    title=(
        f"{SUBJECT} {RUN}: averaged SEEG response "
        f"({len(seeg['data'])} epochs, "
        f"{len(seeg_good_names)} good channels)"
    ),
    display_unit=seeg_display_unit,
    output_file=seeg_figure_file,
)


# ---------------------------------------------------------------------
# Figure 3: normalized spatial RMS comparison
# ---------------------------------------------------------------------

eeg_rms_normalized = (
    eeg_spatial_rms
    / np.max(np.abs(eeg_spatial_rms))
)

seeg_rms_normalized = (
    seeg_spatial_rms
    / np.max(np.abs(seeg_spatial_rms))
)

comparison_figure_file = (
    FIGURE_DIR
    / f"{SUBJECT}_{RUN}_normalized_spatial_rms.png"
)

figure, axis = plt.subplots(
    figsize=(12, 6),
    constrained_layout=True,
)

axis.plot(
    eeg["times"],
    eeg_rms_normalized,
    color="#0284c7",
    linewidth=2,
    label="EEG",
)

axis.plot(
    seeg["times"],
    seeg_rms_normalized,
    color="#ca8a04",
    linewidth=2,
    label="SEEG",
)

axis.axvline(
    0,
    color="black",
    linestyle="--",
    linewidth=1.2,
)

axis.axvspan(
    -0.005,
    0.010,
    color="#fbbf24",
    alpha=0.20,
)

axis.set(
    title=f"{SUBJECT} {RUN}: normalized spatial RMS",
    xlabel="Time relative to stimulation (s)",
    ylabel="Normalized spatial RMS",
    xlim=(eeg["times"][0], eeg["times"][-1]),
)

axis.grid(
    alpha=0.20,
    linewidth=0.5,
)

axis.legend(
    frameon=False,
)

figure.savefig(
    comparison_figure_file,
    dpi=200,
    bbox_inches="tight",
)

plt.close(figure)


# ---------------------------------------------------------------------
# Quantitative summaries
# ---------------------------------------------------------------------

eeg_baseline_mean = np.mean(
    np.abs(eeg_evoked_display[:, eeg_baseline_mask])
)

seeg_baseline_mean = np.mean(
    np.abs(seeg_evoked_display[:, seeg_baseline_mask])
)

eeg_post_peak_per_channel = np.max(
    np.abs(eeg_evoked_display[:, eeg_post_mask]),
    axis=1,
)

seeg_post_peak_per_channel = np.max(
    np.abs(seeg_evoked_display[:, seeg_post_mask]),
    axis=1,
)

eeg_peak_index = int(
    np.argmax(eeg_post_peak_per_channel)
)

seeg_peak_index = int(
    np.argmax(seeg_post_peak_per_channel)
)

eeg_post_indices = np.flatnonzero(eeg_post_mask)
seeg_post_indices = np.flatnonzero(seeg_post_mask)

eeg_peak_sample_within_window = int(
    np.argmax(
        np.abs(
            eeg_evoked_display[
                eeg_peak_index,
                eeg_post_mask,
            ]
        )
    )
)

seeg_peak_sample_within_window = int(
    np.argmax(
        np.abs(
            seeg_evoked_display[
                seeg_peak_index,
                seeg_post_mask,
            ]
        )
    )
)

eeg_peak_time = eeg["times"][
    eeg_post_indices[eeg_peak_sample_within_window]
]

seeg_peak_time = seeg["times"][
    seeg_post_indices[seeg_peak_sample_within_window]
]

# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

print("AVERAGED CCEP RESPONSES")
print("-----------------------")
print(f"Subject: {SUBJECT}")
print(f"Run    : {RUN}")

print("\nEEG")
print("---")
print(f"Epochs              : {len(eeg['data'])}")
print(f"All channels        : {eeg['data'].shape[1]}")
print(f"Good channels       : {len(eeg_good_names)}")
print(f"Stored unit         : {eeg_stored_unit}")
print(f"Displayed unit      : {eeg_display_unit}")
print(
    f"Mean baseline |amp| : "
    f"{eeg_baseline_mean:.4f} {eeg_display_unit}"
)
print(
    f"Largest post peak   : "
    f"{eeg_good_names[eeg_peak_index]} "
    f"({eeg_post_peak_per_channel[eeg_peak_index]:.4f} "
    f"{eeg_display_unit} at "
    f"{eeg_peak_time * 1000:.1f} ms)"
)

print("\nSEEG")
print("----")
print(f"Epochs              : {len(seeg['data'])}")
print(f"All channels        : {seeg['data'].shape[1]}")
print(f"Good channels       : {len(seeg_good_names)}")
print(f"Stored unit         : {seeg_stored_unit}")
print(f"Displayed unit      : {seeg_display_unit}")
print(
    f"Mean baseline |amp| : "
    f"{seeg_baseline_mean:.4f} {seeg_display_unit}"
)
print(
    f"Largest post peak   : "
    f"{seeg_good_names[seeg_peak_index]} "
    f"({seeg_post_peak_per_channel[seeg_peak_index]:.4f} "
    f"{seeg_display_unit} at "
    f"{seeg_peak_time * 1000:.1f} ms)"
)

print("\nFIGURES")
print("-------")
print(f"EEG evoked      : {eeg_figure_file}")
print(f"SEEG evoked     : {seeg_figure_file}")
print(f"RMS comparison  : {comparison_figure_file}")

print("\nEvoked-response inspection complete.")