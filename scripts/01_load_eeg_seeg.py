"""Load one matched high-density EEG and SEEG stimulation run."""

import json
from pathlib import Path

import mne
import nibabel as nib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "ccepcoreg"

SUBJECT = "sub-05"
TASK = "ccepcoreg"
RUN = "run-03"


# ---------------------------------------------------------------------
# Loading function
# ---------------------------------------------------------------------

def load_epochs(
    data_root: Path,
    subject: str,
    task: str,
    run: str,
    modality: str,
) -> tuple[mne.EpochsArray, pd.DataFrame, dict]:
    """Load one EEG or SEEG epoch file and its metadata."""

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
            f"Missing files for {subject}, {run}, {modality}:\n"
            f"{missing_text}"
        )

    data = np.load(data_file)
    channels = pd.read_csv(channels_file, sep="\t")
    events = pd.read_csv(events_file, sep="\t")

    with description_file.open(encoding="utf-8") as file:
        description = json.load(file)

    if data.ndim != 3:
        raise ValueError(
            f"Expected a 3D array, but {data_file.name} has shape {data.shape}"
        )

    n_epochs, n_channels, _ = data.shape

    if n_channels != len(channels):
        raise ValueError(
            f"Array contains {n_channels} channels, but channels.tsv "
            f"contains {len(channels)} rows."
        )

    if n_epochs != len(events):
        raise ValueError(
            f"Array contains {n_epochs} epochs, but epochs.tsv "
            f"contains {len(events)} rows."
        )

    sampling_rates = channels["sampling_frequency"].astype(float).unique()

    if len(sampling_rates) != 1:
        raise ValueError(
            f"Multiple sampling frequencies found: {sampling_rates}"
        )

    sampling_rate = float(sampling_rates[0])

    if "zero_time" in events.columns:
        zero_times = events["zero_time"].astype(float).unique()

        if len(zero_times) != 1:
            raise ValueError(
                f"Multiple zero-time values found: {zero_times}"
            )

        tmin = -float(zero_times[0])
    else:
        tmin = -0.3

    channel_names = channels["name"].astype(str).tolist()
    channel_type = "eeg" if modality == "eeg" else "seeg"

    info = mne.create_info(
        ch_names=channel_names,
        sfreq=sampling_rate,
        ch_types=[channel_type] * n_channels,
    )

    epochs = mne.EpochsArray(
        data=data,
        info=info,
        tmin=tmin,
        metadata=events,
        verbose=False,
    )

    # The exported arrays were already baseline-corrected.
    # Record that information without correcting them a second time.
    if (
        description.get("BaselineCorrection") is True
        and description.get("BaselinePeriod") is not None
    ):
        baseline_period = description["BaselinePeriod"]
        epochs.baseline = tuple(float(value) for value in baseline_period)

    if "status" in channels.columns:
        epochs.info["bads"] = channels.loc[
            channels["status"].str.lower() == "bad",
            "name",
        ].astype(str).tolist()

    return epochs, events, description


# ---------------------------------------------------------------------
# Load matched EEG and SEEG run
# ---------------------------------------------------------------------

eeg_epochs, eeg_events, eeg_description = load_epochs(
    DATA_ROOT,
    SUBJECT,
    TASK,
    RUN,
    modality="eeg",
)

seeg_epochs, seeg_events, seeg_description = load_epochs(
    DATA_ROOT,
    SUBJECT,
    TASK,
    RUN,
    modality="ieeg",
)


# ---------------------------------------------------------------------
# Load participant MRI
# ---------------------------------------------------------------------

mri_file = DATA_ROOT / SUBJECT / "anat" / f"{SUBJECT}_T1w.nii"

if not mri_file.exists():
    raise FileNotFoundError(f"MRI not found: {mri_file}")

mri = nib.load(mri_file)


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

print("SELECTED RECORDING")
print("------------------")
print(f"Subject: {SUBJECT}")
print(f"Task   : {TASK}")
print(f"Run    : {RUN}")

print("\nEEG")
print("---")
print(eeg_epochs)
print(f"Array shape     : {eeg_epochs.get_data().shape}")
print(f"Sampling rate   : {eeg_epochs.info['sfreq']} Hz")
print(f"Time range      : {eeg_epochs.tmin} to {eeg_epochs.tmax} s")
print(f"Bad channels    : {len(eeg_epochs.info['bads'])}")
print(f"Baseline applied: {eeg_description.get('BaselineCorrection')}")
print(f"Baseline period : {eeg_description.get('BaselinePeriod')}")

print("\nSEEG")
print("----")
print(seeg_epochs)
print(f"Array shape     : {seeg_epochs.get_data().shape}")
print(f"Sampling rate   : {seeg_epochs.info['sfreq']} Hz")
print(f"Time range      : {seeg_epochs.tmin} to {seeg_epochs.tmax} s")
print(f"Bad channels    : {len(seeg_epochs.info['bads'])}")
print(f"Baseline applied: {seeg_description.get('BaselineCorrection')}")
print(f"Baseline period : {seeg_description.get('BaselinePeriod')}")

print("\nSIMULTANEOUS RUN CHECK")
print("----------------------")
print(f"Same sampling rate : {eeg_epochs.info['sfreq'] == seeg_epochs.info['sfreq']}")
print(f"Same time vector   : {np.array_equal(eeg_epochs.times, seeg_epochs.times)}")
print(f"EEG epoch count    : {len(eeg_epochs)}")
print(f"SEEG epoch count   : {len(seeg_epochs)}")

print("\nSTIMULATION")
print("-----------")

stimulation_conditions = eeg_events["trial_type"].dropna().unique()
print(f"Unique conditions: {len(stimulation_conditions)}")

for condition in stimulation_conditions:
    print(condition)

print("\nMRI")
print("---")
print(f"File       : {mri_file}")
print(f"Shape      : {mri.shape}")
print(f"Voxel sizes: {mri.header.get_zooms()[:3]}")

print("\nLoading complete.")