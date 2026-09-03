"""Load and validate EEG and SEEG electrode coordinates."""

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
# Paths
# ---------------------------------------------------------------------

epochs_root = DATA_ROOT / "derivatives" / "epochs" / SUBJECT

eeg_dir = epochs_root / "eeg"
seeg_dir = epochs_root / "ieeg"

eeg_channels_file = (
    eeg_dir
    / f"{SUBJECT}_task-{TASK}_{RUN}_channels.tsv"
)
eeg_electrodes_file = (
    eeg_dir
    / f"{SUBJECT}_task-{TASK}_electrodes.tsv"
)
eeg_coordsystem_file = (
    eeg_dir
    / f"{SUBJECT}_task-{TASK}_coordsystem.json"
)

seeg_channels_file = (
    seeg_dir
    / f"{SUBJECT}_task-{TASK}_{RUN}_channels.tsv"
)
seeg_t1_file = (
    seeg_dir
    / f"{SUBJECT}_task-{TASK}_space-T1w_electrodes.tsv"
)
seeg_t1_coordsystem_file = (
    seeg_dir
    / f"{SUBJECT}_task-{TASK}_space-T1w_coordsystem.json"
)
seeg_mni_file = (
    seeg_dir
    / (
        f"{SUBJECT}_task-{TASK}_"
        "space-MNI152NLin2009aSym_electrodes.tsv"
    )
)
seeg_mni_coordsystem_file = (
    seeg_dir
    / (
        f"{SUBJECT}_task-{TASK}_"
        "space-MNI152NLin2009aSym_coordsystem.json"
    )
)

events_file = (
    eeg_dir
    / f"{SUBJECT}_task-{TASK}_{RUN}_epochs.tsv"
)
mri_file = DATA_ROOT / SUBJECT / "anat" / f"{SUBJECT}_T1w.nii"


required_files = [
    eeg_channels_file,
    eeg_electrodes_file,
    eeg_coordsystem_file,
    seeg_channels_file,
    seeg_t1_file,
    seeg_t1_coordsystem_file,
    seeg_mni_file,
    seeg_mni_coordsystem_file,
    events_file,
    mri_file,
]

missing_files = [path for path in required_files if not path.exists()]

if missing_files:
    missing_text = "\n".join(str(path) for path in missing_files)
    raise FileNotFoundError(f"Required files are missing:\n{missing_text}")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_json(path: Path) -> dict:
    """Load a JSON file."""

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def coordinate_array(table: pd.DataFrame) -> np.ndarray:
    """Return x, y, z coordinates as a floating-point array."""

    return table[["x", "y", "z"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)


def names_missing_from_coordinates(
    channel_table: pd.DataFrame,
    electrode_table: pd.DataFrame,
) -> list[str]:
    """Find recording channels without coordinate entries."""

    channel_names = set(channel_table["name"].astype(str))
    coordinate_names = set(electrode_table["name"].astype(str))

    return sorted(channel_names - coordinate_names)


def count_inside_mri(
    coordinates_m: np.ndarray,
    mri_image: nib.Nifti1Image,
) -> tuple[int, int]:
    """Count coordinates falling within the MRI voxel dimensions."""

    coordinates_mm = coordinates_m * 1000.0

    inverse_affine = np.linalg.inv(mri_image.affine)
    voxel_coordinates = nib.affines.apply_affine(
        inverse_affine,
        coordinates_mm,
    )

    finite = np.isfinite(voxel_coordinates).all(axis=1)

    lower_bound = (voxel_coordinates >= 0).all(axis=1)
    upper_bound = (
        voxel_coordinates
        < np.asarray(mri_image.shape[:3], dtype=float)
    ).all(axis=1)

    inside = finite & lower_bound & upper_bound

    return int(inside.sum()), int(len(inside))


# ---------------------------------------------------------------------
# Load tables and metadata
# ---------------------------------------------------------------------

eeg_channels = pd.read_csv(eeg_channels_file, sep="\t")
eeg_electrodes = pd.read_csv(eeg_electrodes_file, sep="\t")

seeg_channels = pd.read_csv(seeg_channels_file, sep="\t")
seeg_t1 = pd.read_csv(seeg_t1_file, sep="\t")
seeg_mni = pd.read_csv(seeg_mni_file, sep="\t")

events = pd.read_csv(events_file, sep="\t")

eeg_coordsystem = load_json(eeg_coordsystem_file)
seeg_t1_coordsystem = load_json(seeg_t1_coordsystem_file)
seeg_mni_coordsystem = load_json(seeg_mni_coordsystem_file)

mri = nib.load(mri_file)


# ---------------------------------------------------------------------
# Convert coordinates to metres
# ---------------------------------------------------------------------

eeg_coordinates_mm = coordinate_array(eeg_electrodes)
eeg_coordinates_m = eeg_coordinates_mm / 1000.0

seeg_t1_coordinates_m = coordinate_array(seeg_t1)
seeg_mni_coordinates_m = coordinate_array(seeg_mni)


# ---------------------------------------------------------------------
# Create an MNE montage for the scalp EEG
# ---------------------------------------------------------------------

eeg_valid = np.isfinite(eeg_coordinates_m).all(axis=1)

eeg_channel_positions = {
    name: position
    for name, position in zip(
        eeg_electrodes.loc[eeg_valid, "name"].astype(str),
        eeg_coordinates_m[eeg_valid],
    )
}

landmarks_mm = eeg_coordsystem["AnatomicalLandmarkCoordinates"]

eeg_montage = mne.channels.make_dig_montage(
    ch_pos=eeg_channel_positions,
    nasion=np.asarray(landmarks_mm["NAS"], dtype=float) / 1000.0,
    lpa=np.asarray(landmarks_mm["LPA"], dtype=float) / 1000.0,
    rpa=np.asarray(landmarks_mm["RPA"], dtype=float) / 1000.0,
    coord_frame="mri",
)


# ---------------------------------------------------------------------
# Match recording channels with coordinate tables
# ---------------------------------------------------------------------

missing_eeg_coordinates = names_missing_from_coordinates(
    eeg_channels,
    eeg_electrodes,
)
missing_seeg_t1_coordinates = names_missing_from_coordinates(
    seeg_channels,
    seeg_t1,
)
missing_seeg_mni_coordinates = names_missing_from_coordinates(
    seeg_channels,
    seeg_mni,
)

t1_names = set(seeg_t1["name"].astype(str))
mni_names = set(seeg_mni["name"].astype(str))

seeg_coordinate_name_match = t1_names == mni_names


# ---------------------------------------------------------------------
# Locate the stimulating contact
# ---------------------------------------------------------------------

stimulation_conditions = events["trial_type"].dropna().astype(str).unique()

if len(stimulation_conditions) != 1:
    raise ValueError(
        "Expected one stimulation condition, but found "
        f"{len(stimulation_conditions)}."
    )

stimulation_condition = stimulation_conditions[0]
stimulation_channel = stimulation_condition.split()[0]

stimulating_rows = seeg_t1.loc[
    seeg_t1["name"].astype(str) == stimulation_channel
]

if len(stimulating_rows) == 1:
    stimulation_coordinate_m = coordinate_array(stimulating_rows)[0]
else:
    stimulation_coordinate_m = None


# ---------------------------------------------------------------------
# MRI-bound checks
# ---------------------------------------------------------------------

eeg_inside, eeg_total = count_inside_mri(
    eeg_coordinates_m[eeg_valid],
    mri,
)

seeg_valid = np.isfinite(seeg_t1_coordinates_m).all(axis=1)

seeg_inside, seeg_total = count_inside_mri(
    seeg_t1_coordinates_m[seeg_valid],
    mri,
)

# ---------------------------------------------------------------------
# Additional coordinate diagnostics
# ---------------------------------------------------------------------

seeg_recording_names = set(seeg_channels["name"].astype(str))
seeg_t1_names = set(seeg_t1["name"].astype(str))

seeg_not_recorded_in_run = sorted(
    seeg_t1_names - seeg_recording_names
)

eeg_coordinates_mm_for_voxels = eeg_coordinates_m[eeg_valid] * 1000.0

eeg_voxel_coordinates = nib.affines.apply_affine(
    np.linalg.inv(mri.affine),
    eeg_coordinates_mm_for_voxels,
)

eeg_inside_mask = (
    np.isfinite(eeg_voxel_coordinates).all(axis=1)
    & (eeg_voxel_coordinates >= 0).all(axis=1)
    & (
        eeg_voxel_coordinates
        < np.asarray(mri.shape[:3], dtype=float)
    ).all(axis=1)
)

valid_eeg_names = (
    eeg_electrodes.loc[eeg_valid, "name"]
    .astype(str)
    .to_numpy()
)

eeg_outside_mri = valid_eeg_names[~eeg_inside_mask].tolist()

# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

print("SELECTED RECORDING")
print("------------------")
print(f"Subject: {SUBJECT}")
print(f"Task   : {TASK}")
print(f"Run    : {RUN}")

print("\nEEG COORDINATES")
print("---------------")
print(
    f"Coordinate system : "
    f"{eeg_coordsystem.get('EEGCoordinateSystem')}"
)
print(
    f"Stored unit       : "
    f"{eeg_coordsystem.get('EEGCoordinateUnits')}"
)
print(f"Recording channels: {len(eeg_channels)}")
print(f"Coordinate rows   : {len(eeg_electrodes)}")
print(f"Finite coordinates: {int(eeg_valid.sum())}")
print(f"Missing channels  : {len(missing_eeg_coordinates)}")
print(f"Within MRI bounds : {eeg_inside}/{eeg_total}")
print(f"MNE montage       : {len(eeg_montage.ch_names)} channels")
print(f"Outside MRI names : {eeg_outside_mri}")

if missing_eeg_coordinates:
    print(f"Missing EEG names : {missing_eeg_coordinates}")

print("\nSEEG COORDINATES — INDIVIDUAL T1")
print("--------------------------------")
print(
    f"Coordinate system : "
    f"{seeg_t1_coordsystem.get('iEEGCoordinateSystemDescription')}"
)
print(
    f"Stored unit       : "
    f"{seeg_t1_coordsystem.get('iEEGCoordinateUnits')}"
)
print(f"Recording channels: {len(seeg_channels)}")
print(f"Coordinate rows   : {len(seeg_t1)}")
print(f"Finite coordinates: {int(seeg_valid.sum())}")
print(f"Missing channels  : {len(missing_seeg_t1_coordinates)}")
print(f"Within MRI bounds : {seeg_inside}/{seeg_total}")
print(
    f"Coordinates not used in this run: "
    f"{len(seeg_not_recorded_in_run)}"
)
print(f"Unused coordinate names: {seeg_not_recorded_in_run}")

if missing_seeg_t1_coordinates:
    print(f"Missing SEEG names: {missing_seeg_t1_coordinates}")

print("\nSEEG COORDINATES — MNI")
print("----------------------")
print(
    f"Coordinate system : "
    f"{seeg_mni_coordsystem.get('iEEGCoordinateSystem')}"
)
print(
    f"Stored unit       : "
    f"{seeg_mni_coordsystem.get('iEEGCoordinateUnits')}"
)
print(f"Coordinate rows   : {len(seeg_mni)}")
print(f"Missing channels  : {len(missing_seeg_mni_coordinates)}")
print(
    f"T1/MNI names match: {seeg_coordinate_name_match}"
)

if missing_seeg_mni_coordinates:
    print(f"Missing MNI names : {missing_seeg_mni_coordinates}")

print("\nSTIMULATION LOCATION")
print("--------------------")
print(f"Condition          : {stimulation_condition}")
print(f"Stimulating channel: {stimulation_channel}")
print(
    f"Found in T1 table  : "
    f"{stimulation_coordinate_m is not None}"
)

if stimulation_coordinate_m is not None:
    print(
        "T1 coordinate (m) : "
        f"{np.round(stimulation_coordinate_m, 5)}"
    )
    print(
        "T1 coordinate (mm): "
        f"{np.round(stimulation_coordinate_m * 1000.0, 2)}"
    )

print("\nCoordinate validation complete.")