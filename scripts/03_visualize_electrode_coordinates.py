"""Visualize EEG and SEEG electrode coordinates.

Outputs
-------
1. A 3D scalp EEG sensor plot.
2. A notebook-style figure containing:
   - the participant's individual T1 MRI;
   - all SEEG coordinates projected in MNI space.
3. A T1-space SEEG validation figure centred on the stimulation site.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting


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
# Paths
# ---------------------------------------------------------------------

epochs_root = (
    DATA_ROOT
    / "derivatives"
    / "epochs"
    / SUBJECT
)

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

seeg_t1_file = (
    seeg_dir
    / f"{SUBJECT}_task-{TASK}_space-T1w_electrodes.tsv"
)

seeg_mni_file = (
    seeg_dir
    / (
        f"{SUBJECT}_task-{TASK}_"
        "space-MNI152NLin2009aSym_electrodes.tsv"
    )
)

events_file = (
    eeg_dir
    / f"{SUBJECT}_task-{TASK}_{RUN}_epochs.tsv"
)

mri_file = (
    DATA_ROOT
    / SUBJECT
    / "anat"
    / f"{SUBJECT}_T1w.nii"
)

required_files = [
    eeg_channels_file,
    eeg_electrodes_file,
    eeg_coordsystem_file,
    seeg_t1_file,
    seeg_mni_file,
    events_file,
    mri_file,
]

missing_files = [
    path for path in required_files if not path.exists()
]

if missing_files:
    missing_text = "\n".join(str(path) for path in missing_files)

    raise FileNotFoundError(
        f"Required files are missing:\n{missing_text}"
    )


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def load_json(path: Path) -> dict:
    """Load a JSON file."""

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def coordinate_array(
    table: pd.DataFrame,
    scale: float = 1.0,
) -> np.ndarray:
    """Return finite x, y, z coordinates with a scale applied."""

    coordinates = (
        table[["x", "y", "z"]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    return coordinates * scale


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

eeg_channels = pd.read_csv(
    eeg_channels_file,
    sep="\t",
)

eeg_electrodes = pd.read_csv(
    eeg_electrodes_file,
    sep="\t",
)

seeg_t1 = pd.read_csv(
    seeg_t1_file,
    sep="\t",
)

seeg_mni = pd.read_csv(
    seeg_mni_file,
    sep="\t",
)

events = pd.read_csv(
    events_file,
    sep="\t",
)

eeg_coordsystem = load_json(
    eeg_coordsystem_file
)

mri = nib.load(mri_file)


# ---------------------------------------------------------------------
# Prepare EEG coordinates
# ---------------------------------------------------------------------

# EEG coordinates are stored in millimetres.
eeg_coordinates_mm = coordinate_array(
    eeg_electrodes,
)

eeg_coordinates_m = (
    eeg_coordinates_mm / 1000.0
)

eeg_names = (
    eeg_electrodes["name"]
    .astype(str)
    .to_numpy()
)

eeg_valid = np.isfinite(
    eeg_coordinates_m
).all(axis=1)

eeg_coordinates_mm = eeg_coordinates_mm[eeg_valid]
eeg_coordinates_m = eeg_coordinates_m[eeg_valid]
eeg_names = eeg_names[eeg_valid]


# ---------------------------------------------------------------------
# Prepare SEEG coordinates
# ---------------------------------------------------------------------

# Both SEEG tables store coordinates in metres.
seeg_t1_coordinates_mm = (
    coordinate_array(seeg_t1) * 1000.0
)

seeg_mni_coordinates_mm = (
    coordinate_array(seeg_mni) * 1000.0
)

seeg_t1_names = (
    seeg_t1["name"]
    .astype(str)
    .to_numpy()
)

seeg_mni_names = (
    seeg_mni["name"]
    .astype(str)
    .to_numpy()
)

seeg_t1_valid = np.isfinite(
    seeg_t1_coordinates_mm
).all(axis=1)

seeg_mni_valid = np.isfinite(
    seeg_mni_coordinates_mm
).all(axis=1)

seeg_t1_coordinates_mm = (
    seeg_t1_coordinates_mm[seeg_t1_valid]
)

seeg_mni_coordinates_mm = (
    seeg_mni_coordinates_mm[seeg_mni_valid]
)

seeg_t1_names = seeg_t1_names[seeg_t1_valid]
seeg_mni_names = seeg_mni_names[seeg_mni_valid]


# ---------------------------------------------------------------------
# Find EEG coordinates outside the MRI voxel box
# ---------------------------------------------------------------------

eeg_voxel_coordinates = nib.affines.apply_affine(
    np.linalg.inv(mri.affine),
    eeg_coordinates_mm,
)

eeg_inside_mask = (
    np.isfinite(eeg_voxel_coordinates).all(axis=1)
    & (eeg_voxel_coordinates >= 0).all(axis=1)
    & (
        eeg_voxel_coordinates
        < np.asarray(mri.shape[:3], dtype=float)
    ).all(axis=1)
)

eeg_outside_names = (
    eeg_names[~eeg_inside_mask].tolist()
)


# ---------------------------------------------------------------------
# Find stimulation coordinates
# ---------------------------------------------------------------------

stimulation_conditions = (
    events["trial_type"]
    .dropna()
    .astype(str)
    .unique()
)

if len(stimulation_conditions) != 1:
    raise ValueError(
        "Expected one stimulation condition, "
        f"but found {len(stimulation_conditions)}."
    )

stimulation_condition = stimulation_conditions[0]
stimulation_channel = stimulation_condition.split()[0]

stimulation_t1_mask = (
    seeg_t1_names == stimulation_channel
)

stimulation_mni_mask = (
    seeg_mni_names == stimulation_channel
)

if stimulation_t1_mask.sum() != 1:
    raise ValueError(
        f"Expected one T1 coordinate for "
        f"{stimulation_channel}, but found "
        f"{stimulation_t1_mask.sum()}."
    )

if stimulation_mni_mask.sum() != 1:
    raise ValueError(
        f"Expected one MNI coordinate for "
        f"{stimulation_channel}, but found "
        f"{stimulation_mni_mask.sum()}."
    )

stimulation_t1_mm = (
    seeg_t1_coordinates_mm[stimulation_t1_mask][0]
)

stimulation_mni_mm = (
    seeg_mni_coordinates_mm[stimulation_mni_mask][0]
)


# ---------------------------------------------------------------------
# Figure 1: 3D scalp EEG sensor arrangement
# ---------------------------------------------------------------------

landmarks_mm = (
    eeg_coordsystem[
        "AnatomicalLandmarkCoordinates"
    ]
)

eeg_channel_positions = {
    name: position
    for name, position in zip(
        eeg_names,
        eeg_coordinates_m,
    )
}

eeg_montage_mri = mne.channels.make_dig_montage(
    ch_pos=eeg_channel_positions,
    nasion=(
        np.asarray(
            landmarks_mm["NAS"],
            dtype=float,
        )
        / 1000.0
    ),
    lpa=(
        np.asarray(
            landmarks_mm["LPA"],
            dtype=float,
        )
        / 1000.0
    ),
    rpa=(
        np.asarray(
            landmarks_mm["RPA"],
            dtype=float,
        )
        / 1000.0
    ),
    coord_frame="mri",
)

# Compute the transformation from the montage's native MRI
# coordinate system to MNE head coordinates using NAS, LPA and RPA.
mri_to_head = mne.channels.compute_native_head_t(
    eeg_montage_mri,
    verbose=False,
)

eeg_montage_head = eeg_montage_mri.copy()
eeg_montage_head.apply_trans(mri_to_head)

eeg_channel_names = (
    eeg_channels["name"]
    .astype(str)
    .tolist()
)

eeg_info = mne.create_info(
    ch_names=eeg_channel_names,
    sfreq=1000.0,
    ch_types="eeg",
)

if "status" in eeg_channels.columns:
    status = (
        eeg_channels["status"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    eeg_info["bads"] = (
        eeg_channels.loc[
            status == "bad",
            "name",
        ]
        .astype(str)
        .tolist()
    )

eeg_info.set_montage(
    eeg_montage_head,
    on_missing="raise",
)

eeg_3d_figure_file = (
    FIGURE_DIR
    / f"{SUBJECT}_{RUN}_eeg_sensors_3d.png"
)

eeg_figure = mne.viz.plot_sensors(
    eeg_info,
    kind="3d",
    ch_type="eeg",
    show_names=False,
    show=False,
)

eeg_figure.suptitle(
    (
        f"{SUBJECT} {RUN}: scalp EEG sensors\n"
        "Bad channels are shown in red"
    ),
    fontsize=12,
)

eeg_figure.savefig(
    eeg_3d_figure_file,
    dpi=200,
    bbox_inches="tight",
)

plt.close(eeg_figure)


# ---------------------------------------------------------------------
# Figure 2: notebook-style MRI and MNI glass brain
# ---------------------------------------------------------------------

notebook_figure_file = (
    FIGURE_DIR
    / f"{SUBJECT}_{RUN}_notebook_style.png"
)

notebook_figure = plt.figure(
    figsize=(12, 10),
    facecolor="black",
)

mri_display = plotting.plot_anat(
    mri,
    display_mode="ortho",
    figure=notebook_figure,
    axes=(0.02, 0.52, 0.96, 0.43),
    annotate=True,
    draw_cross=True,
    black_bg=True,
    colorbar=False,
    title=f"{SUBJECT}: individual T1 MRI",
)

adjacency_matrix = np.zeros(
    (
        len(seeg_mni_coordinates_mm),
        len(seeg_mni_coordinates_mm),
    ),
    dtype=float,
)

glass_display = plotting.plot_connectome(
    adjacency_matrix,
    node_coords=seeg_mni_coordinates_mm,
    display_mode="ortho",
    node_size=22,
    black_bg=True,
    colorbar=False,
    figure=notebook_figure,
    axes=(0.02, 0.02, 0.96, 0.43),
    title="All SEEG coordinates in MNI space",
)

glass_display.add_markers(
    marker_coords=[stimulation_mni_mm],
    marker_color="#ef4444",
    marker_size=80,
)

notebook_figure.savefig(
    notebook_figure_file,
    dpi=200,
    bbox_inches="tight",
    facecolor="black",
)

mri_display.close()
glass_display.close()
plt.close(notebook_figure)


# ---------------------------------------------------------------------
# Figure 3: SEEG validation in individual T1 space
# ---------------------------------------------------------------------

t1_validation_figure_file = (
    FIGURE_DIR
    / f"{SUBJECT}_{RUN}_seeg_T1w_validation.png"
)

t1_display = plotting.plot_anat(
    mri,
    display_mode="ortho",
    cut_coords=stimulation_t1_mm,
    title=(
        f"{SUBJECT} {RUN}: T1-space SEEG; "
        f"stimulation at {stimulation_channel}"
    ),
    annotate=True,
    draw_cross=True,
    black_bg=True,
    colorbar=False,
)

t1_display.add_markers(
    marker_coords=seeg_t1_coordinates_mm,
    marker_color="#facc15",
    marker_size=12,
)

t1_display.add_markers(
    marker_coords=[stimulation_t1_mm],
    marker_color="#ef4444",
    marker_size=80,
)

t1_display.savefig(
    t1_validation_figure_file,
    dpi=200,
    bbox_inches="tight",
)

t1_display.close()


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

print("COORDINATE VISUALIZATION")
print("------------------------")
print(f"Subject                : {SUBJECT}")
print(f"Run                    : {RUN}")
print(f"EEG coordinates        : {len(eeg_coordinates_mm)}")
print(f"EEG bad channels       : {len(eeg_info['bads'])}")
print(f"EEG outside MRI        : {eeg_outside_names}")
print(f"SEEG T1 coordinates    : {len(seeg_t1_coordinates_mm)}")
print(f"SEEG MNI coordinates   : {len(seeg_mni_coordinates_mm)}")
print(f"Stimulating channel    : {stimulation_channel}")
print(
    f"Stimulation T1 (mm)    : "
    f"{np.round(stimulation_t1_mm, 2)}"
)
print(
    f"Stimulation MNI (mm)   : "
    f"{np.round(stimulation_mni_mm, 2)}"
)

print("\nFIGURES")
print("-------")
print(f"EEG sensors    : {eeg_3d_figure_file}")
print(f"Notebook style : {notebook_figure_file}")
print(f"T1 validation  : {t1_validation_figure_file}")

print("\nVisualization complete.")