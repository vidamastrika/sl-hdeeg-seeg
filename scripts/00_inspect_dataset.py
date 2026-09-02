"""Inspect the structure and metadata of the CCEP coregistration dataset."""

import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "ccepcoreg"


# ---------------------------------------------------------------------
# Validate dataset path
# ---------------------------------------------------------------------

print("PROJECT")
print("-------")
print(f"Project root : {PROJECT_ROOT}")
print(f"Dataset root : {DATA_ROOT.resolve()}")
print(f"Dataset found: {DATA_ROOT.exists()}")

if not DATA_ROOT.exists():
    raise FileNotFoundError(
        "Dataset not found. Check the data/raw/ccepcoreg symbolic link."
    )


# ---------------------------------------------------------------------
# Dataset description
# ---------------------------------------------------------------------

description_file = DATA_ROOT / "dataset_description.json"

print("\nDATASET DESCRIPTION")
print("-------------------")

if description_file.exists():
    with description_file.open(encoding="utf-8") as file:
        description = json.load(file)

    print(f"Name        : {description.get('Name', 'Not specified')}")
    print(f"BIDS version: {description.get('BIDSVersion', 'Not specified')}")
else:
    print("dataset_description.json not found")


# ---------------------------------------------------------------------
# Participants
# ---------------------------------------------------------------------

subject_dirs = sorted(
    path for path in DATA_ROOT.glob("sub-*") if path.is_dir()
)

print("\nPARTICIPANTS")
print("------------")
print(f"Participant folders: {len(subject_dirs)}")

if subject_dirs:
    print(f"First participant  : {subject_dirs[0].name}")
    print(f"Last participant   : {subject_dirs[-1].name}")

participants_file = DATA_ROOT / "participants.tsv"

if participants_file.exists():
    participants = pd.read_csv(participants_file, sep="\t")
    print(f"Rows in participants.tsv: {len(participants)}")
    print(f"Columns: {list(participants.columns)}")
else:
    print("participants.tsv not found")


# ---------------------------------------------------------------------
# Modalities
# ---------------------------------------------------------------------

modality_counts = {
    "anat": 0,
    "eeg": 0,
    "ieeg": 0,
}

for subject_dir in subject_dirs:
    for modality in modality_counts:
        if (subject_dir / modality).is_dir():
            modality_counts[modality] += 1

print("\nSUBJECT-LEVEL MODALITIES")
print("------------------------")

for modality, count in modality_counts.items():
    print(f"{modality:<4}: present for {count} participants")


# ---------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------

derivatives_dir = DATA_ROOT / "derivatives"

print("\nDERIVATIVES")
print("-----------")
print(f"Derivatives folder found: {derivatives_dir.is_dir()}")

if derivatives_dir.is_dir():
    derivative_folders = sorted(
        path.name for path in derivatives_dir.iterdir() if path.is_dir()
    )
    print(f"Derivative folders: {derivative_folders}")

# ---------------------------------------------------------------------
# Derived EEG and SEEG epochs
# ---------------------------------------------------------------------

epochs_root = DATA_ROOT / "derivatives" / "epochs"

print("\nDERIVED EEG AND SEEG EPOCHS")
print("---------------------------")

for modality in ("eeg", "ieeg"):
    participants_with_data = []
    runs_per_participant = {}

    for subject_dir in subject_dirs:
        modality_dir = epochs_root / subject_dir.name / modality

        if modality_dir.is_dir():
            run_files = sorted(modality_dir.glob("*_epochs.npy"))

            if run_files:
                participants_with_data.append(subject_dir.name)
                runs_per_participant[subject_dir.name] = len(run_files)

    total_runs = sum(runs_per_participant.values())

    print(f"\nModality: {modality.upper()}")
    print(f"Participants with data: {len(participants_with_data)}")
    print(f"Total runs            : {total_runs}")

    if runs_per_participant:
        run_counts = list(runs_per_participant.values())
        print(f"Minimum runs/subject  : {min(run_counts)}")
        print(f"Maximum runs/subject  : {max(run_counts)}")
        print(
            f"Runs for sub-01       : "
            f"{runs_per_participant.get('sub-01', 'not available')}"
        )

print("\nInspection complete.")