# ========= START SCRIPT =========

import os
import re
import random
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.io import write

# ============================================================
# USER SETTINGS
# ============================================================

ROOT_DIR = "."              # parent folder containing d1, d2, ... d14
OUTPUT_DIR = "mace_data"    # output folder

SUBSAMPLE_RATE = 50         # take 1 frame every 50 frames
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42

SYSTEM_CHARGE = 0.0
SYSTEM_SPIN = 1.0

# CP2K units
POSITIONS_IN_BOHR = False   # usually False for CP2K xyz
CELL_IN_BOHR = False        # usually False if .cell is in Angstrom
FORCES_IN_HARTREE_PER_BOHR = True
ENERGY_IN_HARTREE = True

# If energy is not found in xyz comment line, read from .ener column.
# CP2K .ener usually has potential energy in column 4, zero-based indexing.
ENER_COLUMN = 4

# Unit conversions
HARTREE_TO_EV = 27.211386245988
HARTREE_BOHR_TO_EV_ANG = 51.4220674763
BOHR_TO_ANG = 0.529177210903


# ============================================================
# SMALL HELPER FUNCTIONS
# ============================================================

def natural_sort_key(text):
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", str(text))]


def find_first(folder, pattern):
    files = sorted(Path(folder).glob(pattern), key=natural_sort_key)
    if len(files) == 0:
        return None
    return files[0]


def extract_energy_from_comment(comment_line):
    """
    CP2K pos xyz comment usually contains:
    i = 0, time = 0.000, E = -1545.3012312261
    """
    if "E =" not in comment_line:
        return None

    try:
        return float(comment_line.split("E =")[1].split()[0])
    except Exception:
        return None


def read_ener_file(ener_file):
    """
    Reads CP2K .ener file.
    Default column 4 = potential energy.
    """
    if ener_file is None:
        return None

    data = np.loadtxt(ener_file, comments="#")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    return data[:, ENER_COLUMN]


def read_cell_file(cell_file):
    """
    Reads CP2K .cell file.
    Expected columns:
    step time Ax Ay Az Bx By Bz Cx Cy Cz
    """
    data = np.loadtxt(cell_file, comments="#")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    cells = data[:, 2:11].reshape(-1, 3, 3)

    if CELL_IN_BOHR:
        cells = cells * BOHR_TO_ANG

    return cells


def convert_energy(energy):
    if ENERGY_IN_HARTREE:
        return energy * HARTREE_TO_EV
    return energy


def convert_positions(positions):
    positions = np.array(positions, dtype=float)
    if POSITIONS_IN_BOHR:
        positions = positions * BOHR_TO_ANG
    return positions


def convert_forces(forces):
    forces = np.array(forces, dtype=float)
    if FORCES_IN_HARTREE_PER_BOHR:
        forces = forces * HARTREE_BOHR_TO_EV_ANG
    return forces


# ============================================================
# PARSE ONE CP2K FOLDER
# ============================================================

def parse_one_folder(folder):
    folder = Path(folder)

    pos_file = find_first(folder, "*-pos-1.xyz")
    frc_file = find_first(folder, "*-frc-1.xyz")
    cell_file = find_first(folder, "*.cell")
    ener_file = find_first(folder, "*.ener")

    if pos_file is None or frc_file is None or cell_file is None:
        return []

    print(f"Reading: {folder}")

    cells = read_cell_file(cell_file)
    ener_values = read_ener_file(ener_file)

    frames = []
    frame_index = 0

    with open(pos_file, "r") as fpos, open(frc_file, "r") as ffrc:

        while True:
            line_pos = fpos.readline()
            line_frc = ffrc.readline()

            if not line_pos or not line_frc:
                break

            line_pos = line_pos.strip()
            line_frc = line_frc.strip()

            if line_pos == "":
                continue

            n_atoms = int(line_pos)
            n_atoms_frc = int(line_frc)

            if n_atoms != n_atoms_frc:
                raise ValueError(f"Atom number mismatch in {folder}, frame {frame_index}")

            comment_pos = fpos.readline()
            comment_frc = ffrc.readline()

            symbols = []
            positions = []
            forces = []

            for _ in range(n_atoms):
                p = fpos.readline().split()
                f = ffrc.readline().split()

                symbols.append(p[0])
                positions.append([float(p[1]), float(p[2]), float(p[3])])
                forces.append([float(f[1]), float(f[2]), float(f[3])])

            if frame_index % SUBSAMPLE_RATE == 0:

                energy = extract_energy_from_comment(comment_pos)

                if energy is None:
                    if ener_values is None:
                        raise ValueError(f"No energy found in {folder}")
                    energy = ener_values[frame_index]

                energy_ev = convert_energy(energy)
                positions_ang = convert_positions(positions)
                forces_ev_ang = convert_forces(forces)

                if len(cells) == 1:
                    cell = cells[0]
                else:
                    cell = cells[frame_index]

                atoms = Atoms(
                    symbols=symbols,
                    positions=positions_ang,
                    cell=cell,
                    pbc=True
                )

                # These keys are used later for MACE training
                atoms.info["REF_energy"] = energy_ev
                atoms.arrays["REF_forces"] = forces_ev_ang

                # Extra metadata
                atoms.info["total_charge"] = SYSTEM_CHARGE
                atoms.info["total_spin"] = SYSTEM_SPIN
                atoms.info["config_type"] = folder.name
                atoms.info["source_folder"] = str(folder)
                atoms.info["source_frame"] = frame_index

                frames.append(atoms)

            frame_index += 1

    print(f"  saved {len(frames)} frames from {frame_index} total frames")
    return frames


# ============================================================
# FIND ALL CP2K DATA FOLDERS
# ============================================================

def find_all_data_folders(root):
    root = Path(root)

    data_folders = []

    for folder in root.rglob("*"):
        if not folder.is_dir():
            continue

        pos_file = find_first(folder, "*-pos-1.xyz")
        frc_file = find_first(folder, "*-frc-1.xyz")
        cell_file = find_first(folder, "*.cell")

        if pos_file is not None and frc_file is not None and cell_file is not None:
            data_folders.append(folder)

    data_folders = sorted(data_folders, key=natural_sort_key)
    return data_folders


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    random.seed(RANDOM_SEED)

    output = Path(OUTPUT_DIR)
    output.mkdir(exist_ok=True)

    print("Searching for CP2K folders...")
    data_folders = find_all_data_folders(ROOT_DIR)

    print(f"Found {len(data_folders)} CP2K data folders")

    all_frames = []

    for folder in data_folders:
        frames = parse_one_folder(folder)
        all_frames.extend(frames)

    if len(all_frames) == 0:
        raise RuntimeError("No frames were extracted. Check file names and folder structure.")

    print()
    print(f"Total extracted frames = {len(all_frames)}")

    print("Shuffling frames...")
    random.shuffle(all_frames)

    n_total = len(all_frames)
    n_train = int(TRAIN_RATIO * n_total)
    n_val = int(VAL_RATIO * n_total)

    train_frames = all_frames[:n_train]
    val_frames = all_frames[n_train:n_train + n_val]
    test_frames = all_frames[n_train + n_val:]

    print()
    print("Dataset split:")
    print(f"Train frames = {len(train_frames)}")
    print(f"Val frames   = {len(val_frames)}")
    print(f"Test frames  = {len(test_frames)}")

    write(output / "train.extxyz", train_frames, format="extxyz")
    write(output / "val.extxyz", val_frames, format="extxyz")
    write(output / "test.extxyz", test_frames, format="extxyz")
    write(output / "all.extxyz", all_frames, format="extxyz")

    print()
    print("Done mama.")
    print(f"Files written inside: {OUTPUT_DIR}")
    print()
    print("Use these keys during MACE training:")
    print("--energy_key REF_energy --forces_key REF_forces")


if __name__ == "__main__":
    main()

# ========= END SCRIPT =========