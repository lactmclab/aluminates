#!/usr/bin/env python3
"""
Publication-style MACE parity plots.

Expected input files (created by mace_eval_configs):
    train_pred.extxyz
    val_pred.extxyz
    test_pred.extxyz

The original reference labels must be:
    REF_energy
    REF_forces

MACE predictions are expected as:
    MACE_energy
    MACE_forces

Outputs:
    mace_parity_6panel.png
    mace_parity_6panel.pdf
    mace_parity_6panel.svg

    train_energy_parity.png / .pdf / .svg
    train_force_parity.png  / .pdf / .svg
    ... same for validation and test

RMSE definitions:
    Energy RMSE = sqrt(mean(((E_MACE-E_REF)/N_atoms)^2))
                  reported in meV/atom

    Force RMSE  = sqrt(mean((F_MACE-F_REF)^2))
                  over all x/y/z atomic force components,
                  reported in meV/Angstrom
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ase.io import read

# ---------------- User settings ----------------
FILES = {
    "Train": "train_pred.extxyz",
    "Validation": "val_pred.extxyz",
    "Test": "test_pred.extxyz",
}

REF_E_KEY = "REF_energy"
REF_F_KEY = "REF_forces"
PRED_E_KEY = "MACE_energy"
PRED_F_KEY = "MACE_forces"

# Plotting only: forces can contain >100,000 points.
# RMSE is ALWAYS calculated using all force components.
MAX_FORCE_POINTS_SHOWN = 40000
RANDOM_SEED = 42

POINT_SIZE_ENERGY = 25
POINT_SIZE_FORCE = 5
ALPHA_ENERGY = 0.80
ALPHA_FORCE = 0.18

DPI = 600
# ------------------------------------------------


def rmse(x):
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.mean(x * x))


def load_split(filename):
    frames = read(filename, index=":")

    if len(frames) == 0:
        raise RuntimeError(f"No structures found in {filename}")

    ref_e = []
    pred_e = []
    ref_f = []
    pred_f = []
    natoms = []

    for i, atoms in enumerate(frames):
        missing = []

        if REF_E_KEY not in atoms.info:
            missing.append(REF_E_KEY)
        if PRED_E_KEY not in atoms.info:
            missing.append(PRED_E_KEY)
        if REF_F_KEY not in atoms.arrays:
            missing.append(REF_F_KEY)
        if PRED_F_KEY not in atoms.arrays:
            missing.append(PRED_F_KEY)

        if missing:
            raise KeyError(
                f"{filename}, structure {i}: missing keys {missing}\n"
                f"Available info keys: {list(atoms.info.keys())}\n"
                f"Available array keys: {list(atoms.arrays.keys())}"
            )

        n = len(atoms)
        natoms.append(n)

        # total energy -> energy per atom
        ref_e.append(float(atoms.info[REF_E_KEY]) / n)
        pred_e.append(float(atoms.info[PRED_E_KEY]) / n)

        # flatten x,y,z force components
        ref_f.append(np.asarray(atoms.arrays[REF_F_KEY], dtype=float).reshape(-1))
        pred_f.append(np.asarray(atoms.arrays[PRED_F_KEY], dtype=float).reshape(-1))

    ref_e = np.asarray(ref_e)
    pred_e = np.asarray(pred_e)
    ref_f = np.concatenate(ref_f)
    pred_f = np.concatenate(pred_f)
    natoms = np.asarray(natoms)

    energy_rmse_mev_atom = 1000.0 * rmse(pred_e - ref_e)
    force_rmse_mev_A = 1000.0 * rmse(pred_f - ref_f)

    return {
        "ref_e": ref_e,
        "pred_e": pred_e,
        "ref_f": ref_f,
        "pred_f": pred_f,
        "natoms": natoms,
        "energy_rmse": energy_rmse_mev_atom,
        "force_rmse": force_rmse_mev_A,
    }


def padded_limits(x, y, fraction=0.035):
    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    span = hi - lo
    if span == 0:
        span = max(abs(lo), 1.0) * 0.02
    pad = fraction * span
    return lo - pad, hi + pad


def plot_energy(ax, data, title):
    x = data["ref_e"]
    y = data["pred_e"]
    lo, hi = padded_limits(x, y)

    ax.scatter(
        x, y,
        s=POINT_SIZE_ENERGY,
        alpha=ALPHA_ENERGY,
        edgecolors="none"
    )
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1.2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Reference energy per atom (eV)")
    ax.set_ylabel("MACE energy per atom (eV)")
    ax.set_title(title, fontweight="bold")

    ax.text(
        0.04, 0.95,
        f"RMSE = {data['energy_rmse']:.2f} meV/atom\n"
        f"N = {len(x)} configurations",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.80, linewidth=0.6)
    )

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)


def plot_force(ax, data, title):
    x_all = data["ref_f"]
    y_all = data["pred_f"]

    # RMSE was already calculated on ALL force components.
    # Subsample only the displayed points, to avoid an unreadably dense figure.
    if len(x_all) > MAX_FORCE_POINTS_SHOWN:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(x_all), MAX_FORCE_POINTS_SHOWN, replace=False)
        x = x_all[idx]
        y = y_all[idx]
    else:
        x = x_all
        y = y_all

    lo, hi = padded_limits(x_all, y_all)

    ax.scatter(
        x, y,
        s=POINT_SIZE_FORCE,
        alpha=ALPHA_FORCE,
        edgecolors="none",
        rasterized=True
    )
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1.2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Reference force (eV/$\AA$)")
    ax.set_ylabel(r"MACE force (eV/$\AA$)")
    ax.set_title(title, fontweight="bold")

    ax.text(
        0.04, 0.95,
        f"RMSE = {data['force_rmse']:.2f} meV/$\\AA$\n"
        f"N = {len(x_all):,} force components",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.80, linewidth=0.6)
    )

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)


def save_individual(data_by_split):
    for split, data in data_by_split.items():
        stem = split.lower().replace(" ", "_")

        fig, ax = plt.subplots(figsize=(4.2, 4.0))
        plot_energy(ax, data, f"{split}: Energy")
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"{stem}_energy_parity.{ext}", dpi=DPI, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.2, 4.0))
        plot_force(ax, data, f"{split}: Forces")
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"{stem}_force_parity.{ext}", dpi=DPI, bbox_inches="tight")
        plt.close(fig)


def main():
    for f in FILES.values():
        if not Path(f).exists():
            raise FileNotFoundError(
                f"{f} not found.\n"
                "First run run_mace_parity.sh (or mace_eval_configs manually)."
            )

    data_by_split = {name: load_split(path) for name, path in FILES.items()}

    print("\n========== RMSE summary ==========")
    for name, data in data_by_split.items():
        unique_n = sorted(set(data["natoms"].tolist()))
        print(
            f"{name:10s} : "
            f"E RMSE = {data['energy_rmse']:.4f} meV/atom | "
            f"F RMSE = {data['force_rmse']:.4f} meV/A | "
            f"configs = {len(data['ref_e'])} | atoms/config = {unique_n}"
        )
    print("==================================\n")

    # 2 rows x 3 columns:
    # row 1 = energy parity, row 2 = force parity
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.3))

    for col, (name, data) in enumerate(data_by_split.items()):
        plot_energy(axes[0, col], data, f"{name}: Energy")
        plot_force(axes[1, col], data, f"{name}: Forces")

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    for ax, label in zip(axes.flat, panel_labels):
        ax.text(
            -0.16, 1.04, label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom"
        )

    fig.tight_layout()

    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"mace_parity_6panel.{ext}", dpi=DPI, bbox_inches="tight")

    plt.close(fig)

    save_individual(data_by_split)

    print("Saved combined figure:")
    print("  mace_parity_6panel.png")
    print("  mace_parity_6panel.pdf")
    print("  mace_parity_6panel.svg")
    print("\nAlso saved individual Train / Validation / Test energy and force panels.")


if __name__ == "__main__":
    main()
