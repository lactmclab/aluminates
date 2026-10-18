#!/usr/bin/env python3
"""
Plot validation Energy RMSE and Force RMSE versus epoch directly from
the MACE training .log file.

This reproduces the *top-type* RMSE-vs-epoch panel in your reference figure.

Usage:
    python plot_rmse_vs_epoch.py polar_ft_1l_ca_run-42.log

Outputs:
    mace_validation_rmse_vs_epoch.png
    mace_validation_rmse_vs_epoch.pdf
    mace_validation_rmse_vs_epoch.svg
"""

import re
import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python plot_rmse_vs_epoch.py polar_ft_1l_ca_run-42.log"
    )

logfile = sys.argv[1]

pattern = re.compile(
    r"Epoch\s+(\d+):.*?"
    r"RMSE_E_per_atom=\s*([0-9.+\-Ee]+)\s*meV.*?"
    r"RMSE_F=\s*([0-9.+\-Ee]+)\s*meV\s*/\s*A"
)

records = {}

with open(logfile, "r", errors="replace") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            epoch = int(m.group(1))
            e_rmse = float(m.group(2))
            f_rmse = float(m.group(3))

            # If the log contains a restarted training section,
            # keep the LAST reported value for a duplicated epoch.
            records[epoch] = (e_rmse, f_rmse)

if not records:
    raise RuntimeError("No epoch RMSE records were found in the log.")

epochs = np.array(sorted(records))
e_rmse = np.array([records[e][0] for e in epochs])
f_rmse = np.array([records[e][1] for e in epochs])

fig, ax1 = plt.subplots(figsize=(6.3, 4.4))
ax2 = ax1.twinx()

line1, = ax1.plot(
    epochs, e_rmse,
    linewidth=1.8,
    linestyle="-",
    marker="o",
    markersize=3,
    label="Energy RMSE"
)
line2, = ax2.plot(
    epochs, f_rmse,
    linewidth=1.8,
    linestyle="--",
    label="Force RMSE"
)

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Energy RMSE (meV/atom)")
ax2.set_ylabel(r"Force RMSE (meV/$\AA$)")

ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)

lines = [line1, line2]
labels = [x.get_label() for x in lines]
ax1.legend(lines, labels, loc="best", frameon=False)

best_loss_epoch = 23
if best_loss_epoch in records:
    ax1.axvline(best_loss_epoch, linestyle="--", linewidth=1.0)
    ax1.text(
        best_loss_epoch,
        ax1.get_ylim()[1],
        "  saved epoch 23",
        rotation=90,
        va="top",
        ha="left",
        fontsize=8
    )

fig.tight_layout()

for ext in ("png", "pdf", "svg"):
    fig.savefig(f"mace_validation_rmse_vs_epoch.{ext}", dpi=600, bbox_inches="tight")

plt.close(fig)

print(f"Parsed {len(epochs)} epochs from {logfile}")
print(f"Minimum energy RMSE: {e_rmse.min():.3f} meV/atom at epoch {epochs[e_rmse.argmin()]}")
print(f"Minimum force RMSE : {f_rmse.min():.3f} meV/A at epoch {epochs[f_rmse.argmin()]}")
print("Saved mace_validation_rmse_vs_epoch.png/.pdf/.svg")
