from mace.calculators import MACECalculator
from ase.io import read, write
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase import units
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
import os, json, time

# -------------------------------
# User settings
# -------------------------------
input_xyz = "sys_cip2.xyz"
traj_file = "md.traj"
restart_file = "restart.traj"
state_file = "state.json"
log_file = "md.log"

box_length = 16.16891075

timestep_fs = 0.5
temperature_K = 300
total_steps = 1000000       

save_every_fs = 5.0
save_interval = int(save_every_fs / timestep_fs)   # 10 steps

# -------------------------------
# Restart or fresh start
# -------------------------------
if os.path.exists(restart_file) and os.path.exists(state_file):
    print("Restarting from restart.traj")
    atoms = read(restart_file)

    with open(state_file, "r") as f:
        state = json.load(f)

    completed_steps = state["completed_steps"]
    
    # Reinitialize velocities (important for Nose–Hoover)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    Stationary(atoms)
    ZeroRotation(atoms)

else:
    print("Starting fresh from sys_cip2.xyz")
    atoms = read(input_xyz)
    atoms.wrap()

    atoms.set_pbc([True, True, True])
    atoms.set_cell([box_length, box_length, box_length])

#   write header ONLY for fresh run
    with open(log_file, "w") as f:
         f.write("Step Time(fs) Energy(eV) Temp(K) Wall(s)\n")

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    Stationary(atoms)
    ZeroRotation(atoms)

    completed_steps = 0

# -------------------------------
# Calculator
# -------------------------------
atoms.calc = MACECalculator(
    model_paths="polar_ft_1l_ca.model",
    device="cpu",		# use "cpu" if no GPU
    default_dtype="float64"      
)

# -------------------------------
# MD setup
# -------------------------------
dyn = NoseHooverChainNVT(
    atoms,
    timestep=0.5 * units.fs,
    temperature_K=temperature_K,
    tdamp=100 * units.fs
)

# -------------------------------
# Trajectory append mode
# -------------------------------
traj = Trajectory(traj_file, "a", atoms)

# -------------------------------
# Save function
# -------------------------------
def save_all():
    global completed_steps
    traj.write()
    write(restart_file, atoms)

    with open(state_file, "w") as f:
        json.dump({"completed_steps": completed_steps}, f)

# -------------------------------
# Step counter
# -------------------------------
def update_step():
    global completed_steps
    completed_steps += 1


# -------------------------------
# Logging
# -------------------------------
start_time = time.time()

def print_status():
    epot = atoms.get_potential_energy()
    temp = atoms.get_temperature()
    elapsed = time.time() - start_time

    line = (
        f"{completed_steps:6d} "
        f"{completed_steps * timestep_fs:8.2f} "
        f"{epot:14.6f} "
        f"{temp:8.2f} "
        f"{elapsed:10.2f}"
    )

    print(line)

    with open(log_file, "a") as f:
        f.write(line + "\n")

# -------------------------------
# Attach functions (ORDER MATTERS)
# -------------------------------
dyn.attach(update_step, interval=1)
dyn.attach(print_status, interval=10)
dyn.attach(save_all, interval=save_interval)

# -------------------------------
# Run only remaining steps
# -------------------------------
remaining_steps = total_steps - completed_steps

print(f"Completed steps: {completed_steps}")
print(f"Remaining steps: {remaining_steps}")
print("Starting Nosé–Hoover MD...")

if remaining_steps > 0:
    dyn.run(remaining_steps)
else:
    print("Requested total_steps already completed.")

# -------------------------------
# Final save
# -------------------------------
write("final.xyz", atoms)
write(restart_file, atoms)

print("MD finished.")
