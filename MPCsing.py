import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import warnings
from scipy.io import loadmat
import sys
import os

# Ensure the .mat file exists
mat_file_path = r'C:\Users\Abdallah Eid\Desktop\ss1_matrices.mat'
if not os.path.exists(mat_file_path):
    raise FileNotFoundError(f"Could not find .mat file at {mat_file_path}")

# Load the system matrices from the .mat file
try:
    mat_data = loadmat(mat_file_path)
except Exception as e:
    raise RuntimeError(f"Failed to load .mat file: {e}")

# Extract the system matrices
try:
    A = mat_data['A']
    B = mat_data['B']
    C = mat_data['C']
    D = mat_data['D']
except KeyError as e:
    print(f"Key error: {e}")
    print("Available keys in .mat file:", list(mat_data.keys()))
    raise

# Validate matrix shapes
nx = A.shape[0]  # Number of states
nu = B.shape[1]  # Number of inputs
ny = C.shape[0]  # Number of outputs
if A.shape != (nx, nx) or B.shape != (nx, nu) or C.shape != (ny, nx) or D.shape != (ny, nu):
    raise ValueError(f"Inconsistent matrix shapes: A={A.shape}, B={B.shape}, C={C.shape}, D={D.shape}")

print(f"System dimensions: nx={nx}, nu={nu}, ny={ny}")
print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")
print(f"C shape: {C.shape}")
print(f"D shape: {D.shape}")

# MPC parameters
T = 0.01     # Sampling time
N = 20       # Prediction horizon
sim_time = 20  # Total simulation time

# Input and output constraints
u_max_full = np.array([2, 2, 2, 2, 2, 2])
u_max = u_max_full[:nu]
u_min = -u_max
y_max = 2 * np.pi * np.ones(ny)
y_min = -y_max

# Acceleration constraints
acc_max = 0.5 * np.ones(nu)  # Maximum acceleration: 0.5 rad/s²
acc_min = -acc_max

print(f"u_max: {u_max}")
print(f"y_max length: {len(y_max)}")
print(f"Acceleration limits: [{acc_min[0]:.1f}, {acc_max[0]:.1f}] rad/s²")

# Singularity avoidance parameters
singularity_margin = 0.15  # Safety margin from singularities (radians)

print(f"Singularity avoidance margin: {singularity_margin:.3f} rad")

# CasADi symbolic variables
x = ca.MX.sym('x', nx, 1)
u = ca.MX.sym('u', nu, 1)

# Discrete-time dynamics and output equations
x_next = A @ x + B @ u
y = C @ x + D @ u

f_dynamics = ca.Function('f_dynamics', [x, u], [x_next])
f_output = ca.Function('f_output', [x, u], [y])

# Optimization variables
U = ca.MX.sym('U', nu, N)         # Control inputs over horizon
x0_sym = ca.MX.sym('x0', nx)      # Initial state
Yref_sym = ca.MX.sym('Yref', ny, N)  # Reference output trajectory
u_prev_sym = ca.MX.sym('u_prev', nu, 1)  # Previous control input

# Predict states over horizon
X = ca.MX.zeros(nx, N+1)
X[:, 0] = x0_sym
for k in range(N):
    X[:, k+1] = f_dynamics(X[:, k], U[:, k])

# Objective function weights
Qy = 120 * np.eye(ny)
R = 2 * np.eye(nu)

# Objective and constraints
obj = 0
g = []

for k in range(N):
    y_k = C @ X[:, k] + D @ U[:, k]
    y_ref_k = Yref_sym[:, k]
    obj += (y_k - y_ref_k).T @ Qy @ (y_k - y_ref_k) + U[:, k].T @ R @ U[:, k]

# Output constraints
for k in range(N+1):
    if k < N:
        y_k = C @ X[:, k] + D @ U[:, k]
    else:
        y_k = C @ X[:, k]
    g.append(y_k)

# Acceleration constraints
for k in range(N):
    acc_k = (U[:, k] - (u_prev_sym if k == 0 else U[:, k-1])) / T
    g.append(acc_k)

# Singularity avoidance constraints
for k in range(N):
    y_k = C @ X[:, k] + D @ U[:, k]
    if ny >= 3:  # Joint 3
        joint3 = y_k[2]
        g.append(joint3**2)  # >= singularity_margin²
        g.append(ca.sin(joint3/2)**2)  # <= 1 - singularity_margin²
    if ny >= 5:  # Joint 5
        joint5 = y_k[4]
        g.append(joint5**2)  # >= singularity_margin²
        g.append(ca.sin(joint5/2)**2)  # <= 1 - singularity_margin²

g = ca.vertcat(*g)

# NLP problem
OPT_variables = ca.reshape(U, -1, 1)
P = ca.vertcat(x0_sym, ca.reshape(Yref_sym, -1, 1), u_prev_sym)
nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}

# Solver options
opts = {
    'ipopt.max_iter': 200,
    'ipopt.print_level': 0,
    'print_time': False,
    'ipopt.acceptable_tol': 1e-8,
    'ipopt.acceptable_obj_change_tol': 1e-6
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

# Initial state
y_initial = np.array([0, -1.57, 0.5, -1.0, 0.5, 0])[:ny]
try:
    x0 = np.linalg.pinv(C) @ y_initial
    print(f"Initial output target: {y_initial}")
    print(f"Calculated initial state: {x0}")
    print(f"Verification - C*x0: {C @ x0}")
except Exception as e:
    print(f"Warning: Could not calculate initial state using pinv(C): {e}")
    print("Using zero initial state.")
    x0 = np.zeros(nx)

y0_robot = C @ x0
print(f"Actual robot initial output: {y0_robot}")

# Exponential trajectory parameters
y_targets = np.array([0.3, 0.5, 0.4, 0.6, 0.8, 0.5])[:ny]
y_start = y0_robot.copy()
tau = np.array([3.0, 4.0, 2.5, 3.5, 2.0, 4.5])[:ny]
max_trajectory_acceleration = 0.4

def generate_exponential_reference(t_current, horizon_steps):
    t_horizon = np.array([t_current + k * T for k in range(horizon_steps)])
    Yref = np.zeros((ny, horizon_steps))
    for i in range(ny):
        Yref[i, :] = y_targets[i] + (y_start[i] - y_targets[i]) * np.exp(-t_horizon / tau[i])
    return Yref

def calculate_exponential_acceleration(t_current, horizon_steps):
    t_horizon = np.array([t_current + k * T for k in range(horizon_steps)])
    Yacc = np.zeros((ny, horizon_steps))
    for i in range(ny):
        Yacc[i, :] = (y_start[i] - y_targets[i]) * (1/tau[i]**2) * np.exp(-t_horizon / tau[i])
    return Yacc

# Verify trajectory
print(f"\nExponential trajectory parameters:")
print(f"Start positions: {y_start}")
print(f"Target positions: {y_targets}")
print(f"Time constants: {tau} seconds")

max_traj_accs = []
for i in range(ny):
    max_acc = abs(y_start[i] - y_targets[i]) / tau[i]**2
    max_traj_accs.append(max_acc)
    print(f"Joint {i+1}: Max trajectory acceleration = {max_acc:.3f} rad/s²")
    if max_acc > max_trajectory_acceleration:
        tau_min = np.sqrt(abs(y_start[i] - y_targets[i]) / max_trajectory_acceleration)
        tau[i] = max(tau[i], tau_min * 1.1)
        max_acc_new = abs(y_start[i] - y_targets[i]) / tau[i]**2
        print(f"  → Adjusted time constant to {tau[i]:.2f}s, new max acc: {max_acc_new:.3f} rad/s²")
        max_traj_accs[i] = max_acc_new

print(f"Overall maximum trajectory acceleration: {max(max_traj_accs):.3f} rad/s²")

# Singularity check
y_ref_check = generate_exponential_reference(0, int(sim_time/T))
if ny >= 3:
    joint3_values = y_ref_check[2, :]
    joint3_close_to_sing = np.any(np.abs(joint3_values) < singularity_margin) or \
                          np.any(np.abs(np.abs(joint3_values) - np.pi) < singularity_margin)
    print(f"Joint 3: Range [{np.min(joint3_values):.3f}, {np.max(joint3_values):.3f}], "
          f"Near singularity: {joint3_close_to_sing}")
if ny >= 5:
    joint5_values = y_ref_check[4, :]
    joint5_close_to_sing = np.any(np.abs(joint5_values) < singularity_margin) or \
                          np.any(np.abs(np.abs(joint5_values) - np.pi) < singularity_margin)
    print(f"Joint 5: Range [{np.min(joint5_values):.3f}, {np.max(joint5_values):.3f}], "
          f"Near singularity: {joint5_close_to_sing}")

# Bounds
lbx = np.tile(u_min, N)
ubx = np.tile(u_max, N)
output_lbg = np.tile(y_min, N+1)
output_ubg = np.tile(y_max, N+1)
acc_lbg = np.tile(acc_min, N)
acc_ubg = np.tile(acc_max, N)

# Singularity bounds
sing_lbg = []
sing_ubg = []
for k in range(N):
    if ny >= 3:
        sing_lbg.extend([singularity_margin**2, 0])
        sing_ubg.extend([np.inf, 1 - singularity_margin**2])
    if ny >= 5:
        sing_lbg.extend([singularity_margin**2, 0])
        sing_ubg.extend([np.inf, 1 - singularity_margin**2])

sing_lbg = np.array(sing_lbg)
sing_ubg = np.array(sing_ubg)
lbg = np.concatenate([output_lbg, acc_lbg, sing_lbg])
ubg = np.concatenate([output_ubg, acc_ubg, sing_ubg])

print(f"Total constraints: {len(lbg)} ({len(output_lbg)} output + {len(acc_lbg)} acceleration + {len(sing_lbg)} singularity)")

# Simulation loop
u0 = np.zeros((N, nu))
u_prev = np.zeros(nu)
t0 = 0
mpcctr = 0
max_steps = int(sim_time / T)
y_cl = []
u_cl = []
acc_cl = []
y_ref_cl = []
y_ref_acc_cl = []
t = []
singularity_violations = []

print("\nStarting MPC simulation...")
print("Target positions:", y_targets)
print("=" * 100)
print(f"{'Step':<6} {'Time':<8} {'Max Acc':<10} {'Tracking Error':<15} {'Sing Viol':<10} {'Current Output':<30}")
print("=" * 100)

while mpcctr < max_steps:
    Yref = generate_exponential_reference(t0, N)
    y_ref_current = generate_exponential_reference(t0, 1)[:, 0]
    y_ref_acc_current = calculate_exponential_acceleration(t0, 1)[:, 0]
    
    p = np.concatenate([x0.flatten(), Yref.flatten(order='F'), u_prev.flatten()])
    x0_opt = u0.flatten(order='F')
    
    try:
        sol = solver(x0=x0_opt, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)
        if not solver.stats()['success']:
            print(f"Warning: Solver failed at step {mpcctr}: {solver.stats()['return_status']}")
            break
    except Exception as e:
        print(f"Solver failed at step {mpcctr}: {e}")
        break
    
    u_opt_flat = np.array(sol['x']).flatten()
    u_opt = u_opt_flat.reshape((nu, N), order='F').T
    u_current = u_opt[0, :]
    acc_current = (u_current - u_prev) / T
    max_acc = np.max(np.abs(acc_current))
    
    u_current_col = u_current.reshape(-1, 1)
    x0_new = np.array(f_dynamics(x0, u_current_col)).flatten()
    y0 = np.array(C @ x0_new).flatten()
    
    sing_viol = 0
    if ny >= 3:
        joint3 = y0[2]
        if abs(joint3) < singularity_margin or abs(abs(joint3) - np.pi) < singularity_margin:
            sing_viol += 1
    if ny >= 5:
        joint5 = y0[4]
        if abs(joint5) < singularity_margin or abs(abs(joint5) - np.pi) < singularity_margin:
            sing_viol += 1
    
    y_cl.append(y0)
    u_cl.append(u_current)
    acc_cl.append(acc_current)
    y_ref_cl.append(y_ref_current)
    y_ref_acc_cl.append(y_ref_acc_current)
    singularity_violations.append(sing_viol)
    t.append(t0)
    
    if mpcctr % 50 == 0:
        tracking_error = np.linalg.norm(y0 - y_ref_current, 2)
        y0_str = '[' + ', '.join([f'{val:.3f}' for val in y0[:3]]) + '...]' if len(y0) > 3 else str(y0)
        print(f"{mpcctr:<6} {t0:<8.2f} {max_acc:<10.3f} {tracking_error:<15.6f} {sing_viol:<10d} {y0_str:<30}")
        if max_acc > 0.45:
            print(f"    → High acceleration detected: {max_acc:.3f} rad/s² (limit: {acc_max[0]:.1f})")
        if sing_viol > 0:
            print(f"    → WARNING: Near singularity! Violations: {sing_viol}")
    
    x0 = x0_new
    u_prev = u_current
    t0 += T
    u0 = np.vstack([u_opt[1:, :], u_opt[-1, :].reshape(1, -1)])
    mpcctr += 1

print("=" * 100)
print(f"MPC simulation completed after {mpcctr} steps")

# Check if data was collected
if len(t) == 0:
    raise RuntimeError("No simulation steps run. Nothing to plot.")

# Convert lists to arrays
y_cl = np.array(y_cl).T
y_ref_cl = np.array(y_ref_cl).T
y_ref_acc_cl = np.array(y_ref_acc_cl).T
u_cl = np.array(u_cl)
acc_cl = np.array(acc_cl)
t = np.array(t)
singularity_violations = np.array(singularity_violations)

# Statistics
max_control_acc = np.max(np.abs(acc_cl))
avg_control_acc = np.mean(np.abs(acc_cl))
control_acc_violations = np.sum(np.abs(acc_cl) > acc_max[0])
max_ref_acc = np.max(np.abs(y_ref_acc_cl))
avg_ref_acc = np.mean(np.abs(y_ref_acc_cl))
tracking_errors = np.linalg.norm(y_cl - y_ref_cl, axis=0)
rms_tracking_error = np.sqrt(np.mean(tracking_errors**2))
max_tracking_error = np.max(tracking_errors)
total_singularity_violations = np.sum(singularity_violations)

print(f"\nControl Acceleration Statistics:")
print(f"Maximum control acceleration: {max_control_acc:.3f} rad/s² (limit: {acc_max[0]:.1f})")
print(f"Average control acceleration: {avg_control_acc:.3f} rad/s²")
print(f"Control acceleration violations: {control_acc_violations} steps")

print(f"\nReference Trajectory Acceleration Statistics:")
print(f"Maximum reference acceleration: {max_ref_acc:.3f} rad/s²")
print(f"Average reference acceleration: {avg_ref_acc:.3f} rad/s²")

print(f"\nSingularity Avoidance Statistics:")
print(f"Total singularity violations: {total_singularity_violations} steps")
print(f"Steps with violations: {np.sum(singularity_violations > 0)}")

print(f"\nTracking Performance:")
print(f"RMS tracking error: {rms_tracking_error:.6f}")
print(f"Maximum tracking error: {max_tracking_error:.6f}")

print(f"\nFinal positions vs targets:")
y_final = y_cl[:, -1]
for i in range(ny):
    error = abs(y_final[i] - y_targets[i])
    print(f"Joint {i+1}: {y_final[i]:.3f} (target: {y_targets[i]:.3f}, error: {error:.3f})")

print(f"\n{'='*50}")
print("SIMULATION SUMMARY:")
print("✓ All control accelerations within limits" if control_acc_violations == 0 else 
      f"⚠ Control acceleration violations: {control_acc_violations}")
print("✓ No singularity violations detected" if total_singularity_violations == 0 else 
      f"⚠ Singularity violations: {total_singularity_violations}")
print("✓ Reference trajectory accelerations within limits" if max_ref_acc <= max_trajectory_acceleration else 
      "⚠ Reference trajectory acceleration exceeds limits")
print(f"{'='*50}")

# Plot outputs
fig, axes = plt.subplots(max(1, (ny+1)//2), min(ny, 2), figsize=(12, 3*max(1, (ny+1)//2)))
axes = np.atleast_1d(axes).flatten()

for i in range(ny):
    ax = axes[i]
    ax.plot(t, y_cl[i, :], linewidth=1.5, label='Output')
    ax.plot(t, y_ref_cl[i, :], '--r', linewidth=1.2, label='Reference')
    if i in [2, 4] and ny > i:
        ax.axhline(y=singularity_margin, color='orange', linestyle=':', alpha=0.7, label='Singularity Zone')
        ax.axhline(y=-singularity_margin, color='orange', linestyle=':', alpha=0.7)
        ax.axhline(y=np.pi-singularity_margin, color='orange', linestyle=':', alpha=0.7)
        ax.axhline(y=-np.pi+singularity_margin, color='orange', linestyle=':', alpha=0.7)
    ax.set_title(f'Output y_{i+1} (Target: {y_targets[i]:.1f})')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(f'y_{i+1} [rad]')
    ax.legend()
    ax.grid(True)

for i in range(ny, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()

# Plot errors and violations
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
for i in range(ny):
    ax1.plot(t, np.abs(y_cl[i, :] - y_ref_cl[i, :]), label=f'|y_{i+1} - ref_{i+1}|')
ax1.set_title('Individual Output Tracking Errors')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Absolute Error [rad]')
ax1.legend()
ax1.grid(True)

ax2.plot(t, tracking_errors, label='Total Tracking Error')
ax2.axhline(y=rms_tracking_error, color='r', linestyle='--', label=f'RMS Error = {rms_tracking_error:.6f}')
ax2.set_title('Total Tracking Error (2-norm)')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Tracking Error')
ax2.legend()
ax2.grid(True)

ax3.plot(t, singularity_violations, 'ro-', markersize=4, label='Singularity Violations')
ax3.set_title('Singularity Constraint Violations')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Number of Violations')
ax3.legend()
ax3.grid(True)
ax3.set_ylim(-0.5, max(3, np.max(singularity_violations) + 0.5))

plt.tight_layout()
plt.show()

# Plot controls
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
for i in range(nu):
    ax1.plot(t, u_cl[:, i], label=f'u_{i+1}')
    ax1.axhline(y=u_max[i], color='r', linestyle='--', alpha=0.5)
    ax1.axhline(y=u_min[i], color='r', linestyle='--', alpha=0.5)
ax1.set_title('Control Inputs')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Control Input [rad/s]')
ax1.legend()
ax1.grid(True)

for i in range(nu):
    ax2.plot(t, acc_cl[:, i], label=f'Control acc_{i+1}')
ax2.axhline(y=acc_max[0], color='r', linestyle='--', label='Control Acceleration Limit')
ax2.axhline(y=acc_min[0], color='r', linestyle='--')
ax2.set_title('Control Accelerations')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Control Acceleration [rad/s²]')
ax2.legend()
ax2.grid(True)

for i in range(ny):
    ax3.plot(t, y_ref_acc_cl[i, :], label=f'Ref acc_{i+1}')
ax3.axhline(y=max_trajectory_acceleration, color='g', linestyle='--', 
            label=f'Trajectory Acceleration Limit ({max_trajectory_acceleration})')
ax3.axhline(y=-max_trajectory_acceleration, color='g', linestyle='--')
ax3.set_title('Reference Trajectory Accelerations')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Acceleration [rad/s²]')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()