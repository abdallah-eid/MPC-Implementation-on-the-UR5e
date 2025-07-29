import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import warnings
from scipy.io import loadmat

# Load the system matrices from the .mat file
mat_file_path = r'C:\Users\Abdallah Eid\Desktop\ss1_matrices.mat'
mat_data = loadmat(mat_file_path)

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

print(f"System dimensions: nx={A.shape[0]}, nu={B.shape[1]}, ny={C.shape[0]}")

nx = A.shape[0]
nu = B.shape[1]
ny = C.shape[0]

print(f"System dimensions: nx={nx}, nu={nu}, ny={ny}")
print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")
print(f"C shape: {C.shape}")
print(f"D shape: {D.shape}")

T = 0.01     # Sampling time
N = 20      # Prediction horizon

# Input and output constraints
u_max_full = np.array([2, 2, 2, 2, 2, 2])
u_max = u_max_full[:nu]
u_min = np.array([-2, -2, -2, -2, -2, -2])[:nu]
y_max = 2 * np.pi * np.ones(ny)
y_min = -y_max

# Acceleration constraints (change in velocity)
acc_max = 0.5 * np.ones(nu)  # Maximum acceleration: 0.5 rad/s²
acc_min = -acc_max           # Minimum acceleration: -0.5 rad/s²

print(f"u_max: {u_max}")
print(f"y_max length: {len(y_max)}")
print(f"Acceleration limits: [{acc_min[0]:.1f}, {acc_max[0]:.1f}] rad/s²")

# CasADi symbolic variables
x = ca.MX.sym('x', nx, 1)
u = ca.MX.sym('u', nu, 1)

# Discrete-time dynamics and output equations
x_next = A @ x + B @ u
y = C @ x + D @ u

f_dynamics = ca.Function('f_dynamics', [x, u], [x_next])
f_output = ca.Function('f_output', [x, u], [y])

# Optimization variables
U = ca.MX.sym('U', nu, N)         # control inputs over horizon
x0_sym = ca.MX.sym('x0', nx)      # initial state
Yref_sym = ca.MX.sym('Yref', ny, N) # reference output trajectory
u_prev_sym = ca.MX.sym('u_prev', nu, 1)  # previous control input for acceleration constraint

# Predict states over horizon
X = ca.MX.zeros(nx, N+1)
X[:, 0] = x0_sym
for k in range(N):
    X[:, k+1] = f_dynamics(X[:, k], U[:, k])

# Objective function weights
Qy =30 * np.eye(ny)
R = 10 * np.eye(nu)

# Objective and constraints initialization
obj = 0
g = []

for k in range(N):
    y_k = C @ X[:, k] + D @ U[:, k]
    y_ref_k = Yref_sym[:, k]
    obj = obj + (y_k - y_ref_k).T @ Qy @ (y_k - y_ref_k) + U[:, k].T @ R @ U[:, k]

# Output constraints for all predicted steps
for k in range(N+1):
    if k < N:
        y_k = C @ X[:, k] + D @ U[:, k]
    else:
        y_k = C @ X[:, k]  # Last step, no control input applied
    g.append(y_k)

# Acceleration constraints (change in control input / sampling time)
for k in range(N):
    if k == 0:
        # First step: compare with previous control input
        acc_k = (U[:, k] - u_prev_sym) / T
    else:
        # Subsequent steps: compare with previous step in horizon
        acc_k = (U[:, k] - U[:, k-1]) / T
    g.append(acc_k)

g = ca.vertcat(*g)

# NLP problem definition
OPT_variables = ca.reshape(U, -1, 1)
P = ca.vertcat(x0_sym, ca.reshape(Yref_sym, -1, 1), u_prev_sym)
nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}

# Solver options
opts = {
    'ipopt.max_iter': 200,
    'ipopt.print_level': 0,
    'print_time': False,
    'ipopt.acceptable_tol': 1e-8,
    'ipopt.acceptable_obj_change_tol': 1e-4
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

# Simulation setup - initial position
y_initial = np.array([0, -1.57, 0, -1.57, 0, 0])[:ny]  # Initial configuration
try:
    x0 = np.linalg.pinv(C) @ y_initial
    print(f"Initial output target: {y_initial}")
    print(f"Calculated initial state: {x0}")
    print(f"Verification - C*x0: {C @ x0}")
except:
    print("Warning: Could not calculate initial state using pinv(C). Using zero initial state.")
    x0 = np.zeros(nx)

# Calculate the initial robot output
y0_robot = C @ x0
print(f"Actual robot initial output: {y0_robot}")

# Simulation parameters
sim_time = 20  # Total simulation time

# Define target position
y_targets = np.array([0.8, -0.5, 0.7, -1, 1.3, 2])[:ny]  # Target positions
y_start = y0_robot.copy()  # Starting positions

print(f"\nTarget positions: {y_targets}")
print(f"Starting positions: {y_start}")
print("Using constant reference trajectory (immediate target)")

def generate_constant_reference(t_current, horizon_steps):
    """Generate constant reference trajectory - immediate target"""
    Yref = np.zeros((ny, horizon_steps))
    
    for i in range(ny):
        # Constant target position
        Yref[i, :] = y_targets[i]
    
    return Yref

u0 = np.zeros((N, nu))   # Initial guess for inputs
u_prev = np.zeros(nu)    # Previous control input (starts at zero)

# Bounds
lbx = np.tile(u_min, N)
ubx = np.tile(u_max, N)

# Output constraints bounds
output_lbg = np.tile(y_min, N+1)
output_ubg = np.tile(y_max, N+1)

# Acceleration constraints bounds
acc_lbg = np.tile(acc_min, N)
acc_ubg = np.tile(acc_max, N)

# Combine all constraint bounds
lbg = np.concatenate([output_lbg, acc_lbg])
ubg = np.concatenate([output_ubg, acc_ubg])

print(f"Total constraints: {len(lbg)} ({len(output_lbg)} output + {len(acc_lbg)} acceleration)")

# Storage variables
t0 = 0
mpcctr = 0
max_steps = int(sim_time / T)
y_cl = []
u_cl = []
acc_cl = []  # Store control acceleration data
y_ref_cl = []  # Store reference trajectory
t = []

# Main MPC simulation loop
print("\nStarting MPC simulation with constant reference trajectory...")
print("=" * 80)
print(f"{'Step':<6} {'Time':<8} {'Max Acc':<10} {'Tracking Error':<15} {'Current Output':<30}")
print("=" * 80)

while mpcctr < max_steps:
    
    # Generate constant reference trajectory over prediction horizon
    Yref = generate_constant_reference(t0, N)
    
    # Current reference for tracking error calculation
    y_ref_current = generate_constant_reference(t0, 1)[:, 0]
    
    # Set parameters for solver
    p = np.concatenate([x0.flatten(), Yref.flatten(order='F'), u_prev.flatten()])
    x0_opt = u0.flatten(order='F')
    
    # Solve NLP
    try:
        sol = solver(x0=x0_opt, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)
        
        # Check solver status
        solver_stats = solver.stats()
        if not solver_stats['success']:
            print(f"Warning: Solver did not converge at step {mpcctr}, status: {solver_stats['return_status']}")
            
    except Exception as e:
        print(f"Solver failed at step {mpcctr}: {e}")
        break
    
    # Extract first control input and apply to system
    u_opt_flat = np.array(sol['x']).flatten()
    u_opt = u_opt_flat.reshape((nu, N), order='F').T
    
    # Calculate current acceleration
    u_current = u_opt[0, :]
    acc_current = (u_current - u_prev) / T
    max_acc = np.max(np.abs(acc_current))
    
    # Apply first control input to simulation
    u_current_col = u_current.reshape(-1, 1)
    x0_new = np.array(f_dynamics(x0, u_current_col)).flatten()
    y0 = np.array(C @ x0_new).flatten()
    
    # Store closed-loop data
    y_cl.append(y0)
    u_cl.append(u_current)
    acc_cl.append(acc_current)
    y_ref_cl.append(y_ref_current)
    t.append(t0)
    
    # Print progress with acceleration info
    if mpcctr % 50 == 0:
        tracking_error = np.linalg.norm(y0 - y_ref_current, 2)
        y0_str = '[' + ', '.join([f'{val:.3f}' for val in y0[:3]]) + '...]' if len(y0) > 3 else str(y0)
        print(f"{mpcctr:<6} {t0:<8.2f} {max_acc:<10.3f} {tracking_error:<15.6f} {y0_str:<30}")
        
        # Check if acceleration constraint is active
        if max_acc > 0.45:  # Close to limit
            print(f"    → High acceleration detected: {max_acc:.3f} rad/s² (limit: {acc_max[0]:.1f})")
    
    # Update for next iteration
    x0 = x0_new
    u_prev = u_current  # Store current input as previous for next iteration
    t0 = t0 + T
    u0 = np.vstack([u_opt[1:, :], u_opt[-1, :].reshape(1, -1)])
    mpcctr = mpcctr + 1

print("=" * 80)
print(f"MPC simulation completed after {mpcctr} steps")

# Check if data was collected
if len(t) == 0:
    warnings.warn('No simulation steps run. Nothing to plot.')
    exit()

# Convert lists to numpy arrays
y_cl = np.array(y_cl).T  # ny x time steps
y_ref_cl = np.array(y_ref_cl).T  # ny x time steps
u_cl = np.array(u_cl)    # time steps x nu
acc_cl = np.array(acc_cl)  # time steps x nu
t = np.array(t)

# Statistics
max_control_acc = np.max(np.abs(acc_cl))
avg_control_acc = np.mean(np.abs(acc_cl))
control_acc_violations = np.sum(np.abs(acc_cl) > acc_max[0])

tracking_errors = np.linalg.norm(y_cl - y_ref_cl, axis=0)
rms_tracking_error = np.sqrt(np.mean(tracking_errors**2))
max_tracking_error = np.max(tracking_errors)

print(f"\nControl Acceleration Statistics:")
print(f"Maximum control acceleration: {max_control_acc:.3f} rad/s² (limit: {acc_max[0]:.1f})")
print(f"Average control acceleration: {avg_control_acc:.3f} rad/s²")
print(f"Control acceleration violations: {control_acc_violations} out of {len(t)} steps")

print(f"\nTracking Performance:")
print(f"RMS tracking error: {rms_tracking_error:.6f}")
print(f"Maximum tracking error: {max_tracking_error:.6f}")

# Final positions
print(f"\nFinal positions vs targets:")
y_final = y_cl[:, -1]
for i in range(ny):
    error = abs(y_final[i] - y_targets[i])
    print(f"Joint {i+1}: {y_final[i]:.3f} (target: {y_targets[i]:.3f}, error: {error:.3f})")

# Status summary
print(f"\n{'='*50}")
print("SIMULATION SUMMARY:")
if control_acc_violations == 0:
    print("✓ All control accelerations within limits")
else:
    print(f"⚠ Control acceleration violations: {control_acc_violations}")

print(f"✓ Simulation completed successfully")
print(f"{'='*50}")

# Plot output trajectories with reference
fig, axes = plt.subplots(int(np.ceil(ny/2)), 2, figsize=(15, 10))
if ny == 1:
    axes = [axes]
elif ny <= 2:
    axes = axes.reshape(-1)
else:
    axes = axes.flatten()

for i in range(ny):
    ax = axes[i] if ny > 1 else axes[0]
    ax.plot(t, y_cl[i, :], linewidth=1.5, label='Output', color='blue')
    ax.plot(t, y_ref_cl[i, :], '--r', linewidth=1.2, label='Constant Reference', color='red')
    ax.legend()
    ax.set_title(f'Output y_{i+1} vs Constant Reference')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(f'y_{i+1} [rad]')
    ax.grid(True)

# Hide unused subplots
if ny < len(axes):
    for i in range(ny, len(axes)):
        axes[i].set_visible(False)

plt.tight_layout()
plt.show()

# Plot tracking errors
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Individual tracking errors
for i in range(ny):
    tracking_error_i = np.abs(y_cl[i, :] - y_ref_cl[i, :])
    ax1.plot(t, tracking_error_i, linewidth=1.5, label=f'|y_{i+1} - ref_{i+1}|')
ax1.set_title('Individual Output Tracking Errors')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Absolute Error [rad]')
ax1.legend()
ax1.grid(True)

# Total tracking error
ax2.plot(t, tracking_errors, linewidth=2, color='black', label='Total Tracking Error (2-norm)')
ax2.axhline(y=rms_tracking_error, color='r', linestyle='--', alpha=0.7, label=f'RMS Error = {rms_tracking_error:.6f}')
ax2.set_title('Total Tracking Error (2-norm)')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Tracking Error')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Plot control inputs and accelerations
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Control inputs
for i in range(nu):
    ax1.plot(t, u_cl[:, i], linewidth=1.5, label=f'u_{i+1}')
    ax1.axhline(y=u_max[i], color='r', linestyle='--', alpha=0.5)
    ax1.axhline(y=u_min[i], color='r', linestyle='--', alpha=0.5)
ax1.set_title('Control Inputs')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Control Input [rad/s]')
ax1.legend()
ax1.grid(True)

# Control accelerations
for i in range(nu):
    ax2.plot(t, acc_cl[:, i], linewidth=1.5, label=f'Control acc_{i+1}')
ax2.axhline(y=acc_max[0], color='r', linestyle='--', linewidth=2, label='Control Acceleration Limit')
ax2.axhline(y=acc_min[0], color='r', linestyle='--', linewidth=2)
ax2.set_title('Control Accelerations (Change in Control Input / Sampling Time)')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Control Acceleration [rad/s²]')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Close all figures to free memory
plt.close('all')

print("Simulation and plotting completed successfully.")

# Save results for later analysis (optional)
np.savez('mpc_simulation_results.npz', 
         t=t, 
         y_cl=y_cl, 
         y_ref_cl=y_ref_cl, 
         u_cl=u_cl, 
         acc_cl=acc_cl)

print("Results saved to 'mpc_simulation_results.npz'")