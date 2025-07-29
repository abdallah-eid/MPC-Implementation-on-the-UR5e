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
Qy = 120 * np.eye(ny)
R = 2 * np.eye(nu)

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
# Acceleration = (u_k - u_{k-1}) / T
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
    'ipopt.max_iter': 100,
    'ipopt.print_level': 0,
    'print_time': False,
    'ipopt.acceptable_tol': 1e-8,
    'ipopt.acceptable_obj_change_tol': 1e-6
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

# Simulation setup
y_initial = np.array([0, -1.57, 0, -1.57, 0, 0])[:ny]
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

# Sinusoidal trajectory parameters
sim_time = 20  # Total simulation time

# Initial trajectory parameters (will be adjusted for acceleration constraints)
sin_amplitude_desired = np.array([0.5, 0.8, 0.3, 0.6, 0.4, 0.2])[:ny]  # Desired amplitude for each output
sin_frequency_desired = np.array([0.2, 0.15, 0.25, 0.1, 0.3, 0.18])[:ny]  # Desired frequency (Hz) for each output
sin_phase = np.array([0, np.pi/4, np.pi/2, np.pi/3, 0, np.pi/6])[:ny]  # Phase shift for each output

# Trajectory acceleration constraint
max_trajectory_acceleration = 0.4  # rad/s² - Keep below control acceleration limit

# For a sinusoidal trajectory y(t) = A*sin(2πft + φ) + offset:
# y'(t) = A*2πf*cos(2πft + φ)
# y''(t) = -A*(2πf)²*sin(2πft + φ)
# Maximum acceleration = A*(2πf)²

print(f"\nAdjusting trajectory parameters to satisfy acceleration constraints...")
print(f"Maximum allowed trajectory acceleration: {max_trajectory_acceleration} rad/s²")

sin_amplitude = np.zeros(ny)
sin_frequency = np.zeros(ny)

for i in range(ny):
    # Calculate maximum acceleration for desired parameters
    omega = 2 * np.pi * sin_frequency_desired[i]  # Angular frequency
    max_acc_desired = sin_amplitude_desired[i] * omega**2
    
    if max_acc_desired <= max_trajectory_acceleration:
        # Parameters are acceptable as-is
        sin_amplitude[i] = sin_amplitude_desired[i]
        sin_frequency[i] = sin_frequency_desired[i]
        print(f"Output {i+1}: Desired params OK (max acc = {max_acc_desired:.3f} rad/s²)")
    else:
        # Need to adjust parameters
        # Option 1: Reduce amplitude, keep frequency
        amplitude_option1 = max_trajectory_acceleration / omega**2
        
        # Option 2: Reduce frequency, keep amplitude
        frequency_option2 = np.sqrt(max_trajectory_acceleration / sin_amplitude_desired[i]) / (2 * np.pi)
        
        # Choose the option that gives better trajectory characteristics
        # Prefer reducing amplitude if it doesn't reduce it too much
        if amplitude_option1 >= 0.3 * sin_amplitude_desired[i]:  # Keep at least 30% of desired amplitude
            sin_amplitude[i] = amplitude_option1
            sin_frequency[i] = sin_frequency_desired[i]
            print(f"Output {i+1}: Reduced amplitude {sin_amplitude_desired[i]:.3f} → {sin_amplitude[i]:.3f}")
        else:
            sin_amplitude[i] = sin_amplitude_desired[i]
            sin_frequency[i] = frequency_option2
            print(f"Output {i+1}: Reduced frequency {sin_frequency_desired[i]:.3f} → {sin_frequency[i]:.3f} Hz")
        
        # Verify the adjustment
        omega_new = 2 * np.pi * sin_frequency[i]
        max_acc_new = sin_amplitude[i] * omega_new**2
        print(f"           New max acceleration: {max_acc_new:.3f} rad/s²")

# Calculate trajectory offset to start from robot's initial position
# The sinusoidal part at t=0 is: amplitude * sin(phase)
sin_at_t0 = sin_amplitude * np.sin(sin_phase)
# We want: y0_robot = sin_amplitude * sin(phase) + offset
# Therefore: offset = y0_robot - sin_amplitude * sin(phase)
sin_offset = y0_robot - sin_at_t0

def generate_sin_reference(t_current, horizon_steps):
    """Generate sinusoidal reference trajectory over prediction horizon starting from robot's initial position"""
    t_horizon = np.array([t_current + k * T for k in range(horizon_steps)])
    Yref = np.zeros((ny, horizon_steps))
    
    for i in range(ny):
        Yref[i, :] = (sin_amplitude[i] * np.sin(2 * np.pi * sin_frequency[i] * t_horizon + sin_phase[i]) 
                      + sin_offset[i])
    
    return Yref

def calculate_trajectory_acceleration(t_current, horizon_steps):
    """Calculate the acceleration of the reference trajectory (for analysis)"""
    t_horizon = np.array([t_current + k * T for k in range(horizon_steps)])
    Yacc = np.zeros((ny, horizon_steps))
    
    for i in range(ny):
        omega = 2 * np.pi * sin_frequency[i]
        # Second derivative of sin(ωt + φ) is -ω²sin(ωt + φ)
        Yacc[i, :] = -sin_amplitude[i] * omega**2 * np.sin(omega * t_horizon + sin_phase[i])
    
    return Yacc

print(f"\nFinal sinusoidal trajectory parameters:")
print(f"Amplitudes: {sin_amplitude}")
print(f"Frequencies: {sin_frequency} Hz")
print(f"Calculated offsets: {sin_offset}")
print(f"Phase shifts: {sin_phase} rad")

# Verify maximum trajectory accelerations
max_traj_accs = []
for i in range(ny):
    omega = 2 * np.pi * sin_frequency[i]
    max_acc = sin_amplitude[i] * omega**2
    max_traj_accs.append(max_acc)
    print(f"Output {i+1}: Max trajectory acceleration = {max_acc:.3f} rad/s²")

print(f"Overall maximum trajectory acceleration: {max(max_traj_accs):.3f} rad/s²")

# Verify trajectory starts at robot's initial position
y_ref_t0 = generate_sin_reference(0, 1)[:, 0]
print(f"\nTrajectory verification:")
print(f"Reference at t=0: {y_ref_t0}")
print(f"Robot initial:    {y0_robot}")
print(f"Difference:       {np.abs(y_ref_t0 - y0_robot)}")

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
y_ref_acc_cl = []  # Store reference trajectory accelerations
t = []

# Main MPC loop
print("\nStarting MPC simulation with sinusoidal trajectory...")
print("=" * 90)
print(f"{'Step':<6} {'Time':<8} {'Max Acc':<10} {'Tracking Error':<15} {'Current Output':<30}")
print("=" * 90)

while mpcctr < max_steps:
    
    # Generate sinusoidal reference trajectory over prediction horizon
    Yref = generate_sin_reference(t0, N)
    
    # Current reference for tracking error calculation
    y_ref_current = generate_sin_reference(t0, 1)[:, 0]
    
    # Calculate reference trajectory acceleration for analysis
    y_ref_acc_current = calculate_trajectory_acceleration(t0, 1)[:, 0]
    
    # Set parameters for solver
    p = np.concatenate([x0.flatten(), Yref.flatten(order='F'), u_prev.flatten()])
    x0_opt = u0.flatten(order='F')
    
    # Solve NLP
    try:
        sol = solver(x0=x0_opt, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)
        
        # Check solver status
        solver_stats = solver.stats()
        if not solver_stats['success']:
            print(f"Warning: Solver did not converge at step {mpcctr}")
            
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
    
    # Apply first control input
    u_current_col = u_current.reshape(-1, 1)
    x0_new = np.array(f_dynamics(x0, u_current_col)).flatten()
    y0 = np.array(C @ x0_new).flatten()
    
    # Store closed-loop data
    y_cl.append(y0)
    u_cl.append(u_current)
    acc_cl.append(acc_current)
    y_ref_cl.append(y_ref_current)
    y_ref_acc_cl.append(y_ref_acc_current)
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

print("=" * 90)
print(f"MPC simulation completed after {mpcctr} steps")

# Check if data was collected
if len(t) == 0:
    warnings.warn('No simulation steps run. Nothing to plot.')
    exit()

# Convert lists to numpy arrays
y_cl = np.array(y_cl).T  # ny x time steps
y_ref_cl = np.array(y_ref_cl).T  # ny x time steps
y_ref_acc_cl = np.array(y_ref_acc_cl).T  # ny x time steps
u_cl = np.array(u_cl)    # time steps x nu
acc_cl = np.array(acc_cl)  # time steps x nu
t = np.array(t)

# Control acceleration statistics
max_control_acc = np.max(np.abs(acc_cl))
avg_control_acc = np.mean(np.abs(acc_cl))
control_acc_violations = np.sum(np.abs(acc_cl) > acc_max[0])

# Reference trajectory acceleration statistics
max_ref_acc = np.max(np.abs(y_ref_acc_cl))
avg_ref_acc = np.mean(np.abs(y_ref_acc_cl))

# Tracking performance statistics
tracking_errors = np.linalg.norm(y_cl - y_ref_cl, axis=0)
rms_tracking_error = np.sqrt(np.mean(tracking_errors**2))
max_tracking_error = np.max(tracking_errors)

print(f"\nControl Acceleration Statistics:")
print(f"Maximum control acceleration: {max_control_acc:.3f} rad/s² (limit: {acc_max[0]:.1f})")
print(f"Average control acceleration: {avg_control_acc:.3f} rad/s²")
print(f"Control acceleration violations: {control_acc_violations} out of {len(t)} steps")

print(f"\nReference Trajectory Acceleration Statistics:")
print(f"Maximum reference acceleration: {max_ref_acc:.3f} rad/s²")
print(f"Average reference acceleration: {avg_ref_acc:.3f} rad/s²")
print(f"Reference acceleration limit satisfied: {max_ref_acc <= max_trajectory_acceleration}")

print(f"\nTracking Performance:")
print(f"RMS tracking error: {rms_tracking_error:.6f}")
print(f"Maximum tracking error: {max_tracking_error:.6f}")

if control_acc_violations > 0:
    print("WARNING: Control acceleration constraints were violated!")
else:
    print("✓ All control accelerations within limits")

if max_ref_acc <= max_trajectory_acceleration:
    print("✓ Reference trajectory accelerations within limits")
else:
    print("WARNING: Reference trajectory acceleration exceeds limits!")

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
    ax.plot(t, y_ref_cl[i, :], '--r', linewidth=1.2, label='Sinusoidal Reference', color='red')
    ax.legend()
    ax.set_title(f'Output y_{i+1} vs Sinusoidal Reference')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(f'y_{i+1}')
    ax.grid(True)

# Hide unused subplots
if ny < len(axes):
    for i in range(ny, len(axes)):
        axes[i].set_visible(False)

plt.tight_layout()
plt.show()

# Plot tracking errors
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
for i in range(ny):
    tracking_error_i = np.abs(y_cl[i, :] - y_ref_cl[i, :])
    plt.plot(t, tracking_error_i, linewidth=1.5, label=f'|y_{i+1} - ref_{i+1}|')
plt.title('Individual Output Tracking Errors')
plt.xlabel('Time [s]')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, tracking_errors, linewidth=2, color='black', label='Total Tracking Error (2-norm)')
plt.axhline(y=rms_tracking_error, color='r', linestyle='--', alpha=0.7, label=f'RMS Error = {rms_tracking_error:.6f}')
plt.title('Total Tracking Error (2-norm)')
plt.xlabel('Time [s]')
plt.ylabel('Tracking Error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot control inputs and accelerations
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

# Control inputs
for i in range(nu):
    ax1.plot(t, u_cl[:, i], linewidth=1.5, label=f'u_{i+1}')
    ax1.axhline(y=u_max[i], color='r', linestyle='--', alpha=0.5)
    ax1.axhline(y=u_min[i], color='r', linestyle='--', alpha=0.5)
ax1.set_title('Control Inputs')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Control Input')
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

# Reference trajectory accelerations
for i in range(ny):
    ax3.plot(t, y_ref_acc_cl[i, :], linewidth=1.5, label=f'Ref acc_{i+1}')
ax3.axhline(y=max_trajectory_acceleration, color='g', linestyle='--', linewidth=2, 
           label=f'Trajectory Acceleration Limit ({max_trajectory_acceleration})')
ax3.axhline(y=-max_trajectory_acceleration, color='g', linestyle='--', linewidth=2)
ax3.set_title('Reference Trajectory Accelerations (Second Derivatives)')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Reference Acceleration [rad/s²]')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()

print(f"\nSimulation completed in {mpcctr} steps")
print(f"Final simulation time: {t[-1]:.2f} seconds")