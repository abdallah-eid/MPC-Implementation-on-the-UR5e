import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import warnings
import time
from scipy.io import loadmat
import rtde_control
import rtde_receive

# UR5e Robot Connection Setup
ROBOT_IP = "192.168.1.10"  # Robot IP address

print("Connecting to UR5e robot...")
try:
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    print("Successfully connected to UR5e robot!")
except Exception as e:
    print(f"Failed to connect to robot: {e}")
    print("Please check the robot IP address and network connection.")
    exit()

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

T = 0.02     # Sampling time (same as robot control frequency)
N = 20      # Prediction horizon

# Input and output constraints (adjusted for UR5e)
u_max_full = np.array([2, 2, 2, 2, 2, 2])  # rad/s - UR5e joint velocity limits
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
Qy = 30 * np.eye(ny)
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
    'ipopt.max_iter': 30,
    'ipopt.print_level': 0,
    'print_time': False,
    'ipopt.acceptable_tol': 1e-3,
    'ipopt.acceptable_obj_change_tol': 1e-3
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

# Get initial robot state
print("Reading initial robot state...")
current_joint_positions = rtde_r.getActualQ()
print(f"Current robot joint positions: {current_joint_positions}")

# Initialize system state based on current robot position
y_initial = np.array(current_joint_positions[:ny])  # Use actual robot joint positions
try:
    x0 = np.linalg.pinv(C) @ y_initial
    print(f"Initial state calculated from robot position: {x0}")
    print(f"Verification - C*x0: {C @ x0}")
except:
    print("Warning: Could not calculate initial state using pinv(C). Using zero initial state.")
    x0 = np.zeros(nx)

# Calculate the initial robot output
y0_robot = C @ x0
print(f"Initial MPC output: {y0_robot}")

# Exponential trajectory parameters
sim_time = 5  # Total simulation time

# Define target position
y_targets = np.array([0.8, -0.5, 0.7, -1, 1.3, 2])[:ny]  # Target positions
y_start = y0_robot.copy()  # Starting positions

# Exponential time constants for each joint (controls how fast each joint reaches target)
tau = np.array([3.0, 4.0, 2.5, 3.5, 2.0, 4.5])[:ny]  # Time constants in seconds

# Maximum trajectory acceleration constraint
max_trajectory_acceleration = 0.4  # rad/s² - Keep below control acceleration limit

def generate_exponential_reference(t_current, horizon_steps):
    """Generate exponential reference trajectory over prediction horizon"""
    t_horizon = np.array([t_current + k * T for k in range(horizon_steps)])
    Yref = np.zeros((ny, horizon_steps))
    
    for i in range(ny):
        # Exponential approach: y(t) = y_target + (y_start - y_target) * exp(-t/tau)
        Yref[i, :] = y_targets[i] + (y_start[i] - y_targets[i]) * np.exp(-t_horizon / tau[i])
    
    return Yref

def calculate_exponential_acceleration(t_current, horizon_steps):
    """Calculate the acceleration of the exponential reference trajectory"""
    t_horizon = np.array([t_current + k * T for k in range(horizon_steps)])
    Yacc = np.zeros((ny, horizon_steps))
    
    for i in range(ny):
        # For y(t) = y_target + (y_start - y_target) * exp(-t/tau)
        # y'(t) = -(y_start - y_target) * (1/tau) * exp(-t/tau)
        # y''(t) = (y_start - y_target) * (1/tau²) * exp(-t/tau)
        Yacc[i, :] = (y_start[i] - y_targets[i]) * (1/tau[i]**2) * np.exp(-t_horizon / tau[i])
    
    return Yacc

# Verify trajectory acceleration constraints
print(f"\nExponential trajectory parameters:")
print(f"Start positions: {y_start}")
print(f"Target positions: {y_targets}")
print(f"Time constants: {tau} seconds")

max_traj_accs = []
for i in range(ny):
    # Maximum acceleration occurs at t=0 for exponential trajectories
    max_acc = abs(y_start[i] - y_targets[i]) / tau[i]**2
    max_traj_accs.append(max_acc)
    print(f"Joint {i+1}: Max trajectory acceleration = {max_acc:.3f} rad/s²")
    
    # Adjust time constant if acceleration is too high
    if max_acc > max_trajectory_acceleration:
        tau_min = np.sqrt(abs(y_start[i] - y_targets[i]) / max_trajectory_acceleration)
        tau[i] = max(tau[i], tau_min * 1.1)  # Add 10% safety margin
        max_acc_new = abs(y_start[i] - y_targets[i]) / tau[i]**2
        print(f"  → Adjusted time constant to {tau[i]:.2f}s, new max acc: {max_acc_new:.3f} rad/s²")
        max_traj_accs[i] = max_acc_new

print(f"Overall maximum trajectory acceleration: {max(max_traj_accs):.3f} rad/s²")

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

# Storage variables for logging
t0 = 0
mpcctr = 0
max_steps = int(sim_time / T)
y_cl = []
u_cl = []
acc_cl = []  # Store control acceleration data
y_ref_cl = []  # Store reference trajectory
y_ref_acc_cl = []  # Store reference trajectory accelerations
t_log = []
actual_robot_positions = []  # Store actual robot positions

# Safety check function
def safety_check():
    """Check robot safety conditions"""
    try:
        safety_status = rtde_r.getSafetyMode()
        robot_mode = rtde_r.getRobotMode()
        return safety_status == 1 and robot_mode == 7  # Normal operation
    except:
        return False

# Main real-time MPC loop
print("\nStarting real-time MPC control of UR5e robot...")
print("Press Ctrl+C to stop the control loop safely")
print("=" * 90)
print(f"{'Step':<6} {'Time':<8} {'Max Acc':<10} {'Tracking Error':<15} {'Solve Time':<12} {'Robot Status':<12}")
print("=" * 90)

try:
    while mpcctr < max_steps:
        loop_start_time = time.time()
        
        # Safety check
        if not safety_check():
            print("Robot safety check failed! Stopping control.")
            break
        
        # Get current robot joint positions
        current_q = rtde_r.getActualQ()
        y_actual = np.array(current_q[:ny])
        
        # Update system state (in practice, you might use a state estimator)
        try:
            x0 = np.linalg.pinv(C) @ y_actual
        except:
            # If state estimation fails, predict from previous state
            if mpcctr > 0:
                x0 = np.array(f_dynamics(x0, u_prev.reshape(-1, 1))).flatten()
        
        # Generate exponential reference trajectory over prediction horizon
        Yref = generate_exponential_reference(t0, N)
        
        # Current reference for tracking error calculation
        y_ref_current = generate_exponential_reference(t0, 1)[:, 0]
        
        # Calculate reference trajectory acceleration for analysis
        y_ref_acc_current = calculate_exponential_acceleration(t0, 1)[:, 0]
        
        # Set parameters for solver
        p = np.concatenate([x0.flatten(), Yref.flatten(order='F'), u_prev.flatten()])
        x0_opt = u0.flatten(order='F')
        
        # Solve NLP
        solve_start_time = time.time()
        try:
            sol = solver(x0=x0_opt, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)
            solve_time = time.time() - solve_start_time
            
            # Check solver status
            solver_stats = solver.stats()
            if not solver_stats['success']:
                print(f"Warning: Solver did not converge at step {mpcctr}")
                
        except Exception as e:
            print(f"Solver failed at step {mpcctr}: {e}")
            # Use previous control input in case of failure
            u_current = u_prev
            solve_time = time.time() - solve_start_time
        else:
            # Extract first control input
            u_opt_flat = np.array(sol['x']).flatten()
            u_opt = u_opt_flat.reshape((nu, N), order='F').T
            u_current = u_opt[0, :]
        
        # Calculate current acceleration
        acc_current = (u_current - u_prev) / T
        max_acc = np.max(np.abs(acc_current))
        
        # Send velocity command to robot using speedJ
        # speedJ(velocities, acceleration, time)
        rtde_c.speedJ(u_current.tolist(), max_acc, T)
        
        # Calculate tracking error
        tracking_error = np.linalg.norm(y_actual - y_ref_current, 2)
        
        # Store data for logging
        y_cl.append(y_actual)
        u_cl.append(u_current)
        acc_cl.append(acc_current)
        y_ref_cl.append(y_ref_current)
        y_ref_acc_cl.append(y_ref_acc_current)
        t_log.append(t0)
        actual_robot_positions.append(current_q)
        
        # Print progress
        if mpcctr % 50 == 0:
            robot_status = "OK" if safety_check() else "WARNING"
            print(f"{mpcctr:<6} {t0:<8.2f} {max_acc:<10.3f} {tracking_error:<15.6f} {solve_time*1000:<12.1f} {robot_status:<12}")
            
            # Check if acceleration constraint is active
            if max_acc > 0.45:  # Close to limit
                print(f"    → High acceleration detected: {max_acc:.3f} rad/s²")
        
        # Update for next iteration
        u_prev = u_current
        t0 = t0 + T
        u0 = np.vstack([u_opt[1:, :], u_opt[-1, :].reshape(1, -1)]) if 'u_opt' in locals() else u0
        mpcctr = mpcctr + 1
        
        # Maintain real-time loop timing
        loop_time = time.time() - loop_start_time
        if loop_time < T:
            time.sleep(T - loop_time)
     

except KeyboardInterrupt:
    print("\nControl interrupted by user")
except Exception as e:
    print(f"\nUnexpected error: {e}")
finally:
    # Stop robot motion safely
    print("Stopping robot motion...")
    try:
        rtde_c.speedStop()
        time.sleep(0.1)
        rtde_c.stopScript()
        print("Robot stopped successfully")
    except Exception as e:
        print(f"Error stopping robot: {e}")

print("=" * 90)
print(f"Real-time MPC control completed after {mpcctr} steps")

# Check if data was collected
if len(t_log) == 0:
    warnings.warn('No control steps executed. Nothing to analyze.')
    exit()

# Convert lists to numpy arrays for analysis
y_cl = np.array(y_cl).T  # ny x time steps
y_ref_cl = np.array(y_ref_cl).T  # ny x time steps
y_ref_acc_cl = np.array(y_ref_acc_cl).T  # ny x time steps
u_cl = np.array(u_cl)    # time steps x nu
acc_cl = np.array(acc_cl)  # time steps x nu
t_log = np.array(t_log)
actual_robot_positions = np.array(actual_robot_positions)

# Statistics
max_control_acc = np.max(np.abs(acc_cl))
avg_control_acc = np.mean(np.abs(acc_cl))
control_acc_violations = np.sum(np.abs(acc_cl) > acc_max[0])

max_ref_acc = np.max(np.abs(y_ref_acc_cl))
avg_ref_acc = np.mean(np.abs(y_ref_acc_cl))

tracking_errors = np.linalg.norm(y_cl - y_ref_cl, axis=0)
rms_tracking_error = np.sqrt(np.mean(tracking_errors**2))
max_tracking_error = np.max(tracking_errors)

print(f"\nReal-time Control Performance Statistics:")
print(f"Maximum control acceleration: {max_control_acc:.3f} rad/s² (limit: {acc_max[0]:.1f})")
print(f"Average control acceleration: {avg_control_acc:.3f} rad/s²")
print(f"Control acceleration violations: {control_acc_violations} out of {len(t_log)} steps")

print(f"\nReference Trajectory Acceleration Statistics:")
print(f"Maximum reference acceleration: {max_ref_acc:.3f} rad/s²")
print(f"Average reference acceleration: {avg_ref_acc:.3f} rad/s²")

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
print("REAL-TIME CONTROL SUMMARY:")
if control_acc_violations == 0:
    print("✓ All control accelerations within limits")
else:
    print(f"⚠ Control acceleration violations: {control_acc_violations}")

if max_ref_acc <= max_trajectory_acceleration:
    print("✓ Reference trajectory accelerations within limits")
else:
    print("⚠ Reference trajectory acceleration exceeds limits")

print(f"✓ Real-time control executed successfully")
print(f"{'='*50}")

# Plot results
print("Generating plots...")
fig, axes = plt.subplots(int(np.ceil(ny/2)), 2, figsize=(15, 10))
if ny == 1:
    axes = [axes]
elif ny <= 2:
    axes = axes.reshape(-1)
else:
    axes = axes.flatten()

for i in range(ny):
    ax = axes[i] if ny > 1 else axes[0]
    ax.plot(t_log, y_cl[i, :], linewidth=1.5, label='Actual Robot Position', color='blue')
    ax.plot(t_log, y_ref_cl[i, :], '--r', linewidth=1.2, label='Reference', color='red')
    ax.legend()
    ax.set_title(f'Joint {i+1}: Actual vs Reference Position')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(f'Joint {i+1} Position [rad]')
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
    ax1.plot(t_log, tracking_error_i, linewidth=1.5, label=f'|Joint {i+1} Error|')
ax1.set_title('Individual Joint Tracking Errors')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Absolute Error [rad]')
ax1.legend()
ax1.grid(True)

# Total tracking error
ax2.plot(t_log, tracking_errors, linewidth=2, color='black', label='Total Tracking Error (2-norm)')
ax2.axhline(y=rms_tracking_error, color='r', linestyle='--', alpha=0.7, label=f'RMS Error = {rms_tracking_error:.6f}')
ax2.set_title('Total Tracking Error (2-norm)')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Tracking Error')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Plot control inputs and accelerations
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

# Control inputs (velocities sent to robot)
for i in range(nu):
    ax1.plot(t_log, u_cl[:, i], linewidth=1.5, label=f'Joint {i+1} Velocity')
    ax1.axhline(y=u_max[i], color='r', linestyle='--', alpha=0.5)
    ax1.axhline(y=u_min[i], color='r', linestyle='--', alpha=0.5)
ax1.set_title('Joint Velocities Sent to Robot')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Joint Velocity [rad/s]')
ax1.legend()
ax1.grid(True)

# Control accelerations
for i in range(nu):
    ax2.plot(t_log, acc_cl[:, i], linewidth=1.5, label=f'Joint {i+1} Acceleration')
ax2.axhline(y=acc_max[0], color='r', linestyle='--', linewidth=2, label='Acceleration Limit')
ax2.axhline(y=acc_min[0], color='r', linestyle='--', linewidth=2)
ax2.set_title('Joint Accelerations')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Joint Acceleration [rad/s²]')
ax2.legend()
ax2.grid(True)

# Reference trajectory accelerations
for i in range(ny):
    ax3.plot(t_log, y_ref_acc_cl[i, :], linewidth=1.5, label=f'Ref Joint {i+1} Acc')
ax3.axhline(y=max_trajectory_acceleration, color='g', linestyle='--', linewidth=2, 
           label=f'Trajectory Acceleration Limit ({max_trajectory_acceleration})')
ax3.axhline(y=-max_trajectory_acceleration, color='g', linestyle='--', linewidth=2)
ax3.set_title('Reference Trajectory Accelerations')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Reference Acceleration [rad/s²]')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()

print("Real-time control and analysis completed successfully.")

# Save results for later analysis
try:
    np.savez('ur5e_mpc_results.npz', 
             t=t_log, 
             y_cl=y_cl, 
             y_ref_cl=y_ref_cl, 
             u_cl=u_cl, 
             acc_cl=acc_cl,
             y_ref_acc_cl=y_ref_acc_cl,
             actual_robot_positions=actual_robot_positions)
    print("Results saved to 'ur5e_mpc_results.npz'")
except Exception as e:
    print(f"Error saving results: {e}")

print("All operations completed successfully!")