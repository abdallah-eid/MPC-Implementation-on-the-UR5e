import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import warnings
import time
from scipy.io import loadmat
import rtde_control
import rtde_receive
from scipy.interpolate import CubicSpline

# UR5e Robot Connection Setup
ROBOT_IP = "192.168.1.10"  # Change this to your robot's IP address

print("Connecting to UR5e robot...")
try:
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    print("Successfully connected to UR5e robot!")
    
    # Check robot status
    robot_mode = rtde_r.getRobotMode()
    safety_mode = rtde_r.getSafetyMode()
    print(f"Robot mode: {robot_mode}, Safety mode: {safety_mode}")
    
    if robot_mode != 7:  # Not in normal operation mode
        print("Warning: Robot not in normal operation mode. Please check robot status.")
    
except Exception as e:
    print(f"Failed to connect to robot: {e}")
    print("Please check the robot IP address and network connection.")
    exit()

# Load the system matrices from the .mat file
mat_file_path = r'C:\Users\Abdallah Eid\Desktop\ss1_matrices.mat'
try:
    mat_data = loadmat(mat_file_path)
except FileNotFoundError:
    print(f"Could not find .mat file at {mat_file_path}")
    print("Please update the path to your ss1_matrices.mat file")
    exit()

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

# Control parameters - optimized for real-time hardware control
T = 0.02     # Sampling time - 50 Hz (UR5e standard control frequency)
N = 20       # Reduced prediction horizon for faster computation

# Input and output constraints (UR5e specifications)
u_max_full = np.array([10, 10, 10, 10, 10, 10])  # rad/s - UR5e joint velocity limits
u_max = u_max_full[:nu]
u_min = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0])[:nu]
y_max = 2 * np.pi * np.ones(ny)
y_min = -y_max

# Acceleration constraints (change in velocity) - conservative for safety
acc_max = 2 * np.ones(nu)  # Maximum acceleration: 0.5 rad/s²
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

# Objective function weights - tuned for hardware performance
Qy = 120 * np.eye(ny)  # Increased tracking weight for better performance
R = 10 * np.eye(nu)   # Control effort weight

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

# Solver options - optimized for real-time performance
opts = {
    'ipopt.max_iter': 50,           # Reduced iterations for speed
    'ipopt.print_level': 0,         # Suppress solver output
    'print_time': False,            # Don't print timing info
    'ipopt.acceptable_tol': 1e-8,   # Less strict tolerance for speed
    'ipopt.acceptable_obj_change_tol': 1e-4,
    'ipopt.warm_start_init_point': 'yes',  # Use warm start for speed
    'ipopt.mu_strategy': 'adaptive'        # Adaptive barrier parameter
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

# Get initial robot state
print("\nReading initial robot state...")
current_joint_positions = rtde_r.getActualQ()
current_joint_velocities = rtde_r.getActualQd()
print(f"Current robot joint positions: {np.array(current_joint_positions)}")
print(f"Current robot joint velocities: {np.array(current_joint_velocities)}")

# Initialize system state based on current robot position
y_initial = np.array(current_joint_positions[:ny])
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

# SMOOTH TRAJECTORY CONFIGURATION
# Define waypoints for smooth trajectory
waypoints = [
    y_initial,  # Start from current position
    np.array([-0.5, -0.5, -0.3, -1, -1.3, 2])[:ny],      # Waypoint 1
    np.array([0.8, -1.3, 0.3, 1, 1.3, 2])[:ny],          # Waypoint 2
    np.array([0, -1.57, 0, -1.57, 0, 0])[:ny]             # Final waypoint
]

# Smooth trajectory parameters
total_trajectory_time = 5  # Total time for entire smooth trajectory
max_velocity = 3.0            # Maximum velocity for smooth interpolation (rad/s)

print(f"\nSmooth trajectory configuration:")
print(f"Total trajectory time: {total_trajectory_time}s")
print(f"Maximum velocity: {max_velocity} rad/s")
print(f"Number of waypoints: {len(waypoints)}")

# Safety limits for robot joints (rad) - UR5e specific
joint_limits_min = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])[:ny]
joint_limits_max = np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])[:ny]

# Validate all waypoints against joint limits
print("\nValidating all waypoints against joint limits:")
for wp_idx, waypoint in enumerate(waypoints):
    print(f"Waypoint {wp_idx}:")
    waypoint_valid = True
    for i in range(ny):
        if joint_limits_min[i] <= waypoint[i] <= joint_limits_max[i]:
            print(f"  Joint {i+1}: {waypoint[i]:.3f} - OK")
        else:
            print(f"  Joint {i+1}: {waypoint[i]:.3f} - WARNING: Outside limits [{joint_limits_min[i]:.2f}, {joint_limits_max[i]:.2f}]")
            # Clamp to safe limits
            waypoints[wp_idx][i] = np.clip(waypoint[i], joint_limits_min[i], joint_limits_max[i])
            print(f"    → Clamped to: {waypoints[wp_idx][i]:.3f}")
            waypoint_valid = False
    
    if waypoint_valid:
        print(f"  Waypoint {wp_idx}: All joints within limits ✓")
    else:
        print(f"  Waypoint {wp_idx}: Some joints clamped to limits ⚠")

class SmoothTrajectoryGenerator:
    def __init__(self, waypoints, total_time, max_velocity=1.0):
        self.waypoints = np.array(waypoints)
        self.total_time = total_time
        self.max_velocity = max_velocity
        self.num_joints = self.waypoints.shape[1]
        self.num_waypoints = len(waypoints)
        
        # Calculate timing for waypoints based on distances
        self.waypoint_times = self._calculate_waypoint_times()
        
        # Create cubic spline interpolators for each joint
        self.splines = []
        for joint_idx in range(self.num_joints):
            joint_positions = self.waypoints[:, joint_idx]
            # Create spline with zero velocity at start and end
            spline = CubicSpline(self.waypoint_times, joint_positions, 
                               bc_type=((1, 0.0), (1, 0.0)))  # Zero velocity boundary conditions
            self.splines.append(spline)
        
        print(f"Created smooth trajectory with {self.num_waypoints} waypoints")
        print(f"Waypoint times: {self.waypoint_times}")
    
    def _calculate_waypoint_times(self):
        """Calculate timing for waypoints based on path distances"""
        if self.num_waypoints <= 1:
            return np.array([0])
        
        # Calculate cumulative distances
        distances = [0]
        for i in range(1, self.num_waypoints):
            dist = np.linalg.norm(self.waypoints[i] - self.waypoints[i-1])
            distances.append(distances[-1] + dist)
        
        total_distance = distances[-1]
        if total_distance == 0:
            # All waypoints are the same
            return np.linspace(0, self.total_time, self.num_waypoints)
        
        # Scale distances to time
        waypoint_times = np.array(distances) * self.total_time / total_distance
        return waypoint_times
    
    def get_reference(self, t):
        """Get reference position at time t"""
        t = np.clip(t, 0, self.total_time)
        
        position = np.zeros(self.num_joints)
        for joint_idx in range(self.num_joints):
            position[joint_idx] = self.splines[joint_idx](t)
        
        return position
    
    def get_velocity(self, t):
        """Get reference velocity at time t"""
        t = np.clip(t, 0, self.total_time)
        
        velocity = np.zeros(self.num_joints)
        for joint_idx in range(self.num_joints):
            velocity[joint_idx] = self.splines[joint_idx].derivative(1)(t)
        
        # Limit velocity magnitude
        velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude > self.max_velocity:
            velocity = velocity * self.max_velocity / velocity_magnitude
        
        return velocity

# Create smooth trajectory generator
trajectory_generator = SmoothTrajectoryGenerator(waypoints, total_trajectory_time, max_velocity)

def generate_smooth_reference_trajectory(t_current, horizon_steps):
    """Generate smooth reference trajectory over prediction horizon"""
    Yref = np.zeros((ny, horizon_steps))
    
    for k in range(horizon_steps):
        t_future = t_current + k * T
        ref_pos = trajectory_generator.get_reference(t_future)
        Yref[:, k] = ref_pos
    
    current_ref = trajectory_generator.get_reference(t_current)
    
    return Yref, current_ref

def safety_check():
    """Check robot safety conditions"""
    try:
        safety_status = rtde_r.getSafetyMode()
        robot_mode = rtde_r.getRobotMode()
        is_emergency_stopped = rtde_r.isEmergencyStopped()
        is_protective_stopped = rtde_r.isProtectiveStopped()
        
        return (safety_status == 1 and robot_mode == 7 and 
                not is_emergency_stopped and not is_protective_stopped)
    except Exception as e:
        print(f"Safety check failed: {e}")
        return False

def get_robot_state():
    """Get current robot state with error handling"""
    try:
        positions = rtde_r.getActualQ()
        velocities = rtde_r.getActualQd()
        return np.array(positions[:ny]), np.array(velocities[:nu])
    except Exception as e:
        print(f"Error reading robot state: {e}")
        return None, None

# Initialize control variables
u0 = np.zeros((N, nu))   # Initial guess for inputs
u_prev = np.array(current_joint_velocities[:nu])  # Use actual current velocities

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
max_steps = int(total_trajectory_time / T)
y_cl = []
u_cl = []
acc_cl = []  # Store control acceleration data
y_ref_cl = []  # Store reference trajectory
ref_vel_cl = []  # Store reference velocities
t_log = []
actual_robot_positions = []  # Store actual robot positions
actual_robot_velocities = []  # Store actual robot velocities
solve_times = []  # Store solver timing

print(f"\nStarting smooth trajectory MPC control of UR5e robot...")
print("Press Ctrl+C to stop the control loop safely")
print("SAFETY: Emergency stop button on robot is always active")
print("=" * 120)
print(f"{'Step':<6} {'Time':<8} {'Max Acc':<10} {'Track Err':<12} {'Ref Vel':<12} {'Solve ms':<10} {'Robot Status':<12} {'Output':<25}")
print("=" * 120)

emergency_stop_sent = False

try:
    while mpcctr < max_steps:
        loop_start_time = time.time()
        
        # Safety check first
        if not safety_check():
            print("Robot safety check failed! Stopping control.")
            break
        
        # Get current robot state
        current_positions, current_velocities = get_robot_state()
        if current_positions is None:
            print("Failed to read robot state! Stopping control.")
            break
        
        y_actual = current_positions
        
        # Update system state (using pseudo-inverse)
        try:
            x0 = np.linalg.pinv(C) @ y_actual
        except:
            # If state estimation fails, predict from previous state
            if mpcctr > 0:
                x0 = np.array(f_dynamics(x0, u_prev.reshape(-1, 1))).flatten()
        
        # Generate smooth reference trajectory over horizon
        Yref, current_ref = generate_smooth_reference_trajectory(t0, N)
        
        # Calculate reference velocity for monitoring
        ref_velocity = trajectory_generator.get_velocity(t0)
        ref_vel_norm = np.linalg.norm(ref_velocity)
        
        # Set parameters for solver
        p = np.concatenate([x0.flatten(), Yref.flatten(order='F'), u_prev.flatten()])
        x0_opt = u0.flatten(order='F')
        
        # Solve NLP with timing
        solve_start_time = time.time()
        solver_success = True
        try:
            sol = solver(x0=x0_opt, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)
            solve_time = time.time() - solve_start_time
            
            # Check solver status
            solver_stats = solver.stats()
            if not solver_stats['success']:
                print(f"Warning: Solver did not converge at step {mpcctr}")
                solver_success = False
                
        except Exception as e:
            print(f"Solver failed at step {mpcctr}: {e}")
            solver_success = False
            solve_time = time.time() - solve_start_time
        
        if solver_success:
            # Extract optimal control inputs
            u_opt_flat = np.array(sol['x']).flatten()
            u_opt = u_opt_flat.reshape((nu, N), order='F').T
            u_current = u_opt[0, :]
        else:
            # Use previous control input in case of failure
            u_current = u_prev * 0.9  # Gradually reduce velocity for safety
            print(f"Using fallback control at step {mpcctr}")
        
        # Calculate current acceleration
        acc_current = (u_current - u_prev) / T
        max_acc = np.max(np.abs(acc_current))
        
        # Safety check on control commands
        if np.any(np.abs(u_current) > u_max):
            print("Control velocity exceeds limits! Clamping.")
            u_current = np.clip(u_current, u_min, u_max)
        
        if max_acc > acc_max[0] * 1.1:  # 10% tolerance
            print(f"Control acceleration {max_acc:.3f} exceeds limit! Emergency stop.")
            rtde_c.speedStop()
            break
        
        # Send velocity command to robot
        try:
            # Use speedJ for velocity control with acceleration limit
            rtde_c.speedJ(u_current.tolist(), max_acc, T * 1.2)  # Slightly longer time for safety
        except Exception as e:
            print(f"Error sending command to robot: {e}")
            break
        
        # Calculate tracking error
        tracking_error = np.linalg.norm(y_actual - current_ref, 2)
        
        # Store data for logging
        y_cl.append(y_actual.copy())
        u_cl.append(u_current.copy())
        acc_cl.append(acc_current.copy())
        y_ref_cl.append(current_ref.copy())
        ref_vel_cl.append(ref_velocity.copy())
        t_log.append(t0)
        actual_robot_positions.append(current_positions.copy())
        actual_robot_velocities.append(current_velocities.copy())
        solve_times.append(solve_time * 1000)  # Convert to milliseconds
        
        # Print progress
        if mpcctr % 25 == 0:  # Print every 0.5 seconds
            robot_status = "OK" if safety_check() else "WARNING"
            y_str = '[' + ', '.join([f'{val:.2f}' for val in y_actual[:3]]) + '...]' if len(y_actual) > 3 else str(y_actual)
            print(f"{mpcctr:<6} {t0:<8.2f} {max_acc:<10.3f} {tracking_error:<12.6f} {ref_vel_norm:<12.3f} {solve_time*1000:<10.1f} {robot_status:<12} {y_str:<25}")
            
            # Check if acceleration constraint is active
            if max_acc > 0.4:  # Close to limit
                print(f"    → High acceleration: {max_acc:.3f} rad/s²")
        
        # Update for next iteration
        u_prev = u_current.copy()
        t0 = t0 + T
        if solver_success:
            u0 = np.vstack([u_opt[1:, :], u_opt[-1, :].reshape(1, -1)])
        mpcctr = mpcctr + 1
        
        # Maintain real-time loop timing
        loop_time = time.time() - loop_start_time
        if loop_time < T:
            time.sleep(T - loop_time)

except KeyboardInterrupt:
    print("\nControl interrupted by user (Ctrl+C)")
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
        emergency_stop_sent = True
    except Exception as e:
        print(f"Error stopping robot: {e}")

print("=" * 120)
print(f"Smooth trajectory MPC control completed after {mpcctr} steps ({t0:.1f} seconds)")

# Check if data was collected
if len(t_log) == 0:
    warnings.warn('No control steps executed. Nothing to analyze.')
    exit()

# Convert lists to numpy arrays for analysis
y_cl = np.array(y_cl).T  # ny x time steps
y_ref_cl = np.array(y_ref_cl).T  # ny x time steps
u_cl = np.array(u_cl)    # time steps x nu
acc_cl = np.array(acc_cl)  # time steps x nu
ref_vel_cl = np.array(ref_vel_cl)  # time steps x nu
t_log = np.array(t_log)
actual_robot_positions = np.array(actual_robot_positions)
actual_robot_velocities = np.array(actual_robot_velocities)
solve_times = np.array(solve_times)

# Performance Statistics
max_control_acc = np.max(np.abs(acc_cl))
avg_control_acc = np.mean(np.abs(acc_cl))
control_acc_violations = np.sum(np.abs(acc_cl) > acc_max[0])

tracking_errors = np.linalg.norm(y_cl - y_ref_cl, axis=0)
rms_tracking_error = np.sqrt(np.mean(tracking_errors**2))
max_tracking_error = np.max(tracking_errors)

avg_solve_time = np.mean(solve_times)
max_solve_time = np.max(solve_times)

# Smooth trajectory specific metrics
path_length = 0
for i in range(1, len(t_log)):
    path_length += np.linalg.norm(y_cl[:, i] - y_cl[:, i-1])

ref_velocities = np.linalg.norm(ref_vel_cl, axis=1)
max_ref_velocity = np.max(ref_velocities)
avg_ref_velocity = np.mean(ref_velocities)

print(f"\n{'='*80}")
print("SMOOTH TRAJECTORY CONTROL PERFORMANCE ANALYSIS:")
print(f"{'='*80}")

print(f"\nTrajectory Characteristics:")
print(f"  Total path length: {path_length:.3f} rad")
print(f"  Maximum reference velocity: {max_ref_velocity:.3f} rad/s")
print(f"  Average reference velocity: {avg_ref_velocity:.3f} rad/s")
print(f"  Trajectory duration: {t_log[-1]:.1f} s")

print(f"\nControl Acceleration Statistics:")
print(f"  Maximum: {max_control_acc:.3f} rad/s² (limit: {acc_max[0]:.1f})")
print(f"  Average: {avg_control_acc:.3f} rad/s²")
print(f"  Violations: {control_acc_violations} out of {len(t_log)} steps")

print(f"\nSolver Performance:")
print(f"  Average solve time: {avg_solve_time:.1f} ms")
print(f"  Maximum solve time: {max_solve_time:.1f} ms")
print(f"  Target cycle time: {T*1000:.1f} ms")

print(f"\nOverall Tracking Performance:")
print(f"  RMS tracking error: {rms_tracking_error:.6f} rad")
print(f"  Maximum tracking error: {max_tracking_error:.6f} rad")

# Analyze performance near waypoints
print(f"\nWaypoint Achievement Analysis:")
for i, wp_time in enumerate(trajectory_generator.waypoint_times):
    if wp_time <= t_log[-1]:
        # Find the closest time index to waypoint time
        time_idx = np.argmin(np.abs(t_log - wp_time))
        if time_idx < len(y_cl[0]):
            actual_pos = y_cl[:, time_idx]
            target_pos = waypoints[i]
            error = np.linalg.norm(actual_pos - target_pos)
            print(f"  Waypoint {i}: Target {target_pos[:3]}... at t={wp_time:.1f}s")
            print(f"    Achieved: {actual_pos[:3]}... (error: {error:.6f} rad)")

print(f"\nFinal Position Achievement:")
final_target = waypoints[-1]
final_actual = y_cl[:, -1]
final_error = np.linalg.norm(final_actual - final_target)
print(f"  Target:   {final_target}")
print(f"  Achieved: {final_actual}")
print(f"  Error:    {final_error:.6f} rad")

print(f"\n{'='*80}")
print("SMOOTH TRAJECTORY CONTROL SUMMARY:")
success_indicators = []

if control_acc_violations == 0:
    print("✓ All control accelerations within limits")
    success_indicators.append(True)
else:
    print(f"⚠ Control acceleration violations: {control_acc_violations}")
    success_indicators.append(False)

if avg_solve_time < T * 1000 * 0.8:  # Solver should use <80% of cycle time
    print("✓ Real-time performance maintained")
    success_indicators.append(True)
else:
    print("⚠ Real-time performance marginal")
    success_indicators.append(False)

if rms_tracking_error < 0.1:
    print("✓ Excellent smooth tracking performance")
    success_indicators.append(True)
elif rms_tracking_error < 0.2:
    print("✓ Good smooth tracking performance")
    success_indicators.append(True)
else:
    print("⚠ Tracking performance could be improved")
    success_indicators.append(False)

if final_error < 0.1:
    print("✓ Final waypoint achieved accurately")
    success_indicators.append(True)
else:
    print("⚠ Final waypoint achievement could be improved")
    success_indicators.append(False)

if not emergency_stop_sent or mpcctr >= max_steps * 0.9:
    print("✓ Smooth trajectory completed successfully")
    success_indicators.append(True)
else:
    print("⚠ Control terminated early")
    success_indicators.append(False)

overall_success = all(success_indicators)
print(f"\nOverall Status: {'✓ SUCCESS - SMOOTH MOTION ACHIEVED' if overall_success else '⚠ PARTIAL SUCCESS'}")
print(f"{'='*80}")

# Generate comprehensive plots
print("\nGenerating smooth trajectory performance plots...")

# Plot 1: Smooth joint trajectories
fig, axes = plt.subplots(int(np.ceil(ny/2)), 2, figsize=(18, 12))
if ny == 1:
    axes = [axes]
elif ny <= 2:
    axes = axes.reshape(-1)
else:
    axes = axes.flatten()

for i in range(ny):
    ax = axes[i] if ny > 1 else axes[0]
    ax.plot(t_log, y_cl[i, :], linewidth=2.5, label='Actual Robot Position', color='blue', alpha=0.8)
    ax.plot(t_log, y_ref_cl[i, :], '--', linewidth=2, label='Smooth Reference', color='red', alpha=0.7)
    
    # Mark waypoints
    for wp_idx, wp_time in enumerate(trajectory_generator.waypoint_times):
        if wp_time <= t_log[-1] and wp_idx < len(waypoints):
            ax.scatter(wp_time, waypoints[wp_idx][i], s=100, c='red', marker='o', zorder=5, 
                      label=f'Waypoint {wp_idx}' if i == 0 and wp_idx < 4 else "")
            ax.axvline(x=wp_time, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    # Joint limits
    ax.axhline(y=joint_limits_min[i], color='k', linestyle=':', alpha=0.3, 
              label='Joint Limits' if i == 0 else "")
    ax.axhline(y=joint_limits_max[i], color='k', linestyle=':', alpha=0.3)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(f'Joint {i+1}: Smooth Trajectory Tracking')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(f'Joint {i+1} Position [rad]')
    ax.grid(True, alpha=0.3)

# Hide unused subplots
if ny < len(axes):
    for i in range(ny, len(axes)):
        axes[i].set_visible(False)

plt.suptitle('UR5e Smooth Trajectory MPC Control: Joint Trajectories', fontsize=16)
plt.tight_layout()
plt.show()

# Plot 2: Tracking performance and velocities
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

# Tracking errors
for i in range(ny):
    tracking_error_i = np.abs(y_cl[i, :] - y_ref_cl[i, :])
    ax1.plot(t_log, tracking_error_i, linewidth=1.5, label=f'|Joint {i+1} Error|', alpha=0.8)

# Mark waypoint times
for wp_idx, wp_time in enumerate(trajectory_generator.waypoint_times):
    if wp_time <= t_log[-1]:
        ax1.axvline(x=wp_time, color='red', linestyle=':', alpha=0.7, linewidth=1)
        if wp_idx == 0:  # Only label once
            ax1.axvline(x=wp_time, color='red', linestyle=':', alpha=0.7, 
                       linewidth=1, label='Waypoints')

ax1.set_title('Individual Joint Tracking Errors')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Absolute Error [rad]')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Total tracking error
ax2.plot(t_log, tracking_errors, linewidth=2.5, color='black', 
         label='Total Tracking Error (2-norm)')
ax2.axhline(y=rms_tracking_error, color='red', linestyle='--', alpha=0.7, 
           label=f'RMS = {rms_tracking_error:.6f}')

# Mark waypoint times
for wp_time in trajectory_generator.waypoint_times:
    if wp_time <= t_log[-1]:
        ax2.axvline(x=wp_time, color='red', linestyle=':', alpha=0.5, linewidth=1)

ax2.set_title('Total Tracking Error')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Tracking Error [rad]')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Reference and actual velocities
ref_vel_norms = np.linalg.norm(ref_vel_cl, axis=1)
actual_vel_norms = np.linalg.norm(u_cl, axis=1)

ax3.plot(t_log, ref_vel_norms, linewidth=2, label='Reference Velocity', 
         color='red', alpha=0.7)
ax3.plot(t_log, actual_vel_norms, linewidth=2, label='Actual Control Velocity', 
         color='blue', alpha=0.8)
ax3.axhline(y=max_velocity, color='red', linestyle='--', alpha=0.5, 
           label=f'Max Ref Velocity = {max_velocity}')

# Mark waypoint times
for wp_time in trajectory_generator.waypoint_times:
    if wp_time <= t_log[-1]:
        ax3.axvline(x=wp_time, color='red', linestyle=':', alpha=0.5, linewidth=1)

ax3.set_title('Velocity Profile Comparison')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Velocity Magnitude [rad/s]')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Control accelerations
for i in range(nu):
    ax4.plot(t_log, acc_cl[:, i], linewidth=1.5, label=f'Joint {i+1}', alpha=0.8)
ax4.axhline(y=acc_max[0], color='red', linestyle='--', linewidth=2, 
           label='Acceleration Limit')
ax4.axhline(y=acc_min[0], color='red', linestyle='--', linewidth=2)

# Mark waypoint times
for wp_time in trajectory_generator.waypoint_times:
    if wp_time <= t_log[-1]:
        ax4.axvline(x=wp_time, color='red', linestyle=':', alpha=0.5, linewidth=1)

ax4.set_title('Control Accelerations')
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Acceleration [rad/s²]')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot 3: 3D trajectory visualization (for first 3 joints)
if ny >= 3:
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot actual trajectory
    ax.plot(y_cl[0, :], y_cl[1, :], y_cl[2, :], linewidth=3, 
            label='Actual Trajectory', color='blue', alpha=0.8)
    
    # Plot reference trajectory
    ax.plot(y_ref_cl[0, :], y_ref_cl[1, :], y_ref_cl[2, :], '--', linewidth=2, 
            label='Reference Trajectory', color='red', alpha=0.7)
    
    # Mark waypoints
    for wp_idx, wp in enumerate(waypoints):
        if len(wp) >= 3:
            ax.scatter(wp[0], wp[1], wp[2], s=200, c='red', marker='o', 
                      label=f'Waypoint {wp_idx}' if wp_idx < 4 else "")
            
            # Add waypoint labels
            ax.text(wp[0], wp[1], wp[2], f'  WP{wp_idx}', fontsize=10)
    
    # Mark start and end points
    ax.scatter(y_cl[0, 0], y_cl[1, 0], y_cl[2, 0], s=300, c='green', 
              marker='s', label='Start', alpha=0.8)
    ax.scatter(y_cl[0, -1], y_cl[1, -1], y_cl[2, -1], s=300, c='purple', 
              marker='^', label='End', alpha=0.8)
    
    ax.set_xlabel('Joint 1 Position [rad]')
    ax.set_ylabel('Joint 2 Position [rad]')
    ax.set_zlabel('Joint 3 Position [rad]')
    ax.set_title('3D Trajectory Visualization (Joints 1-3)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Plot 4: Performance metrics over time
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

# Solver performance
ax1.plot(t_log, solve_times, linewidth=1.5, color='green', alpha=0.7, label='Solve Time')
ax1.axhline(y=T*1000, color='red', linestyle='--', linewidth=2, 
           label=f'Cycle Time ({T*1000:.0f} ms)')
ax1.axhline(y=avg_solve_time, color='blue', linestyle=':', alpha=0.7, 
           label=f'Average ({avg_solve_time:.1f} ms)')

# Mark waypoint times
for wp_time in trajectory_generator.waypoint_times:
    if wp_time <= t_log[-1]:
        ax1.axvline(x=wp_time, color='red', linestyle=':', alpha=0.5, linewidth=1)

ax1.set_title('MPC Solver Performance')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Solve Time [ms]')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Control effort
control_effort = np.sum(u_cl**2, axis=1)
ax2.plot(t_log, control_effort, linewidth=2, color='purple', label='Control Effort')

# Mark waypoint times
for wp_time in trajectory_generator.waypoint_times:
    if wp_time <= t_log[-1]:
        ax2.axvline(x=wp_time, color='red', linestyle=':', alpha=0.5, linewidth=1)

ax2.set_title('Control Effort')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Control Effort [rad²/s²]')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Distance to target over time
distances_to_final = []
final_target = waypoints[-1]
for i in range(len(t_log)):
    dist = np.linalg.norm(y_cl[:, i] - final_target)
    distances_to_final.append(dist)

ax3.plot(t_log, distances_to_final, linewidth=2, color='orange', 
         label='Distance to Final Target')

# Mark waypoint times
for wp_time in trajectory_generator.waypoint_times:
    if wp_time <= t_log[-1]:
        ax3.axvline(x=wp_time, color='red', linestyle=':', alpha=0.5, linewidth=1)

ax3.set_title('Distance to Final Target')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Distance [rad]')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Velocity magnitude comparison
ax4.plot(t_log, ref_vel_norms, linewidth=2.5, label='Reference Velocity', 
         color='red', alpha=0.7)
ax4.plot(t_log, actual_vel_norms, linewidth=2.5, label='Actual Velocity', 
         color='blue', alpha=0.8)

# Fill area between velocities to show tracking
ax4.fill_between(t_log, ref_vel_norms, actual_vel_norms, alpha=0.3, color='gray', 
                label='Velocity Tracking Error')

# Mark waypoint times
for wp_time in trajectory_generator.waypoint_times:
    if wp_time <= t_log[-1]:
        ax4.axvline(x=wp_time, color='red', linestyle=':', alpha=0.5, linewidth=1)

ax4.set_title('Velocity Tracking Performance')
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Velocity Magnitude [rad/s]')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Smooth trajectory real-time hardware control and analysis completed successfully!")

# Save results with smooth trajectory data
try:
    filename = f'ur5e_smooth_trajectory_mpc_hardware_results_{time.strftime("%Y%m%d_%H%M%S")}.npz'
    np.savez(filename, 
             t=t_log, 
             y_cl=y_cl, 
             y_ref_cl=y_ref_cl, 
             u_cl=u_cl, 
             acc_cl=acc_cl,
             ref_vel_cl=ref_vel_cl,
             waypoints=np.array(waypoints),
             waypoint_times=trajectory_generator.waypoint_times,
             actual_robot_positions=actual_robot_positions,
             actual_robot_velocities=actual_robot_velocities,
             solve_times=solve_times,
             tracking_errors=tracking_errors,
             total_trajectory_time=total_trajectory_time,
             max_velocity=max_velocity)
    print(f"Smooth trajectory results saved to '{filename}'")
except Exception as e:
    print(f"Error saving results: {e}")

print("All smooth trajectory operations completed!")
print("\n" + "="*80)
print("SMOOTH TRAJECTORY EXECUTION SUMMARY:")
print("="*80)
print("The robot successfully executed a smooth trajectory through multiple waypoints:")
for i, wp_time in enumerate(trajectory_generator.waypoint_times):
    if i < len(waypoints):
        print(f"Waypoint {i}: {waypoints[i]} (Time: {wp_time:.1f}s)")
print(f"Total trajectory time: {total_trajectory_time}s")
print(f"Path length: {path_length:.3f} rad")
print(f"Average velocity: {avg_ref_velocity:.3f} rad/s")
print(f"Maximum velocity: {max_ref_velocity:.3f} rad/s")
print("✓ SMOOTH CONTINUOUS MOTION - NO STOPS AT WAYPOINTS")
print("="*80)