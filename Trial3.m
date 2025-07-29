% Load UR5e robot model
 robot = loadrobot('universalUR5e', 'DataFormat', 'row');  % Load UR5e model
 robot.Gravity = [0, 0, -9.81];

% Load real data from workspace
positions_real = evalin('base', 'positions');
vel_data = evalin('base', 'vel_data');
torques_real = evalin('base', 'torques');

% Trim everything to same valid length
num_samples =188; 
positions_real = positions_real(1:num_samples, :);
vel_data = vel_data(1:num_samples, :);
torques_real = torques_real(1:num_samples, :);
time_vector = time_vector(1:num_samples);
% Interpolate real torques over time
torque_interp = @(t) interp1(time_vector, torques_real, t, 'linear', 'extrap');

% Initial state from real data
q0 = positions_real(1,:);
dq0 = vel_data(1,:);
x0 = [q0, dq0]';

% Define acceleration profile
ddq_profile = @(t) 0.1 * ones(1, 6);
q_now = @(t) 0.05 * t^2 * ones(1,6)+q0;   % q(t) = ∫∫ 0.1 dt² = 0.05 t²
dq_now = @(t) 0.1 * t * ones(1,6)+dq0;
% Declare global to store tau_link inside dynamics function

tau_link_log = zeros(num_samples, 6);

% Simulate dynamics using ODE45
[t, x] = ode45(@(t, x) robotDynamicsWithFullInertia(t, x, robot, ddq_profile,q_now,dq_now), time_vector, x0);
% Extract simulated joint positions and velocities
q_simulated = x(:, 1:6);     % desired joint positions over time
dq_simulated = x(:, 7:12);   % optional: desired joint velocities

% Save to workspace for use in MPC
assignin('base', 'q_desired_mpc', q_simulated);
assignin('base', 'dq_desired_mpc', dq_simulated);
assignin('base', 't_desired_mpc', t);  % optional: time vector

% Recompute torque using inverse dynamics
dt = mean(diff(time_vector));
torque_sim = zeros(num_samples, 6);
Jm = [6.6119, 4.6147, 7.9773, 1.2249, 1.1868, 1.1981] * 1e-5;
Jr = [10.7, 10.7, 10.7, 0.91, 0.91, 0.91] * 1e-5;
Jtotal = Jm + Jr;

for k = 2:num_samples
 

    tau_link = inverseDynamics(robot, q, dq, ddq);
    tau_motor = 100 * Jtotal .* ddq;
    torque_sim(k,:) = tau_link;
end

%--- Plot Simulated Torques from inverseDynamics ---
figure('Name','Simulated Torques Only');
for i = 1:6
    subplot(3,2,i);
    plot(time_vector, torque_sim(:,i), 'b-', 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel(sprintf('\tau_%d (Nm)', i));
    title(sprintf('Joint %d Simulated Torque', i));
    grid on;
end
sgtitle('Simulated Torques (with motor + reducer inertia)');
%}

%% --- Plot Real Velocities Only ---
figure('Name','Real Velocities Only');
for i = 1:6
    subplot(3,2,i);
    plot(time_vector, vel_data(:,i), 'r-', 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel(sprintf('dq_%d (rad/s)', i));
    title(sprintf('Joint %d Real Velocity', i));
    grid on;
end
sgtitle('Real Joint Velocities');

%% --- Plot Joint Positions (Simulated vs Real) ---
figure('Name','Joint Position Comparison');
for i = 1:6
    subplot(3,2,i);
    plot(t, x(:,i), 'b-', 'DisplayName', 'Simulated');
    hold on;
    plot(time_vector, positions_real(:,i), 'r--', 'DisplayName', 'Real');
    xlabel('Time (s)');
    ylabel(sprintf('q_%d (rad)', i));
    title(sprintf('Joint %d Position', i));
    legend; grid on;
end
sgtitle('Joint Position: Simulated vs Real');

%% --- Plot Joint Velocities (Simulated vs Real) ---
figure('Name','Joint Velocity Comparison');
for i = 1:6
    subplot(3,2,i);
    plot(t, x(:,i+6), 'b-', 'DisplayName', 'Simulated');
    hold on;
    plot(time_vector, vel_data(:,i), 'r--', 'DisplayName', 'Real');
    xlabel('Time (s)');
    ylabel(sprintf('dq_%d (rad/s)', i));
    title(sprintf('Joint %d Velocity', i));
    legend; grid on;
end
sgtitle('Joint Velocity: Simulated vs Real');

%% --- Plot Torque Comparison: Simulated vs Real ---
figure('Name','Joint Torque Comparison');
for i = 1:6
    subplot(3,2,i);
    %plot(time_vector, torque_sim(:,i), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Simulated');
    %hold on;
    plot(time_vector, torques_real(:,i), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Real');
    xlabel('Time (s)');
    ylabel(sprintf('\\tau_%d (Nm)', i));
    title(sprintf('Joint %d Torque', i));
    legend; grid on;
end
sgtitle('Joint Torque: Simulated vs Real');
%}

%% --- Dynamics Function ---

function dx = robotDynamicsWithFullInertia(t, x, robot, ddq_profile,q_now,dq_now)

    q = x(1:6)';
    dq = x(7:12)';

    % Robot dynamics
    M = massMatrix(robot, q);
    Cqd = velocityProduct(robot, q, dq);
    G = computeGravityMatrix(q);

    % Desired joint accelerations
    ddq_out = ddq_profile(t);
    q_out=   q_now(t);
    dq_out= dq_now(t);
    % Compute joint torques
    tau_link = inverseDynamics(robot, q, dq, ddq_out);
    
  

    % Inertia modeling
    % Jm = [6.6119, 4.6147, 7.9773, 1.2249, 1.1868, 1.1981] * 1e-5;
    %Jr = [10.7, 10.7, 10.7, 0.91, 0.91, 0.91] * 1e-5;
    %Jtotal = Jm + Jr;
    %tau_motor = 100 * Jtotal .* ddq_out;

    % Final acceleration
    ddq = (tau_link - Cqd - G)/(M);
    dx = [dq, ddq]';
end

