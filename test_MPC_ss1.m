% MATLAB script to simulate sinusoidal trajectories for all UR5e joints

% Define parameters
duration = 20;      % Duration in seconds (matches the plot)
t_step = 0.02;      % Time step in seconds (20 ms)
t = 0:t_step:duration; % Time vector

% Define parameters for each joint (J1 to J6)
freq = [0.2, 0.15, 0.1, 0.2, 0.1, 0.2];  % Frequencies in Hz (inspired by y_1 to y_6)
amp = [0.3, 0.5, 0.6, 0.4, 0.5, 0.3];    % Amplitudes in radians (inspired by y_1 to y_6)
delay_samples = 5;  % Approximate delay (5 steps = 0.1 s) for actual output

% Initialize arrays for all joints
theta_ref = zeros(length(t), 6);  % Reference trajectories for J1 to J6
theta_actual = zeros(length(t), 6); % Actual trajectories for J1 to J6

% Generate sinusoidal trajectories for each joint
for i = 1:6
    % Reference trajectory: A * sin(2*pi*f*t)
    theta_ref(:, i) = amp(i) * sin(2 * pi * freq(i) * t);
    
    % Actual output with delay and noise
    % Fix: Transpose zeros to match column vector dimension
    theta_actual(1:delay_samples, i) = zeros(delay_samples, 1); % Pad with zeros
    theta_actual(delay_samples+1:end, i) = theta_ref(1:end-delay_samples, i);
    theta_actual(:, i) = theta_actual(:, i) + 0.01 * randn(size(theta_actual(:, i)));
end

% Define amplitude bounds and centerline for each joint
upper_bound = amp' * ones(1, length(t));  % Upper bounds for each joint
lower_bound = -amp' * ones(1, length(t)); % Lower bounds for each joint
center_line = zeros(length(t), 6);        % Center line for each joint

% Create figure with 2x3 subplot grid
figure('Name', 'Sinusoidal Trajectories for UR5e Joints', 'NumberTitle', 'off');

% Loop through each joint to create subplots
figure
for i = 1:6
    subplot(3,2,i);
    plot(t,theta_actual(:,i), 'r-', 'LineWidth', 1.5); hold on;
    plot(t, theta_ref(:, i), 'b--', 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel(['posistions', num2str(i), ' (rad)']);
    title(['Joint ', num2str(i)]);
    legend('MPC', 'Reference');
    grid on;
end


% Display current date and time
disp(['Simulation run at: ', datestr(now, 'HH:MM PM dd-mmm-yyyy')]);