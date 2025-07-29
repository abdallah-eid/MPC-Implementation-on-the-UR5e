% --- Discrete-time system matrices ---
Model=ss1;
A = Model.A;
B = Model.B;
C = Model.C;
D = Model.D;
Ts = Model.Ts;  % Sampling time

% --- Time vector ---

%t = 0:0.016:3;
t=t_log;
nT = length(t);
% --- Define input (ramp velocity) ---
alpha = 0.1;
vel_input=[];
acc=[];
%{
for p = 0:0.016 :3
 vel_input=[vel_input;alpha*ones(1,6)*p];
 acc=[acc;alpha*ones(1,6)];

end
%}
u_matrix = [vel_data];  % [nT x 6]
% --- Initial state: start from specific joint positions, zero velocity ---
q0 =[0;-1.57;0;-1.57;0;0];   % 6x1
%dq0 = zeros(6,1);      % 6x1
x0 = inv(C)*q0;         % 12x1
x = x0;

% --- Preallocate state and output logs ---
X = zeros(nT, 6);  % State history
Y = zeros(nT, 6);   % Output history

% --- Discrete-time simulation ---
for k = 1:nT                                       
    u_k = u_matrix(k, :)'; % 6x1 input
    Y(k, :) = (C * x )';   % 6x1 output
    X(k, :) = x';          % store state

    if k < nT
        x = A * x + B * u_k; % update state
    end
end
vel_sim = zeros(nT, 6);
for i = 1:6
    vel_sim(1:end-1, i) = diff(Y(:, i)) / 0.016;  % Finite difference
    vel_sim(end, i) = vel_sim(end-1, i);       % Pad last value for equal length
end


% --- Plot Simulated Joint Positions Only ---
figure
for i = 1:6
    subplot(3,2,i);
    plot(t, Y(:,i), 'b-', 'LineWidth', 1.5); hold on;
    plot(t, positions(:, i), 'r--', 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel(['posistions', num2str(i), ' (rad)']);
    title(['Joint ', num2str(i)]);
    legend('Simulated Positions', 'Measured Positions');
    grid on;
end
figure

for i = 1:6
    subplot(3,2,i);
    plot(t, vel_sim(:, i), 'b-', 'LineWidth', 1.5); hold on;
    plot(t, vel_Actual(:, i), 'r--', 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel(['vel ', num2str(i), ' (rad/s)']);
    title(['Joint ', num2str(i)]);
    legend('Simulated vel', 'Measured vel');
    grid on;

end