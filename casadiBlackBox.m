import casadi.*

plant = ss1;

A = double(plant.A);
B = double(plant.B);
C = double(plant.C);
D = double(plant.D);

nx = size(A, 1);
nu = size(B, 2);
ny = size(C, 1);

T = 0.01;
N = 20;

u_max = 10 * ones(nu,1);
u_min = -10 * ones(nu,1);
y_max = 2*pi * ones(ny,1);
y_min = -y_max;

x = MX.sym('x', nx);
u = MX.sym('u', nu);

x_next = A*x + B*u;
y = C*x + D*u;

f_dynamics = Function('f_dynamics', {x, u}, {x_next});
f_output = Function('f_output', {x, u}, {y});

U = MX.sym('U', nu, N);
x0_sym = MX.sym('x0', nx);
Yref_sym = MX.sym('Yref', ny, N);

X = MX.zeros(nx, N+1);
X(:,1) = x0_sym;
for k = 1:N
    X(:,k+1) = f_dynamics(X(:,k), U(:,k));
end

Qy = 150 * eye(ny);
R = 2 * eye(nu);
obj = 0;
g = [];

for k = 1:N
    y_k = C*X(:,k) + D*U(:,k);
    y_ref_k = Yref_sym(:,k);
    obj = obj + (y_k - y_ref_k)'*Qy*(y_k - y_ref_k) + U(:,k)'*R*U(:,k);
end

for k = 1:N+1
    if k <= N
        y_k = C*X(:,k) + D*U(:,k);
    else
        y_k = C*X(:,k);
    end
    g = [g; y_k];
end

OPT_variables = U(:);
P = [x0_sym; reshape(Yref_sym, ny*N, 1)];

nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g, 'p', P);

opts = struct;
opts.ipopt.max_iter = 100;
opts.ipopt.print_level = 0;
opts.print_time = false;
opts.ipopt.acceptable_tol = 1e-8;
opts.ipopt.acceptable_obj_change_tol = 1e-6;

solver = nlpsol('solver', 'ipopt', nlp_prob, opts);

x0 = inv(C)*[0;-1.57;0;-1.57;0;0];
y_initial = C * x0;

y_center = y_initial;
y_amplitude = [0.3; 0.5; 0.4; 0.6; 0.8; 0.5];
frequency = [0.2; 0.15; 0.25; 0.1; 0.3; 0.2];
phase_shift = zeros(ny, 1);

u0 = zeros(N, nu);
args = struct;
args.lbx = repmat(u_min, N, 1);
args.ubx = repmat(u_max, N, 1);
args.lbg = repmat(y_min, N+1, 1);
args.ubg = repmat(y_max, N+1, 1);

t0 = 0;
sim_time = 20;
mpcctr = 0;
y_cl = [];
u_cl = [];
t = [];

while (mpcctr < sim_time / T)
    t_future = t0 + (0:N-1) * T;
    Yref = zeros(ny, N);
    for k = 1:N
        for i = 1:ny
            Yref(i, k) = y_center(i) + y_amplitude(i) * sin(2*pi*frequency(i)*t_future(k) + phase_shift(i));
        end
    end

    args.p = [x0; reshape(Yref, ny*N, 1)];
    args.x0 = reshape(u0', nu*N, 1);

    sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx, ...
                 'lbg', args.lbg, 'ubg', args.ubg, 'p', args.p);

    u_opt = reshape(full(sol.x), nu, N)';
    x0 = full(f_dynamics(x0, u_opt(1,:)'));
    y0 = full(C * x0);

    y_cl = [y_cl, y0];
    u_cl = [u_cl; u_opt(1,:)];
    t(end+1) = t0;

    t0 = t0 + T;
    u0 = [u_opt(2:end, :); u_opt(end,:)];
    mpcctr = mpcctr + 1;
end

% === Plotting ===

figure;
time = t;
rows = ceil(ny/2); cols = 2;
t_plot = 0:T:max(time);
y_ref_plot = zeros(ny, length(t_plot));
for k = 1:length(t_plot)
    for i = 1:ny
        y_ref_plot(i, k) = y_center(i) + y_amplitude(i) * sin(2*pi*frequency(i)*t_plot(k) + phase_shift(i));
    end
end

for i = 1:ny
    subplot(rows, cols, i);
    plot(time, y_cl(i,:), 'b-', 'LineWidth', 1.5); hold on;
    plot(t_plot, y_ref_plot(i,:), '--r', 'LineWidth', 1.2);
    plot(time, y_center(i)*ones(size(time)), ':g', 'LineWidth', 1);
    plot(time, (y_center(i)+y_amplitude(i))*ones(size(time)), ':k');
    plot(time, (y_center(i)-y_amplitude(i))*ones(size(time)), ':k');
    legend('Actual Output', 'Sinusoidal Reference', 'Center Line', 'Amplitude Bounds', 'Location', 'best');
    title(['Output y_' num2str(i) ' (f = ' num2str(frequency(i)) ' Hz, A = ' num2str(y_amplitude(i)) ')']);
    xlabel('Time [s]'); ylabel(['y_' num2str(i)]); grid on;
end

figure;
rows_u = ceil(nu/2); cols_u = 2;
for i = 1:nu
    subplot(rows_u, cols_u, i);
    plot(time, u_cl(:,i), 'b-', 'LineWidth', 1.5); hold on;
    plot(time, u_max(i)*ones(size(time)), '--r');
    plot(time, u_min(i)*ones(size(time)), '--r');
    legend('Control Input', 'Constraints', 'Location', 'best');
    title(['Control Input u_' num2str(i)]);
    xlabel('Time [s]'); ylabel(['u_' num2str(i)]); grid on;
end
