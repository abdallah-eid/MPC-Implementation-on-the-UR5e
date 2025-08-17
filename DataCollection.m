clear;
import py.rtde_receive.RTDEReceiveInterface
import py.rtde_control.RTDEControlInterface

% Connect to UR5e
rtde_r = RTDEReceiveInterface("192.168.1.10");
rtde_c = RTDEControlInterface("192.168.1.10");

% Define all 9 joint velocities
joint_velocities = [
    0.3,  0.3,  0.3,  0.3,  0.3,  0.3;
    0.1,  0,    0,    0,    0,    0;
    0,    -0.1,  0,    0,    0,    0;
    0,    0,    0.1,  0,    0,    0;
    0,    0,    0,    0.1,  0,    0;
    0,    0,    0,    0,    0.1,  0;
    0,    0,    0,    0,    0,    0.1;
    0,    0,    0,    -0.1,  0,    0;
   -0.2, -0.5, -0.7, 0.3, 0.3, 0.3;
    0.1, 0.6, 0.5,-0.1,-0.5,-0.4;
    0  , -0.1,0.3,0,0,0;
    -0.4,0.2,-0.7,0.2,-0.5,-0.9;
     0.5,0,0,0,0,0;
     0,-0.2,-0.4,0.1,0.3,0.6;
     -0.325, 0.53,0.27,0.15,0.8,0.56;
      0, 0, 0.3,0,0,0;
      0.5,-0.1,-0.6,0,0,0
];

acceleration = 0.2;
time = 2.5;      % Duration per velocity command
dt = 0.008;
steps_per_segment = ceil(time / dt);
total_segments = size(joint_velocities, 1);

% Preallocate large arrays
max_total_steps = steps_per_segment * total_segments;
positions = zeros(max_total_steps, 6);
vel_Target  = zeros(max_total_steps, 6);
vel_Actual  = zeros(max_total_steps, 6);
acc_data  = zeros(max_total_steps, 6);
torques   = zeros(max_total_steps, 6);
tcp       = zeros(max_total_steps, 6);
t_log     = zeros(max_total_steps, 1);

% Loop through each velocity command
k = 1;
for i = 1:total_segments
   if i==2 || i==3 || i==4 || i==5 || i==6
     acceleration=0.1;
   end   
   if i==7 || i==8 || i==9 
     acceleration=0.09;
   end   
   if i==10 || i==11 || i==12
     acceleration=0.05;
   end   
    if i==13
     acceleration=0.02;
   end  
      if i==14 || i==15 
      acceleration=0.15;
      end 
    fprintf("Executing velocity set %d...\n", i);
    joint_velocity = joint_velocities(i, :).';
    joint_velocity_py = py.list(num2cell(joint_velocity.'));
    rtde_c.speedJ(joint_velocity_py, acceleration, time);
    t_segment_start = tic;
    while toc(t_segment_start) < time
        q_py   = rtde_r.getActualQ();
        Tqd_py  = rtde_r.getTargetQd();
        Aqd_py  = rtde_r.getActualQd();
        qdd_py = rtde_r.getTargetQdd();
        tau_py = rtde_r.getTargetMoment();
        q_tcp  = rtde_r.getActualTCPPose();

        positions(k, :) = double(q_py);
        vel_Target(k, :)  = double(Tqd_py);
        vel_Actual(k, :)  = double(Aqd_py);
        acc_data(k,:)   = double(qdd_py);
        torques(k, :)   = double(tau_py);
        tcp(k, :)       = double(q_tcp);
        t_log(k)        = (i-1)*time + toc(t_segment_start);

        pause(dt);
        k = k + 1;
    end
end

% Stop robot safely
rtde_c.speedStop(pyargs('a', acceleration))

% Truncate to actual size
positions = positions(1:k-1, :);
vel_Target  = vel_Target(1:k-1, :);
vel_Actual  = vel_Actual(1:k-1,:);
acc_data  = acc_data(1:k-1, :);
torques   = torques(1:k-1, :);
tcp       = tcp(1:k-1, :);
t_log     = t_log(1:k-1);
Input=[vel_Target,acc_data];
Output=[positions,vel_Actual];


