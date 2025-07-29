function nlobj = NMPC_ss1(plant)
    % Load the identified 12-state model
    %plant = evalin('base', 'ss3');
    A = plant.A; B = plant.B; C = plant.C; D = plant.D;

    nx = 6;    % 12 states
    ny = 6;    % typically 6 outputs (joint positions)
    nu = 6;    % 6 inputs (joint velocities)

    % Create nlmpc object with correct dimensions
    nlobj = nlmpc(nx, ny, nu);
    nlobj.Ts = 0.008;
    nlobj.PredictionHorizon = 10;
    nlobj.ControlHorizon = 1;

    % Define model dynamics (state and output)
    nlobj.Model.StateFcn  = @(x,u) A*x + B*u;
    nlobj.Model.OutputFcn = @(x,u) C*x ;

    % (Optional) Jacobians
   % nlobj.Jacobian.StateFcn  = @(x,u) A;
   % nlobj.Jacobian.OutputFcn = @(x,u) C;

    % Cost weights
nlobj.Weights.OutputVariables = [65,65,65,65,65,65];
nlobj.Weights.ManipulatedVariablesRate = [1.5 1.5 1.5 1.5 1.5 1.5];
  
OVmax=-OVmin;
MVmax=[10,10,10,10,10,10];
MVmin=-MVmax;
    % Input constraints
    for i = 1:6
        nlobj.MV(i).Min = MVmin(i);
        nlobj.MV(i).Max =  MVmax(i);
        
    end

   for i = 1:6
       
         nlobj.OutputVariables(i).Min = OVmin(i);
        nlobj.OutputVariables(i).Max = OVmax(i);
    end


 nlobj.Optimization.SolverOptions.Algorithm           = 'sqp';
    nlobj.Optimization.SolverOptions.Display             = 'iter';
    nlobj.Optimization.SolverOptions.OptimalityTolerance = 1e-4;
    nlobj.Optimization.SolverOptions.ConstraintTolerance = 1e-4;
end

q_min  = -[10,10,10,10,10,10];
  OVmin=q_min;
  %q_max  = deg2rad([ 360  360  360  360  360  360]);