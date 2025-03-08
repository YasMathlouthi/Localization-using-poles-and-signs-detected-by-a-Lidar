function ukf3()
% ukf3 - UKF with GNSS and Lidar, fully in one function file.
%
% Usage:
%   1) Place "ukf3.m" and "data.mat" in the same folder.
%   2) Run: >> ukf3
% 
% The code loads 'data.mat' with variables:
%   t, gnss, obs, ref, v, omega
%
% The UKF uses fixCovariance() to keep P positive-definite.
 
    clc; close all;
 
    %----------------------------------------------------------------------
    % 1) Load Data from "data.mat"
    %----------------------------------------------------------------------
    load('data.mat'); 
    % Should have t, gnss, obs, ref, v, omega.
 
    %----------------------------------------------------------------------
    % 2) Define Problem
    %----------------------------------------------------------------------
    % State: x = [px, py, heading, v, omega]
    x = [gnss(1).x; ...
         gnss(1).y; ...
         gnss(1).heading; ...
         v(1); ...
         omega(1)];
 
    % Initial covariance
    P = diag([0.5, 0.5, 0.1, 0.1, 0.1]);
    n = length(x);  % dimension
 
    % Noise
    Q       = diag([0.01, 0.01, 0.001, 0.01, 0.001]);  % Process noise
    R_gnss  = diag([0.2, 0.2, 0.01]);                 % GNSS [x, y, heading]
    R_lidar = diag([0.1, 0.1]);                       % Lidar [range, bearing]
 
    dt = mean(diff(t)); % time step
 
    % Unscented parameters
    alpha  = 1e-3;
    beta   = 2;
    kappa  = 0;
    lambda = alpha^2*(n + kappa) - n;
    w0     = lambda/(n + lambda);  % center weight
 
    % Arrays for storing results
    ukf_estimates = zeros(length(t), n);
    ukf_estimates(1,:) = x';
 
    % Optional: store Lidar obs in global coords
    lidar_observations = [];
 
    %----------------------------------------------------------------------
    % 3) Main UKF Loop
    %----------------------------------------------------------------------
    for k_idx = 2:length(t)
        %------------------------------------------------------
        % 3.1 PREDICTION
        %------------------------------------------------------
        P = fixCovariance(P);  % ensure positivity
 
        % Generate sigma points
        [sigma_points, wPts, nPts] = SigmaPoints_cholesky(x, P, w0);
 
        % Propagate each sigma point
        x_i_pred = zeros(size(sigma_points));
        for i = 1:nPts
            Xi = sigma_points(:, i);
 
            px_next      = Xi(1) + Xi(4)*dt*cos(Xi(3));
            py_next      = Xi(2) + Xi(4)*dt*sin(Xi(3));
            heading_next = Xi(3) + Xi(5)*dt;
            v_next       = v(k_idx);      % overwrite with measurement
            omega_next   = omega(k_idx);  % overwrite
 
            x_i_pred(:, i) = [px_next; py_next; heading_next; v_next; omega_next];
        end
 
        % Predicted mean
        x_pred = zeros(n,1);
        for i = 1:nPts
            x_pred = x_pred + wPts(i)*x_i_pred(:, i);
        end
 
        % Predicted covariance
        P_pred = Q;
        for i = 1:nPts
            diff_i = x_i_pred(:, i) - x_pred;
            P_pred = P_pred + wPts(i)*(diff_i*diff_i');
        end
 
        P_pred = fixCovariance(P_pred);
 
        % Update state/cov for next steps
        x = x_pred;
        P = P_pred;
 
        %------------------------------------------------------
        % 3.2 GNSS UPDATE
        %------------------------------------------------------
        if ~isnan(gnss(k_idx).x)
            % z_gnss = [x; y; heading]
            z_gnss = [gnss(k_idx).x; gnss(k_idx).y; gnss(k_idx).heading];
            % Sigma points in measurement space
            z_sigma_gnss = x_i_pred(1:3, :);
            % Predicted measurement mean
            z_pred_gnss = zeros(3,1);
            for i = 1:nPts
                z_pred_gnss = z_pred_gnss + wPts(i)*z_sigma_gnss(:, i);
            end
 
            % Covariance + cross-cov
            S_gnss = R_gnss;
            T_gnss = zeros(n,3);
 
            for i = 1:nPts
                z_diff = z_sigma_gnss(:, i) - z_pred_gnss;
                x_diff = x_i_pred(:, i)      - x_pred;
                S_gnss = S_gnss + wPts(i)*(z_diff*z_diff');
                T_gnss = T_gnss + wPts(i)*(x_diff*z_diff');
            end
 
            % Kalman gain
            K_gnss = T_gnss / S_gnss;
 
            % Update
            x_pred = x_pred + K_gnss*(z_gnss - z_pred_gnss);
            P_pred = P_pred - K_gnss*S_gnss*K_gnss';
 
            P_pred = fixCovariance(P_pred);
 
            % Overwrite
            x = x_pred;
            P = P_pred;
        end
 
        %------------------------------------------------------
        % 3.3 Lidar UPDATE
        %------------------------------------------------------
        if ~isempty(obs(k_idx).x_map)
            n_landmarks = length(obs(k_idx).x_map);
            dim_z = 2*n_landmarks;
 
            % Build measurement sigma points
            z_sigma_lidar = zeros(dim_z, nPts);
            for i_sp = 1:nPts
                Xi = x_i_pred(:, i_sp);
                z_loc = zeros(dim_z,1);
 
                for L = 1:n_landmarks
                    dx = obs(k_idx).x_map(L) - Xi(1);
                    dy = obs(k_idx).y_map(L) - Xi(2);
                    r       = sqrt(dx^2 + dy^2);
                    bearing = atan2(dy, dx) - Xi(3);
 
                    row = 2*(L-1)+1;
                    z_loc(row)   = r;
                    z_loc(row+1) = bearing;
                end
                z_sigma_lidar(:, i_sp) = z_loc;
            end
 
            % Predicted measurement mean
            z_pred_lidar = zeros(dim_z,1);
            for i_sp = 1:nPts
                z_pred_lidar = z_pred_lidar + wPts(i_sp)*z_sigma_lidar(:, i_sp);
            end
 
            % Actual Lidar measurement from x_pred
            z_lidar = zeros(dim_z,1);
            for L = 1:n_landmarks
                dx = obs(k_idx).x_map(L) - x_pred(1);
                dy = obs(k_idx).y_map(L) - x_pred(2);
                r_meas = sqrt(dx^2 + dy^2);
                bearing_meas = atan2(dy, dx) - x_pred(3);
 
                row = 2*(L-1)+1;
                z_lidar(row)   = r_meas;
                z_lidar(row+1) = bearing_meas;
            end
 
            % Covariance + cross-cov
            R_lidar_full = kron(eye(n_landmarks), R_lidar);
            S_lidar = R_lidar_full;
            T_lidar = zeros(n, dim_z);
 
            for i_sp = 1:nPts
                z_diff = z_sigma_lidar(:, i_sp) - z_pred_lidar;
                x_diff = x_i_pred(:, i_sp) - x_pred;
                S_lidar = S_lidar + wPts(i_sp)*(z_diff*z_diff');
                T_lidar = T_lidar + wPts(i_sp)*(x_diff*z_diff');
            end
 
            % Gain
            K_lidar = T_lidar / S_lidar;
 
            % Update
            x_pred = x_pred + K_lidar*(z_lidar - z_pred_lidar);
            P_pred = P_pred - K_lidar*S_lidar*K_lidar';
 
            P_pred = fixCovariance(P_pred);
 
            x = x_pred;
            P = P_pred;
 
            % (Optional) store Lidar points for plotting
            for L = 1:n_landmarks
                row = 2*(L-1)+1;
                rr = z_lidar(row);
                br = z_lidar(row+1);
 
                obs_x = x_pred(1) + rr*cos(x_pred(3) + br);
                obs_y = x_pred(2) + rr*sin(x_pred(3) + br);
                lidar_observations = [lidar_observations; obs_x, obs_y];
            end
        end
 
        %------------------------------------------------------
        % 3.4 Store current estimate
        %------------------------------------------------------
        ukf_estimates(k_idx,:) = x';
    end
 
    %----------------------------------------------------------------------
    % 4) Plot Results
    %----------------------------------------------------------------------
    figure; hold on; grid on;
 
    plot([ref.x],[ref.y], 'g-', 'LineWidth',1.5, 'DisplayName','Reference');
    plot(ukf_estimates(:,1), ukf_estimates(:,2), 'b-', 'LineWidth',1.5, ...
         'DisplayName','UKF Estimate');
    plot([gnss.x],[gnss.y], 'ro', 'DisplayName','GNSS');
 
    if ~isempty(lidar_observations)
        plot(lidar_observations(:,1), lidar_observations(:,2), 'kx', ...
             'DisplayName','Lidar Obs');
    end
 
    legend('Location','best');
    xlabel('East (m)');
    ylabel('North (m)');
    title('UKF with GNSS + Lidar');
 
end
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           SUB-FUNCTION: fixCovariance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function P = fixCovariance(P)
    % Force symmetry
    P = 0.5*(P + P');
    % Eigen check
    [V, D] = eig(P);
    d = diag(D);
    min_eig = min(d);
    if min_eig < 0
        P = P + eye(size(P))*1.1*abs(min_eig);
    end
    % Small regularization
    eps_reg = 1e-12;
    P = P + eps_reg*eye(size(P));
    % Re-symmetrize
    P = 0.5*(P + P');
end
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           SUB-FUNCTION: SigmaPoints_cholesky
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xPts, wPts, nPts] = SigmaPoints_cholesky(x, P, w0)
    % Sigma point generation with Cholesky
    P = fixCovariance(P);  % again just to be safe
 
    n = length(x);
    nPts = 2*n + 1;
    xPts = zeros(n, nPts);
 
    M = chol(P, 'lower');
    scale = sqrt(n/(1 - w0));
 
    for i = 1:n
        xPts(:, i)   = x + scale*M(:,i);
        xPts(:, i+n) = x - scale*M(:,i);
    end
    xPts(:, nPts) = x;
 
    wPts = ones(1, nPts)*(1 - w0)/(2*n);
    wPts(nPts) = w0;
end