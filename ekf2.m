clear;
close all;

% Load data
load("data.mat");


% Initialize EKF variables
% Initial state: x, y, theta, v, omega
x = [gnss(1).x; gnss(1).y; gnss(1).heading; v(1); omega(1)];
P = diag([0.5, 0.5, 0.1, 0.1, 0.1]); % Initial covariance

% Noise matrices
Q = diag([0.01, 0.01, 0.001, 0.01, 0.001]); % Process noise
R_gnss = diag([0.2, 0.2, 0.01, 0.1, 0.1]); % GNSS observation noise (now includes v, omega)
R_lidar = diag([0.1, 0.1]); % Lidar observation noise

% Time step
dt = mean(diff(t));

% Storage for EKF estimates
ekf_estimates = zeros(length(t), 3); % Only store x, y, theta for visualization
ekf_estimates(1, :) = x(1:3)';
lidar_observations = []; % Store lidar observations for plotting

% EKF loop
for k = 2:length(t)
    % Prediction Step (Standard EKF equations)
    % State prediction
    x = [x(1) + x(4) * dt * cos(x(3));
         x(2) + x(4) * dt * sin(x(3));
         x(3) + x(5) * dt;
         x(4);
         x(5)];

    % Jacobian of the state transition function
    F = [1, 0, -x(4) * dt * sin(x(3)), dt * cos(x(3)), 0;
         0, 1, x(4) * dt * cos(x(3)), dt * sin(x(3)), 0;
         0, 0, 1, 0, dt;
         0, 0, 0, 1, 0;
         0, 0, 0, 0, 1];
     

    % Covariance prediction
    P = F * P * F' + Q;

    % Correction Step (Lidar)
    if ~isempty(obs(k).x_map)
        % Measurement
        z_lidar = [obs(k).x_map'; obs(k).y_map'];
        lidar_observations = [lidar_observations; z_lidar']; % Save for plotting
        % Observation model Jacobian
        H_lidar = [1, 0, 0, 0, 0;
                   0, 1, 0, 0, 0];
        % Predicted measurement
        z_pred = H_lidar * x;
        % Kalman gain
        K = P * H_lidar' / (H_lidar * P * H_lidar' + R_lidar);
        % State update
        x = x + K * (z_lidar - z_pred);
        % Covariance update
        P = (eye(size(P)) - K * H_lidar) * P;
    end

    % Correction Step (GNSS, if available)
    if ~isnan(gnss(k).x)
        % Measurement
        z_gnss = [gnss(k).x; gnss(k).y; gnss(k).heading; v(k); omega(k)];
        % Observation model Jacobian
        H_gnss = [1, 0, 0, 0, 0;
                  0, 1, 0, 0, 0;
                  0, 0, 1, 0, 0;
                  0, 0, 0, 1, 0;
                  0, 0, 0, 0, 1];
        % Predicted measurement
        z_pred = H_gnss * x;
        % Kalman gain
        K = P * H_gnss' / (H_gnss * P * H_gnss' + R_gnss);
        % State update
        x = x + K * (z_gnss - z_pred);
        % Covariance update
        P = (eye(size(P)) - K * H_gnss) * P;
    end


    % Save EKF estimate for visualization
    ekf_estimates(k, :) = x(1:3)';
end

% Plot results
figure;
plot([ref.x], [ref.y], 'g-', 'DisplayName', 'Reference');
hold on;
plot(ekf_estimates(:, 1), ekf_estimates(:, 2), 'b-', 'DisplayName', 'EKF Estimate'); % Continuous line for EKF
plot([gnss.x], [gnss.y], 'ro', 'DisplayName', 'GNSS Observations');
plot(lidar_observations(:, 1), lidar_observations(:, 2), 'kx', 'DisplayName', 'Lidar Observations');
legend;
title('EKF Localization with GNSS and Lidar Observations');
xlabel('East (m)');
ylabel('North (m)');
grid on;
