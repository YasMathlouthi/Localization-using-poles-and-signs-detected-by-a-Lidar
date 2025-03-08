clear;
clc;

% Load simulation data
load("data.mat");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Simulation parameters
dt = 0.1; % Time step
N = length(t); % Number of time steps

% Process noise covariance
Q = diag([0.1, 0.1, 0.0001, 0.1, 0.0001]);

% Observation noise covariance
R_gnss = diag([0.04, 0.04, 0.0001]); % GNSS noise (x, y, theta)
R_lidar = diag([0.1, 0.1]); % Lidar noise (x, y)

% State vector initialization
x_hat = [ref(1).x; ref(1).y; ref(1).heading; 0; 0]; % Initial estimate [x, y, theta, v, omega]
P = eye(5); % Initial covariance
% Associating lidar measurements with known map features ensures that only meaningful data is used to update the vehicle's state, improving localization accuracy.
% The lidar may detect multiple features at once. The algorithm assigns each detection to the nearest valid landmark.

% Preallocate storage
X_est = zeros(5, N); % EKF estimates

% Evolution model
f = @(x, u) [
    x(1) + x(4) * cos(x(3)) * dt;
    x(2) + x(4) * sin(x(3)) * dt;
    x(3) + x(5) * dt;
    x(4) + u(1) * dt;
    x(5) + u(2) * dt
];

% Jacobian of Evolution model
F = @(x) [
    1, 0, -x(4) * sin(x(3)) * dt, cos(x(3)) * dt, 0;
    0, 1,  x(4) * cos(x(3)) * dt, sin(x(3)) * dt, 0;
    0, 0,  1, 0, dt;
    0, 0,  0, 1, 0;
    0, 0,  0, 0, 1
];

% Define GNSS observation model
h_gnss = @(x) [x(1); x(2); x(3)]; % Vehicle pose
H_gnss = [
    1, 0, 0, 0, 0;
    0, 1, 0, 0, 0;
    0, 0, 1, 0, 0
];

% Define Lidar observation model
h_lidar = @(x, z) [
    x(1) + z(1) * cos(x(3)) - z(2) * sin(x(3));
    x(2) + z(1) * sin(x(3)) + z(2) * cos(x(3))
]; % Map frame coordinates of lidar_obs (poles_obs & signs_obs)
H_lidar = @(x, z) [
    1, 0, -z(1) * sin(x(3)) - z(2) * cos(x(3)), 0, 0;
    0, 1,  z(1) * cos(x(3)) - z(2) * sin(x(3)), 0, 0
];

% Run EKF
for i = 1:N
    % Predict step
    u = [v(i); omega(i)];
    x_hat = f(x_hat, u);
    F_k = F(x_hat);
    P = F_k * P * F_k' + Q;

    % Update step (GNSS observation)
    if ~isnan(gnss(i).x) && ~isnan(gnss(i).y) && ~isnan(gnss(i).heading)
        z_gnss = [gnss(i).x; gnss(i).y; gnss(i).heading];
        z_pred = h_gnss(x_hat);
        y = z_gnss - z_pred; % Innovation
        S = H_gnss * P * H_gnss' + R_gnss; % Innovation covariance
        K = P * H_gnss' / S; % Kalman gain
        x_hat = x_hat + K * y; % Update state
        P = (eye(5) - K * H_gnss) * P; % Update covariance
    end

    % Update step (Poles observation)
    if ~isempty(poles_obs(i).x)
        associations = data_association(poles_obs(i), map, x_hat);
        for j = 1:length(associations)
            if isnan(associations(j))
                continue; % Skip unmatched detections
            end
            z_lidar = [poles_obs(i).x(j); poles_obs(i).y(j)];
            z_map = map(associations(j), :)';
            z_pred = h_lidar(x_hat, z_lidar);
            y = z_map - z_pred; % Innovation
            H_k_lidar = H_lidar(x_hat, z_lidar);
            S = H_k_lidar * P * H_k_lidar' + R_lidar;
            K = P * H_k_lidar' / S;
            x_hat = x_hat + K * y; % Update state
            P = (eye(5) - K * H_k_lidar) * P; % Update covariance
        end
    end

    % Update step (Signs observation)
    if ~isempty(signs_obs(i).x)
        associations = data_association(signs_obs(i), map, x_hat);
        for j = 1:length(associations)
            if isnan(associations(j))
                continue; % Skip unmatched detections
            end
            z_lidar = [signs_obs(i).x(j); signs_obs(i).y(j)];
            z_map = map(associations(j), :)';
            z_pred = h_lidar(x_hat, z_lidar);
            y = z_map - z_pred; % Innovation
            H_k_lidar = H_lidar(x_hat, z_lidar);
            S = H_k_lidar * P * H_k_lidar' + R_lidar;
            K = P * H_k_lidar' / S;
            x_hat = x_hat + K * y; % Update state
            P = (eye(5) - K * H_k_lidar) * P; % Update covariance
        end
    end

    % Store estimate
    X_est(:, i) = x_hat;
end

% Plot results
% The convergence of the Pose (x, y, theta)
figure;
subplot(3, 1, 1);
plot(t, X_est(1, :), 'r', t, [ref.x], 'b--');
legend('Estimated X', 'Reference X');
xlabel('Time (s)'); ylabel('X Position (m)'); grid on;
subplot(3, 1, 2);
plot(t, X_est(2, :), 'r', t, [ref.y], 'b--');
legend('Estimated Y', 'Reference Y');
xlabel('Time (s)'); ylabel('Y Position (m)'); grid on;
subplot(3, 1, 3);
plot(t, X_est(3, :), 'r', t, [ref.heading], 'b--');
legend('Estimated Heading', 'Reference Heading');
xlabel('Time (s)'); ylabel('Heading (rad)'); grid on;

% Plot results
figure;
plot([ref.x], [ref.y], 'g-', 'DisplayName', 'Reference');
hold on
plot([gnss.x], [gnss.y],"b+", 'DisplayName', 'GNSS')
hold on;
plot(X_est(1, :), X_est(2, :), 'r', 'DisplayName', 'Estimated');
legend;
xlabel('East (m)');
ylabel('North (m)');
grid on;

% After the EKF loop
Px1 = P(1,1);
Px2 = P(2,2);
Px3 = P(3,3);
% Extract estimated  positions and heading from the estimates
est_x = X_est(1, :);
est_y = X_est(2, :);
est_heading = X_est(3, :);

% True  positions and heading 
true_x = [ref.x];
true_y = [ref.y];
true_heading = [ref.heading];

% Error calculation
ex = est_x - true_x;
% Mean Error in x
ex_m = mean(ex);
% Maximum Absolute Error in x
ex_abs_max = max(abs(ex));
% Mean Square Error in x
ex_MSE = mean(ex.^2);

% Consistency check using Chi-square distribution for error normalization
DoF = 1; % Degree of freedom for x position
Pr = 0.95; % Chosen percentile
Th = chi2inv(Pr, DoF); % Threshold

% Assuming Px1 contains the variances of the x position estimates at each step
% We calculate the normalized error and check against the threshold
Consistency_x = mean((ex.^2 ./ Px1(1,:)) < Th);

% Display results
disp(['Mean Error in x = ', num2str(ex_m)]);
disp(['Max Error in x = ', num2str(ex_abs_max)]);
disp(['Mean Square Error in x = ', num2str(ex_MSE)]);
disp(['Consistency in x = ', num2str(Consistency_x)]);


% Error calculation
ey = est_y - true_y;
% Mean Error in x
ey_m = mean(ey);
% Maximum Absolute Error in x
ey_abs_max = max(abs(ey));
% Mean Square Error in x
ey_MSE = mean(ey.^2);

% Consistency check using Chi-square distribution for error normalization
DoF = 1; % Degree of freedom for x position
Pr = 0.95; % Chosen percentile
Th = chi2inv(Pr, DoF); % Threshold

% Assuming Px1 contains the variances of the x position estimates at each step
% We calculate the normalized error and check against the threshold
Consistency_y = mean((ey.^2 ./ Px2(1,:)) < Th);

% Display results
disp(['Mean Error in y = ', num2str(ey_m)]);
disp(['Max Error in y = ', num2str(ey_abs_max)]);
disp(['Mean Square Error in y = ', num2str(ey_MSE)]);
disp(['Consistency in y = ', num2str(Consistency_y)]);


% Error calculation
eh = est_heading - true_heading;
% Mean Error in x
eh_m = mean(eh);
% Maximum Absolute Error in x
eh_abs_max = max(abs(eh));
% Mean Square Error in x
eh_MSE = mean(eh.^2);

% Consistency check using Chi-square distribution for error normalization
DoF = 1; % Degree of freedom for x position
Pr = 0.95; % Chosen percentile
Th = chi2inv(Pr, DoF); % Threshold

% Assuming Px1 contains the variances of the x position estimates at each step
% We calculate the normalized error and check against the threshold
Consistency_h = mean((eh.^2 ./ Px3(1,:)) < Th);

% Display results
disp(['Mean Error in heading = ', num2str(eh_m)]);
disp(['Max Error in heading = ', num2str(eh_abs_max)]);
disp(['Mean Square Error in heading = ', num2str(eh_MSE)]);
disp(['Consistency in heading = ', num2str(Consistency_h)]);

function associations = data_association(detections, map, state_est)
    % Inputs:
    % detections: Struct with fields x, y (lidar detections in the local frame)
    % map: Matrix of landmarks [x_map, y_map] in the ENU frame
    % state_est: Current state estimate [x_est, y_est, theta_est]
    % Outputs:
    % associations: Array of associated landmark indices for each detection

    % Extract state estimates
    x_est = state_est(1);
    y_est = state_est(2);
    theta_est = state_est(3);

    % Transform detections to the global frame
    num_detections = length(detections.x);
    detections_global = zeros(num_detections, 2);
    for i = 1:num_detections
        detections_global(i, :) = [
            x_est + detections.x(i) * cos(theta_est) - detections.y(i) * sin(theta_est),
            y_est + detections.x(i) * sin(theta_est) + detections.y(i) * cos(theta_est)
        ];
    end

    % Initialize associations
    associations = nan(num_detections, 1);

    % Match detections to landmarks
    for i = 1:num_detections
        min_dist = inf;
        best_match = -1;
        for j = 1:size(map, 1)
            dist = norm(detections_global(i, :) - map(j, :)); % Euclidean distance
            if dist < min_dist && dist < 2 % Threshold of 2 meters
                min_dist = dist;
                best_match = j;
            end
        end
        if best_match > 0
            associations(i) = best_match; % Store landmark index
        end
    end
end

