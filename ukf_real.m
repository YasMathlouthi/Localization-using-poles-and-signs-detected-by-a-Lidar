function ukf_real_data_with_two_lidar_types
    clear; clc;

    %% 1) Load real-world data
    load("data.mat"); % Replace with your real dataset file

    % Simulation parameters
    dt = 0.1;                % Time step
    N  = length(t);          % Number of time steps

    %% 2) Define process & measurement noise covariances
    Q = diag([0.1, 0.1, 0.0001, 0.1, 0.0001]);  % Process noise
    R_gnss  = diag([0.04, 0.04, 0.0001]);       % GNSS noise (x, y, theta)
    R_lidar = diag([0.1, 0.1]);                 % LiDAR noise (x, y)

    %% 3) State initialization
    % State = [ x; y; heading; v; omega ]
    x_hat = [ref(1).x; ref(1).y; ref(1).heading; 0; 0];
    P = eye(5);

    % Preallocate storage
    X_est = zeros(5, N);

    %% 4) Define the motion and measurement models
    % State transition:
    f = @(x, u) [
        x(1) + x(4)*cos(x(3))*dt;
        x(2) + x(4)*sin(x(3))*dt;
        x(3) + x(5)*dt;
        x(4) + u(1)*dt;
        x(5) + u(2)*dt
    ];

    % GNSS measurement model
    h_gnss = @(x) [ x(1); x(2); x(3) ];

    % LiDAR measurement model
    h_lidar = @(x, z) [
        x(1) + z(1)*cos(x(3)) - z(2)*sin(x(3));
        x(2) + z(1)*sin(x(3)) + z(2)*cos(x(3))
    ];

    %% 5) Parameters for Sigma Points
    w0 = 0.3;  % Weight for the center sigma point

    %% 6) UKF Loop with Two Types of LiDAR Observations
    for i = 1:N
        % --- Prediction Step ---
        [Xsig, wSig, nPts] = SigmaPoints_cholesky(x_hat, P, w0);

        % Propagate each sigma point
        u = [v(i); omega(i)];
        Xsig_pred = zeros(5, nPts);
        for k = 1:nPts
            Xsig_pred(:,k) = f(Xsig(:,k), u);
        end

        % Predicted state mean
        x_pred = zeros(5,1);
        for k = 1:nPts
            x_pred = x_pred + wSig(k)*Xsig_pred(:,k);
        end

        % Predicted covariance
        P_pred = zeros(5);
        for k = 1:nPts
            diff = Xsig_pred(:,k) - x_pred;
            P_pred = P_pred + wSig(k)*(diff*diff');
        end
        P_pred = P_pred + Q;

        % Update x_hat and P
        x_hat = x_pred;
        P     = P_pred;

        % --- Update Step: GNSS ---
        if ~isnan(gnss(i).x) && ~isnan(gnss(i).y) && ~isnan(gnss(i).heading)
            % Generate sigma points
            [Xsig, wSig, nPts] = SigmaPoints_cholesky(x_hat, P, w0);

            % Transform sigma points into measurement space
            Zsig = zeros(3, nPts);
            for k = 1:nPts
                Zsig(:,k) = h_gnss(Xsig(:,k));
            end

            % Predicted measurement mean
            z_pred = zeros(3,1);
            for k = 1:nPts
                z_pred = z_pred + wSig(k)*Zsig(:,k);
            end

            % Innovation covariance
            S = zeros(3);
            for k = 1:nPts
                diff_z = Zsig(:,k) - z_pred;
                S = S + wSig(k)*(diff_z*diff_z');
            end
            S = S + R_gnss;

            % Cross covariance
            Pxz = zeros(5,3);
            for k = 1:nPts
                diff_x = Xsig(:,k) - x_hat;
                diff_z = Zsig(:,k) - z_pred;
                Pxz = Pxz + wSig(k)*(diff_x*diff_z');
            end

            % Kalman gain
            K = Pxz / S;

            % Update state with GNSS measurement
            z_meas = [gnss(i).x; gnss(i).y; gnss(i).heading];
            y = z_meas - z_pred;  % Innovation
            x_hat = x_hat + K*y;
            P = P - K*S*K';
        end

        % --- Update Step: Poles Observations ---
        if ~isempty(poles_obs(i).x)
            associations = data_association(poles_obs(i), map, x_hat);
            for j = 1:length(associations)
                if isnan(associations(j))
                    continue; % Skip unmatched detections
                end
                z_lidar = [poles_obs(i).x(j); poles_obs(i).y(j)];
                z_map = map(associations(j), :)';

                % Generate sigma points
                [Xsig, wSig, nPts] = SigmaPoints_cholesky(x_hat, P, w0);

                % Transform sigma points to measurement space
                Zsig = zeros(2, nPts);
                for k = 1:nPts
                    Zsig(:,k) = h_lidar(Xsig(:,k), z_lidar);
                end

                % Predicted measurement mean
                z_pred = zeros(2,1);
                for k = 1:nPts
                    z_pred = z_pred + wSig(k)*Zsig(:,k);
                end

                % Innovation covariance
                S = zeros(2);
                for k = 1:nPts
                    diff_z = Zsig(:,k) - z_pred;
                    S = S + wSig(k)*(diff_z*diff_z');
                end
                S = S + R_lidar;

                % Cross covariance
                Pxz = zeros(5,2);
                for k = 1:nPts
                    diff_x = Xsig(:,k) - x_hat;
                    diff_z = Zsig(:,k) - z_pred;
                    Pxz = Pxz + wSig(k)*(diff_x*diff_z');
                end

                % Kalman gain
                K = Pxz / S;

                % Update state
                y = z_map - z_pred;  % Innovation
                x_hat = x_hat + K*y;
                P = P - K*S*K';
            end
        end

        % --- Update Step: Signs Observations ---
        if ~isempty(signs_obs(i).x)
            associations = data_association(signs_obs(i), map, x_hat);
            for j = 1:length(associations)
                if isnan(associations(j))
                    continue; % Skip unmatched detections
                end
                z_lidar = [signs_obs(i).x(j); signs_obs(i).y(j)];
                z_map = map(associations(j), :)';

                % Generate sigma points
                [Xsig, wSig, nPts] = SigmaPoints_cholesky(x_hat, P, w0);

                % Transform sigma points to measurement space
                Zsig = zeros(2, nPts);
                for k = 1:nPts
                    Zsig(:,k) = h_lidar(Xsig(:,k), z_lidar);
                end

                % Predicted measurement mean
                z_pred = zeros(2,1);
                for k = 1:nPts
                    z_pred = z_pred + wSig(k)*Zsig(:,k);
                end

                % Innovation covariance
                S = zeros(2);
                for k = 1:nPts
                    diff_z = Zsig(:,k) - z_pred;
                    S = S + wSig(k)*(diff_z*diff_z');
                end
                S = S + R_lidar;

                % Cross covariance
                Pxz = zeros(5,2);
                for k = 1:nPts
                    diff_x = Xsig(:,k) - x_hat;
                    diff_z = Zsig(:,k) - z_pred;
                    Pxz = Pxz + wSig(k)*(diff_x*diff_z');
                end

                % Kalman gain
                K = Pxz / S;

                % Update state
                y = z_map - z_pred;  % Innovation
                x_hat = x_hat + K*y;
                P = P - K*S*K';
            end
        end

        % Store state estimate
        X_est(:,i) = x_hat;
    end

    %% 7) Plot Results
    figure; hold on; grid on;
    plot([ref.x], [ref.y], 'k--', 'LineWidth', 1.5);
    plot([gnss.x], [gnss.y], 'b+', 'MarkerSize', 3);
    plot(X_est(1,:), X_est(2,:), 'r', 'LineWidth', 1.2);
    legend('Reference', 'GNSS', 'UKF Estimation');
    xlabel('East (m)'); ylabel('North (m)');
    title('UKF with Poles and Signs Observations');
    
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
end


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


