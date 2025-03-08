# Localization-using-poles-and-signs-detected-by-a-Lidar
Data fusion methods that improve the localization estimates of an autonomous vehicles equipped with a LIDAR and GNSS
# Introduction

This project leverages a 360° vehicle-mounted LiDAR along with GNSS to detect geo-referenced landmarks, such as poles and signs, enhancing positioning accuracy in urban environments. The primary focus is on implementing and evaluating the Extended Kalman Filter (EKF) and the Unscented Kalman Filter (UKF) to estimate vehicle trajectories accurately.

## Project Overview

The goals of this project include:
- Developing a stochastic state space representation for sensor fusion.
- Implementing and validating the EKF on simulated data.
- Integrating a data association algorithm to link LiDAR measurements with map landmarks.
- Implementing and evaluating the UKF on both simulated and real data.
- Comparing the performance of EKF and UKF in terms of accuracy and robustness.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YasMathlouthi/Localization-using-poles-and-signs-detected-by-a-Lidar

2.   **Navigate to the main files:**
-ekf2.m for EKF simulations.
-ekfreal.m for applying EKF on real data.
-ukf.m for UKF simulations.
-ukf_real.m for applying UKF on real data.
These scripts maintain the solution for each scenario and can be run in MATLAB to replicate the experiments.

3. **View results:** Numerical results and trajectories plots.

