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

## Repository Structure

```plaintext
/
|- data/
|  |- data.mat           # Dataset including GNSS, lidar observations, and reference trajectories.
|
|- src/
|  |- ekf.m              # Extended Kalman Filter implementation (applied on simulated data sets).
|  |- ekfreal.m          # EKF applied on real data sets.

|  |- ukf.m              # Unscented Kalman Filter implementation (applied on simulated data sets).
|  |- ukf_real.m         # Unscented applied on real data sets
|
|- docs/
|  |- Project_Report.pdf # Detailed project report.
|
|- README.md

