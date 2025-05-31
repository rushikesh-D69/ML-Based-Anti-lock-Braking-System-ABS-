# ABS Braking Logic Simulation with Road Surface Prediction

[![MATLAB](https://img.shields.io/badge/MATLAB-R2024a-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![Simulink](https://img.shields.io/badge/Simulink-Model-orange.svg)](https://www.mathworks.com/products/simulink.html)

This repository contains a Simulink model for simulating ABS (Anti-lock Braking System) behavior based on road surface condition predictions using a machine learning model.

---

##  Files

- `untitled11.slx`: Simulink model file implementing the ABS braking logic.
- `ClassificationLearnerSession.mat`:Contains the trained ML model predicting road conditions.

---


## ⚙ Initial Parameters

Before running the simulation, the following variables **must be initialized** in the MATLAB workspace:



```matlab
g = 9.8;     % Acceleration due to gravity (m/s^2)
l = 5;       % Wheelbase or lever arm (meters)
m = 75;      % Mass of the vehicle (kg)
R = 1.25;    % Wheel radius (meters)
v0 = 44;     % Initial vehicle speed (km/h or m/s based on model)

