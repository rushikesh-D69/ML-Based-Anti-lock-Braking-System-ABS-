# ABS Braking Logic Simulation with Road Surface Prediction

[![MATLAB](https://img.shields.io/badge/MATLAB-R2024a-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![Simulink](https://img.shields.io/badge/Simulink-Model-orange.svg)](https://www.mathworks.com/products/simulink.html)

This repository contains a Simulink model for simulating ABS (Anti-lock Braking System) behavior based on road surface condition predictions using a machine learning model.

---

##  Files

- `untitled11.slx`: Simulink model file implementing the ABS braking logic.
- `ClassificationLearnerSession.mat`: (if included) Contains the trained ML model predicting road conditions.

---

##  Model Overview

<p align="center">
  <img src="https://github.com/user-attachments/assets/d49179eb-b81e-4014-84ee-2166a969af5d" alt="Simulink Model" width="90%">
</p>

---

## ⚙ Initial Parameters

Before running the simulation, the following variables **must be initialized** in the MATLAB workspace:

<p align="center">
  <img src="https://github.com/user-attachments/assets/ba2eb764-8d2a-4030-b23f-1c255cc52717" alt="Initial Parameters" width="40%">
</p>

```matlab
g = 9.8;     % Acceleration due to gravity (m/s^2)
l = 5;       % Wheelbase or lever arm (meters)
m = 75;      % Mass of the vehicle (kg)
R = 1.25;    % Wheel radius (meters)
v0 = 44;     % Initial vehicle speed (km/h or m/s based on model)

