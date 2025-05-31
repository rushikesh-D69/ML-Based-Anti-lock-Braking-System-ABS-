# ML-Enhanced Anti-lock Braking System (ABS)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2024a-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![Status: Research](https://img.shields.io/badge/Status-Research-red.svg)]()

##  Overview

A novel machine learning-enhanced Anti-lock Braking System that uses real-time road surface classification to adaptively control brake pressure. This system integrates ML-based road condition prediction with traditional ABS control logic to improve braking performance across diverse road surfaces.

### Key Features
- **Real-time Classification**: 56,000+ predictions per second
- **Compact Model**: 875 kB model size suitable for automotive ECUs
- **High Accuracy**: 85.6% test accuracy across road surface conditions
- **Automotive Ready**: Compatible with existing ABS hardware architectures

  
![image](https://github.com/user-attachments/assets/6b97cb10-e17b-404e-8f21-d8d7d52379df)



## 📊 Performance Results

| Metric | Value |
|--------|--------|
| **Validation Accuracy** | 84.4% |
| **Test Accuracy** | 85.6% |
| **Model Size (Compact)** | 875 kB |
| **Prediction Speed** | ~56,000 obs/sec |
| **Training Time** | 33.915 sec |
| **Error Rate** | 14.4% |



##  System Architecture

```
Vehicle Sensors → Feature Extraction → ML Classifier → Friction Estimator → ABS Controller → Brake Modulation
     ↓                    ↓                ↓              ↓                 ↓              ↓
  8 Parameters    Preprocessing    Road Surface    Friction Coeff.    Slip Control    Optimal Braking
                                  Classification      (μ values)
```

![Machine Learning-Enhanced Anti-lock Braking System_ Real-time Road Surface Classification for Adaptive Brake Control - visual selection](https://github.com/user-attachments/assets/7c0ca2d4-9dcc-4fb0-a565-e9a4755797bc)


### Input Features (8 Parameters)
- Vehicle Speed (km/h)
- Longitudinal Acceleration (m/s²)
- Vertical Acceleration (m/s²)
- Mass Air Flow (g/s)
- Engine Load (%)
- Engine RPM
- Manifold Absolute Pressure (kPa)
- Intake Air Temperature (°C)

### Road Surface Classes
- **Class 0**: Smooth Condition (μ = 0.8) - Dry asphalt, high friction
- **Class 1**: Uneven Condition (μ = 0.4) - Wet pavement, moderate friction
- **Class 2**: Full of Holes (μ = 0.1) - Icy/gravel surfaces, low friction



# Real-time prediction
sensor_data = {
    'vehicle_speed': 45.0,
    'longitudinal_accel': -2.1,
    'vertical_accel': 0.3,
    'mass_air_flow': 15.2,
    'engine_load': 35.0,
    'engine_rpm': 2100,
    'manifold_pressure': 85.0,
    'intake_temp': 25.0
}

# Predict road surface and get friction coefficient
road_class = classifier.predict(sensor_data)
friction_coeff = abs_system.get_friction_coefficient(road_class)

print(f"Predicted road surface: {road_class}")
print(f"Friction coefficient: {friction_coeff}")

##  How It Works

1. **Preprocessing (Python):**
   - Normalize and impute vehicle sensor data
   - Balance classes for road surface

2. **Model Training (MATLAB):**
   - Trained using Classification Learner
   - Exported to `roadPredictor.mdl`

3. **Simulation (Simulink):**
   - Vehicle dynamics modeled
   - Friction is estimated from predicted class
   - Braking control adjusts based on slip and μ

![image](https://github.com/user-attachments/assets/82da4104-ea81-456a-ae23-b7766305a719)


##  Results

-  Vehicle comes to rest smoothly based on predicted road surface
-  Speed vs Time and Slip vs Time graphs validate controller
-  High classification accuracy of road surface conditions

<p align="center">
  <img src="https://github.com/user-attachments/assets/57c468d3-89ce-467b-aac6-15504ad5811a" alt="ScopePrintToFigure" width="30%">
  <img src="https://github.com/user-attachments/assets/11d2f01f-f301-4005-affa-545d8e39375f" alt="354" width="30%">
  <img src="https://github.com/user-attachments/assets/31410141-b4a8-4077-a68c-810654889966" alt="12" width="30%">
</p>





### Abstract

This research presents a novel approach to enhancing vehicle braking performance by integrating machine learning-based road surface classification with traditional Anti-lock Braking System technology. The ML model achieved 84.4% validation accuracy and 85.6% test accuracy with a compact 875 kB model size, enabling real-time deployment on automotive ECUs with prediction speeds of ~56,000 observations per second.


##  References

1. Wellstead, P. E., & Pettit, N. B. (1997). Analysis and redesign of an antilock brake system controller. *IEE Proceedings-Control Theory and Applications*.

2. Mirzaei, M., & Mirzaeinejad, H. (2010). Fuzzy scheduled optimal control of integrated vehicle braking and steering systems. *IEEE/ASME Transactions on Mechatronics*.

3. Zhang, J., Chen, H., & Guo, K. (2018). Sliding mode control for antilock braking system with road surface identification. *International Journal of Automotive Technology*.

4.https://www.youtube.com/watch?v=G8VbGtLOmR8

---

**⭐ Star this repository if you find it useful!**

**🍴 Fork it to contribute to the project!**
