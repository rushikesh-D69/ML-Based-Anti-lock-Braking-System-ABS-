# ClassificationLearnerSession

This repository contains a saved MATLAB Classification Learner App session file: `ClassificationLearnerSession.mat`.
![image](https://github.com/user-attachments/assets/8cbe4531-69f5-4a91-853a-cb1f006ee9b1)

## Overview

The session file is created using MATLAB's Classification Learner App and includes:
- Imported training dataset (features and labels)
- Preprocessing settings (e.g., normalization)
- Trained classification models
- Cross-validation results and performance metrics (e.g., accuracy, confusion matrix)
- Possibly exported trained model(s)

## File Contents

- `ClassificationLearnerSession.mat`: A MATLAB session file containing:
  - Trained classification models (e.g., KNN, Decision Tree, SVM)
  - Training history and cross-validation results
  - Feature selection and transformation settings

## Usage

1. **Open in MATLAB**:
   - Launch MATLAB.
   - Use the command:
     ```matlab
     classificationLearner
     ```
   - In the Classification Learner App, open the `.mat` session using **File > Open Session**.

2. **Export Trained Model**:
   - Within the app, select the best-performing model.
   - Use **Export Model > Export Model to Workspace** to use the model in scripts or Simulink.

3. **Programmatic Use**:
   After exporting the model, you can use it in MATLAB like this:
   ```matlab
   label = trainedModel.predictFcn(newData);

