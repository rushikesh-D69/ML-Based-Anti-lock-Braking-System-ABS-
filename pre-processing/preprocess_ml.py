import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.impute import KNNImputer

# Read the cleaned data
print("Reading cleaned data...")
df = pd.read_csv('cleaned_road_data.csv', sep=';', decimal=',')

# Display initial information
print("\nInitial data shape:", df.shape)
print("\nColumns in dataset:")
print(df.columns.tolist())

# Analyze missing values
print("\nAnalyzing missing values...")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_info = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
})
print("\nMissing values analysis:")
print(missing_info[missing_info['Missing Values'] > 0].sort_values('Percentage', ascending=False))

# Select features for ML
numeric_features = [
    'VehicleSpeedInstantaneous',
    'LongitudinalAcceleration',
    'VerticalAcceleration',
    'MassAirFlow',
    'EngineLoad',
    'EngineRPM',
    'ManifoldAbsolutePressure',
    'IntakeAirTemperature'
]

# Handle missing values based on their characteristics
print("\nHandling missing values...")
for feature in numeric_features:
    missing_pct = (df[feature].isnull().sum() / len(df)) * 100
    print(f"\nProcessing {feature}:")
    print(f"Missing percentage: {missing_pct:.2f}%")
    
    if missing_pct > 30:  # If more than 30% missing
        print(f"Feature {feature} has too many missing values. Dropping...")
        numeric_features.remove(feature)
    else:
        # Use KNN imputation for features with moderate missing values
        imputer = KNNImputer(n_neighbors=5)
        df[feature] = imputer.fit_transform(df[[feature]])
        print(f"Imputed missing values using KNN")

print("\nFinal features after handling missing values:", numeric_features)

# Normalize numeric features
print("\nNormalizing numeric features...")
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Balance the classes
print("\nBalancing classes...")
# Get the minimum count among all classes
min_count = df['roadSurface'].value_counts().min()
print(f"Minimum class count: {min_count}")

# Create balanced dataset
balanced_dfs = []
for surface_type in df['roadSurface'].unique():
    surface_df = df[df['roadSurface'] == surface_type]
    if len(surface_df) > min_count:
        surface_df = surface_df.sample(n=min_count, random_state=42)
    balanced_dfs.append(surface_df)

# Combine balanced datasets
df_balanced = pd.concat(balanced_dfs, ignore_index=True)

# Shuffle the balanced dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced road surface distribution:")
print(df_balanced['roadSurface'].value_counts())

# Prepare target variable (roadSurface)
y = df_balanced['roadSurface']

# Prepare feature matrix
X = df_balanced[numeric_features]

# Split the data into training and testing sets
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save processed data
print("\nSaving processed data...")
# Save training data
train_data = pd.concat([X_train, y_train], axis=1)
train_data.to_csv('train_data.csv', index=False)

# Save test data
test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_csv('test_data.csv', index=False)

# Print summary
print("\nData preprocessing complete!")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print("\nTraining set class distribution:")
print(pd.Series(y_train).value_counts())
print("\nTest set class distribution:")
print(pd.Series(y_test).value_counts())
print("\nFeature statistics after normalization:")
print(X_train.describe())

# Save the scaler and imputer for later use
import joblib
joblib.dump(scaler, 'scaler.joblib')
print("\nScaler saved as 'scaler.joblib'") 