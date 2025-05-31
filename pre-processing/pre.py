import pandas as pd
import os

# List of input CSV files
file_paths = [
    'archive/opel_corsa_01.csv',
    'archive/peugeot_207_02.csv',
    'archive/peugeot_207_01.csv',
    'archive/opel_corsa_02.csv'
]

# Read and combine datasets
dataframes = []
for path in file_paths:
    print(f"Reading file: {path}")
    try:
        df = pd.read_csv(path, sep=';', encoding='utf-8', decimal=',', on_bad_lines='skip')
        print(f"Shape of {path}: {df.shape}")
        df['source_file'] = os.path.basename(path)
        dataframes.append(df)
    except Exception as e:
        print(f"Error reading {path}: {str(e)}")

# Combine all into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)
print(f"\nCombined DataFrame shape: {combined_df.shape}")

# Print unique values in roadSurface column before encoding
print("\nUnique values in roadSurface column before encoding:")
print(combined_df['roadSurface'].unique())

# Create a copy of the original road surface values
combined_df['roadSurface_original'] = combined_df['roadSurface']

# Replace ',' with '.' in string columns, convert to numeric where possible
for col in combined_df.columns:
    if col != 'roadSurface' and col != 'roadSurface_original':  # Skip road surface columns
        if combined_df[col].dtype == 'object':
            combined_df[col] = combined_df[col].astype(str).str.replace(',', '.', regex=False)
            try:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
            except:
                pass

# Encode 'roadSurface' labels
combined_df['roadSurface'] = combined_df['roadSurface'].map({
    'SmoothCondition': 0,
    'UnevenCondition': 1,
    'FullOfHolesCondition': 2
})

# Print unique values in roadSurface column after encoding
print("\nUnique values in roadSurface column after encoding:")
print(combined_df['roadSurface'].unique())

# Print value counts for roadSurface
print("\nValue counts for roadSurface:")
print(combined_df['roadSurface'].value_counts())

# Drop rows with NaN in essential columns
critical_features = [
    'VehicleSpeedInstantaneous',
    'LongitudinalAcceleration',
    'VerticalAcceleration',
    'MassAirFlow',
    'EngineLoad',
    'roadSurface'
]
print("\nChecking for missing values in critical features:")
print(combined_df[critical_features].isnull().sum())

# Save all data to CSV (including rows with NaN values)
try:
    # First, verify the DataFrame is not empty
    if combined_df.empty:
        print("Error: DataFrame is empty!")
    else:
        # Save with explicit encoding and error handling
        combined_df.to_csv('cleaned_road_data.csv', 
                         index=False, 
                         sep=';', 
                         decimal=',',
                         encoding='utf-8')
        
        # Verify the file was created and has content
        if os.path.exists('cleaned_road_data.csv'):
            file_size = os.path.getsize('cleaned_road_data.csv')
            print(f"\nFile size: {file_size} bytes")
            
            # Read back the first few lines to verify content
            with open('cleaned_road_data.csv', 'r', encoding='utf-8') as f:
                first_lines = [next(f) for _ in range(5)]
            print("\nFirst few lines of output file:")
            for line in first_lines:
                print(line.strip())
        else:
            print("Error: Output file was not created!")
            
except Exception as e:
    print(f"Error saving file: {str(e)}")

print("\n✅ Data saved to 'cleaned_road_data.csv'")
print(f"✔️ Total rows: {combined_df.shape[0]}")
