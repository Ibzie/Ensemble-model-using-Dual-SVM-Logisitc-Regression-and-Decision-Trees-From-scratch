import os
import json
import numpy as np
import pandas as pd
from docx import Document
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from Dual_SVM import KernelSVM, cross_validation  # Ensure you import your SVM class and cross-validation function correctly

folder_path = 'Flights Data'
all_flight_data = []
for file_name in os.listdir(folder_path):
    if file_name.endswith('.docx'):
        file_path = os.path.join(folder_path, file_name)
        doc = Document(file_path)
        full_text = ''
        for para in doc.paragraphs:
            full_text += para.text
        
        # Attempt to parse the text as JSON
        try:
            flight_data = json.loads(full_text)

            if isinstance(flight_data, list):
                all_flight_data.extend(flight_data)
            else:
                all_flight_data.append(flight_data)
        except json.JSONDecodeError:
            print(f"Error parsing JSON data from file: {file_name}")
            continue

df = pd.DataFrame(all_flight_data)

df_flattened = pd.json_normalize(df)

df_flattened.columns = df_flattened.columns.str.replace('.', '_')

df_flattened.replace({np.nan: None}, inplace=True)

def flatten_nested_columns(df):
    """
    Flattens all columns in a DataFrame that contain nested lists or dictionaries.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
        
    Returns:
        pandas.DataFrame: The DataFrame with nested columns flattened.
    """
    flattened_cols = []
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            flattened = pd.json_normalize(df[col])
            flattened.columns = [f"{col}_{c}" if c else col for c in flattened.columns]
            flattened_cols.append(flattened)
        else:
            flattened_cols.append(df[col])
    
    return pd.concat(flattened_cols, axis=1)

filtered_df = flatten_nested_columns(df)

# Drop unnecessary columns
columns_to_drop = [
    'arrival_actualTime', 'arrival_estimatedRunway', 'arrival_actualRunway',
    'codeshared_airline.name', 'codeshared_airline.iataCode', 'codeshared_airline.icaoCode',
    'codeshared_flight.number', 'codeshared_flight.iataNumber', 'codeshared_flight.icaoNumber',
    'departure_gate', 'departure_estimatedRunway', 'departure_actualRunway',
    'departure_terminal', 'arrival_estimatedTime', 'arrival_terminal','arrival_baggage', 
    'arrival_gate', 'departure_actualTime', 'departure'
]
filtered_df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

# Fill NaN values in numeric columns with mean and in categorical columns with mode
filtered_df = filtered_df.apply(lambda col: col.fillna(col.mean()) if col.dtype in ['float64', 'int64'] else col.fillna(col.mode()[0]))

# Convert columns to numeric, drop non-numeric columns
numeric_df = filtered_df.select_dtypes(include=['float64', 'int64'])  # Keep only numeric columns

# Bin departure_delay to create delay_bin target
bin_edges = [-np.inf, 0, 5, 10, 20, 30, 60, 120, np.inf]
bin_labels = ['On Time', '0-5 mins', '5-10 mins', '10-20 mins', '20-30 mins', '30-60 mins', '60-120 mins', '120+ mins']
numeric_df['delay_bin'] = pd.cut(filtered_df['departure_delay'], bins=bin_edges, labels=bin_labels)

# Label encode categorical columns
categorical_cols = ['status', 'departure_iataCode', 'departure_icaoCode', 'arrival_iataCode', 
                    'arrival_icaoCode', 'airline_name', 'airline_iataCode', 'airline_icaoCode', 
                    'flight_iataNumber', 'flight_icaoNumber']
le = LabelEncoder()
for col in categorical_cols:
    if col in filtered_df.columns:  # Check if the column exists
        filtered_df[col] = le.fit_transform(filtered_df[col])

# Label encode target variable
le_target = LabelEncoder()
numeric_df['delay_bin'] = le_target.fit_transform(numeric_df['delay_bin'])

# Include encoded categorical columns
numeric_df = pd.concat([numeric_df, filtered_df[categorical_cols]], axis=1)

# Randomly sample 5000 entries
sampled_df = numeric_df.sample(n=5000, random_state=69)

# Split data into features and target variable
X = sampled_df.drop(columns=['delay_bin']).values  # Convert to NumPy array
y = sampled_df['delay_bin'].values

# Print data types in X to diagnose issue
print("Data types in X before scaling:\n", sampled_df.drop(columns=['delay_bin']).dtypes)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize the KernelSVM model
svm = KernelSVM(C=1.0, kernel='linear', gamma=0.5)  # Adjust kernel and parameters as needed

# Perform k-fold cross-validation
cross_validation(svm, X, y, k=5)

###########################################################################################################
