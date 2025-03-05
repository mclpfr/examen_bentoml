import pandas as pd 
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
import os  
import bentoml

os.makedirs('data/raw', exist_ok=True)  
os.makedirs('data/processed', exist_ok=True) 

dataset_url = "https://assets-datascientest.s3.eu-west-1.amazonaws.com/MLOPS/bentoml/admission.csv"
raw_data_path = "data/raw/admission.csv"
data = pd.read_csv(raw_data_path)

print("\nMissing values per column:")
print(data.isnull().sum())

# Removing the "Serial No." column if it exists
if 'Serial No.' in data.columns:
    data = data.drop('Serial No.', axis=1)

# Defining features and target variable
X = data.drop('Chance of Admit ', axis=1)
y = data['Chance of Admit ']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler with bentoML
bentoml.sklearn.save_model("admission_scaler", scaler)
print("Scaler saved with bentoML")

# Saving the data in CSV format
print("Saving data in CSV format...")

# Converting normalized data to DataFrame
X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Saving files separately
X_train_df.to_csv('data/processed/X_train.csv', index=False)
X_test_df.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print("Data processing complete. CSV files saved in data/processed/ directory.")
