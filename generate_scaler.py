import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load your dataset (replace with your actual dataset path)
df = pd.read_csv('heart.csv')  # Make sure you have the correct dataset here

# Assume the dataset has a 'target' column and we want to scale the features
features = df.drop('target', axis=1)  # Drop the target column
scaler = StandardScaler()

# Fit the scaler on the features
scaler.fit(features)

# Save the scaler to a file
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as 'scaler.pkl'")
