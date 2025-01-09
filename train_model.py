import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your dataset (replace with your actual dataset path)
df = pd.read_csv('heart.csv')  # Make sure you have the correct dataset here

# Separate features and target
X = df.drop('target', axis=1)  # Features
y = df['target']  # Target (Heart Disease: 1, No Heart Disease: 0)

# Load the pre-trained scaler from 'scaler.pkl'
scaler = joblib.load('scaler.pkl')

# Scale the features using the loaded scaler
X_scaled = scaler.transform(X)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'heart_disease_model.pkl')
print("Model saved as 'heart_disease_model.pkl'")
