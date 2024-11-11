import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv('crop_recommendation.csv')

# Features (N, P, K, temperature, humidity, pH, rainfall)
X = data.drop('label', axis=1)

# Target variable (crop label)
y = data['label']

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build a RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100)

# Train the model
model.fit(X_train, y_train)

# Save the model using pickle
with open('crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
