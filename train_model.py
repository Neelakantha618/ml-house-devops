import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
data = pd.read_csv("dataset/housing.csv")

# Features
X = data[['area','bedrooms','bathrooms','age']]

# Target
y = data['price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = RandomForestRegressor()

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/house_model.pkl")

print("Model trained successfully!")
