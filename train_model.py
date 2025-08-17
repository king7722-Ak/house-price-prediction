import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample dataset (replace with your real one)
data = pd.DataFrame({
    'area': [1000, 1500, 2000, 2500, 3000],
    'bedrooms': [2, 3, 4, 3, 5],
    'age': [10, 15, 20, 5, 25],
    'garage': [1, 2, 2, 1, 3],
    'bathrooms': [1, 2, 2, 1, 3],
    'pool': [0, 1, 1, 0, 1],
    'gym': [1, 1, 0, 0, 1],
    'price': [200000, 300000, 400000, 250000, 500000]
})

# Features and target
X = data[['area', 'bedrooms', 'age', 'garage', 'bathrooms', 'pool', 'gym']]
y = data['price']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, 'house_price_model.pkl')
print("✅ Model saved successfully.")
