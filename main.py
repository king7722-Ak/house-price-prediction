import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("house_data.csv")

# Use all 7 features
X = data[['area', 'bedrooms', 'age', 'garage', 'pool', 'gym']]
y = data['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("✅ Model trained successfully")
print("RMSE:", rmse)

# Save the model
joblib.dump(model, 'house_price_model.pkl')
print("✅ Model saved as house_price_model.pkl")
