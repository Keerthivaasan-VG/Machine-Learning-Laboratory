# ============================================
# Simple Linear Regression - Ice Cream Sales
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("ice_cream_sales.csv")

print("First 5 Rows:")
print(df.head())

# -------------------------
# Independent and Dependent Variables
# -------------------------
X = df[['Temperature(C)']]
y = df['Ice_Creams_Sold']

# -------------------------
# Split Dataset
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -------------------------
# Train Model
# -------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------
# Predictions
# -------------------------
y_pred = model.predict(X_test)

# -------------------------
# Model Parameters
# -------------------------
print("\nIntercept:", model.intercept_)
print("Slope:", model.coef_[0])

# Regression Equation
print(f"\nRegression Equation:")
print(f"Sales = {model.coef_[0]:.2f} × Temperature + {model.intercept_:.2f}")

# -------------------------
# Model Evaluation
# -------------------------
print("\nModel Performance")
print("---------------------")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
print("R² Score:", r2_score(y_test, y_pred))

# -------------------------
# Predict New Value
# -------------------------
temp = float(input("\nEnter Temperature (°C): "))
prediction = model.predict([[temp]])

print(f"\nPredicted Ice Cream Sales: {prediction[0]:.0f}")

# -------------------------
# Regression Graph
# -------------------------
plt.figure(figsize=(8,6))

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')

plt.title("Simple Linear Regression")
plt.xlabel("Temperature (°C)")
plt.ylabel("Ice Creams Sold")
plt.legend()
plt.grid(True)

plt.show()