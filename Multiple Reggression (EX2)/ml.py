import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# --------------------------
# Load Dataset
# --------------------------

df = pd.read_csv("house_prices.csv")

print(df.head())

# --------------------------
# Features and Target
# --------------------------

X = df[["Bedrooms","Size_sqft","Age","Zipcode"]]

y = df["Price"]

# --------------------------
# Split Dataset
# --------------------------

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# --------------------------
# Train Model
# --------------------------

model = LinearRegression()

model.fit(X_train,y_train)

# --------------------------
# Prediction
# --------------------------

y_pred = model.predict(X_test)

# --------------------------
# Model Parameters
# --------------------------

print("\nIntercept")
print(model.intercept_)

print("\nCoefficients")

for feature,coef in zip(X.columns,model.coef_):
    print(feature,"=",coef)

# --------------------------
# Regression Equation
# --------------------------

print("\nRegression Equation")

print(
    f"Price = {model.intercept_:.2f}"
    f" + ({model.coef_[0]:.2f} × Bedrooms)"
    f" + ({model.coef_[1]:.2f} × Size)"
    f" + ({model.coef_[2]:.2f} × Age)"
    f" + ({model.coef_[3]:.2f} × Zipcode)"
)

# --------------------------
# Evaluation
# --------------------------

print("\nModel Performance")

print("Mean Error (ME):", np.mean(y_test - y_pred))
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
print("Relative RMSE (RELMSE):", (mean_squared_error(y_test, y_pred) ** 0.5) / np.mean(y_test))
print("RSS :", np.sum((y_test - y_pred) ** 2))
print("TSS :", np.sum((y_test - np.mean(y_test)) ** 2))
print("R² Score:", r2_score(y_test, y_pred))
print("CV Score:", ((mean_squared_error(y_test, y_pred) ** 0.5) / np.mean(y_test)) )

# --------------------------
# User Prediction
# --------------------------

bed = int(input("\nBedrooms: "))
size = int(input("House Size (sqft): "))
age = int(input("House Age: "))
zipc = int(input("Zipcode: "))

new_house = pd.DataFrame({
    "Bedrooms":[bed],
    "Size_sqft":[size],
    "Age":[age],
    "Zipcode":[zipc]
})

predicted_price = model.predict(new_house)

print("\nPredicted House Price = ₹{:,.0f}".format(predicted_price[0]))

# --------------------------
# Graph
# --------------------------

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred, color='blue')

# Ideal prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red',
    linewidth=2,
    label='Ideal Prediction'
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(True)
plt.show()