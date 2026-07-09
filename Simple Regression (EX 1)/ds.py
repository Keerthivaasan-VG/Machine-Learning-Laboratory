import pandas as pd
import random

# -------------------------------
# Get number of rows from user
# -------------------------------
rows = int(input("Enter the number of rows: "))

temperature = []
ice_cream_sold = []

for _ in range(rows):
    # Generate temperature between 20°C and 45°C
    temp = round(random.uniform(20, 45), 1)

    # Realistic sales formula with random noise
    sales = int((temp * 6) + random.randint(-20, 20))

    # Ensure sales are not negative
    sales = max(sales, 0)

    temperature.append(temp)
    ice_cream_sold.append(sales)

# Create DataFrame
df = pd.DataFrame({
    "Temperature(C)": temperature,
    "Ice_Creams_Sold": ice_cream_sold
})

# Save to CSV
df.to_csv("ice_cream_sales.csv", index=False)

print("\nDataset Generated Successfully!")
print(df.head())
print("\nCSV saved as 'ice_cream_sales.csv'")
