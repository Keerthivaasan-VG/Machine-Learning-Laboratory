import pandas as pd
import random

# Number of rows
rows = int(input("Enter number of rows: "))

data = []

for _ in range(rows):

    bedrooms = random.randint(1, 6)

    size = random.randint(600, 4000)      # Square feet

    age = random.randint(0, 40)           # Years

    zipcode = random.choice([600001,600002,600003,600004,600005])

    # Price Formula (Realistic)
    price = (
        size * 180 +
        bedrooms * 300000 -
        age * 8000 +
        (zipcode - 600000) * 150000 +
        random.randint(-200000, 200000)
    )

    data.append([bedrooms, size, age, zipcode, price])

df = pd.DataFrame(data, columns=[
    "Bedrooms",
    "Size_sqft",
    "Age",
    "Zipcode",
    "Price"
])

df.to_csv("house_prices.csv", index=False)

print(df.head())

print("\nDataset saved as house_prices.csv")