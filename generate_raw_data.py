"""Generates a realistic, deliberately messy retail sales dataset (synthetic)."""
import numpy as np, pandas as pd
rng = np.random.default_rng(2025)
n = 3000

dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
# seasonality: boost Nov/Dec by resampling extra
extra = pd.to_datetime("2025-11-01") + pd.to_timedelta(rng.integers(0, 61, 450), unit="D")
dates = pd.DatetimeIndex(np.concatenate([dates, extra]))
n = len(dates)

cat_products = {
    "Electronics": (["Headphones", "Smartwatch", "Bluetooth Speaker", "Tablet"], 180, 70),
    "Clothing":    (["T-Shirt", "Jeans", "Jacket", "Sneakers"], 55, 20),
    "Home & Kitchen": (["Air Fryer", "Cookware Set", "Desk Lamp", "Bedsheet Set"], 70, 25),
    "Beauty":      (["Face Cream", "Perfume", "Hair Dryer", "Lipstick Set"], 35, 12),
    "Sports":      (["Yoga Mat", "Dumbbell Set", "Running Shoes", "Water Bottle"], 45, 18),
}
cats = rng.choice(list(cat_products), n, p=[.22, .26, .20, .14, .18])
products, prices = [], []
for c in cats:
    plist, mu, sd = cat_products[c]
    products.append(rng.choice(plist))
    prices.append(max(5, rng.normal(mu, sd)))
region_p = {"North": .22, "South": .28, "East": .20, "West": .30}
region = rng.choice(list(region_p), n, p=list(region_p.values()))
qty = rng.choice([1, 2, 3, 4, 5], n, p=[.45, .25, .15, .10, .05]).astype(float)
disc = rng.choice([0, 5, 10, 15, 20, 25], n, p=[.35, .2, .2, .12, .08, .05]).astype(float)
age = np.clip(rng.normal(36, 11, n), 18, 70).round()
rating = np.clip(np.round(rng.normal(4.1, .8, n)), 1, 5)
gender = rng.choice(["Male", "Female"], n)
pay = rng.choice(["Credit Card", "UPI", "Debit Card", "Cash on Delivery", "Net Banking"], n, p=[.3, .3, .15, .15, .1])
price = np.round(prices, 2)

df = pd.DataFrame({
    "Order ID": [f"ORD{100000+i}" for i in range(n)],
    "Order Date": dates, "Customer ID": [f"CUST{rng.integers(1000, 2200)}" for _ in range(n)],
    "Customer Age": age, "Gender": gender, "Region": region, "Category": cats, "Product": products,
    "Quantity": qty, "Unit Price": price, "Discount %": disc,
    "Payment Method": pay, "Customer Rating": rating})
df["Sales"] = (df["Quantity"] * df["Unit Price"] * (1 - df["Discount %"] / 100)).round(2)

# ---------- inject mess ----------
def idx(k): return rng.choice(n, k, replace=False)
# mixed date formats
d = df["Order Date"].dt.strftime("%Y-%m-%d").astype(object)
i1, i2 = idx(500), idx(400)
d.iloc[i1] = df["Order Date"].iloc[i1].dt.strftime("%d/%m/%Y")
d.iloc[i2] = df["Order Date"].iloc[i2].dt.strftime("%b %d, %Y")
d.iloc[idx(15)] = "not available"
df["Order Date"] = d
# inconsistent text
df.loc[idx(150), "Region"] = df.loc[idx(150), "Region"].str.upper()
df["Region"] = df["Region"].astype(object)
for i in idx(120): df.at[i, "Region"] = str(df.at[i, "Region"]).lower() + " "
df["Gender"] = df["Gender"].astype(object)
for i in idx(200): df.at[i, "Gender"] = {"Male": "M", "Female": "F"}[df.at[i, "Gender"]]
for i in idx(80):  df.at[i, "Gender"] = df.at[i, "Gender"].lower()
for i in idx(100): df.at[i, "Payment Method"] = {"UPI": "upi", "Credit Card": "CREDIT CARD", "Debit Card": "debit card",
                                                   "Cash on Delivery": "COD", "Net Banking": "net banking"}[df.at[i, "Payment Method"]]
for i in idx(70):  df.at[i, "Category"] = df.at[i, "Category"].replace("Home & Kitchen", "Home and Kitchen").upper()
# sales as currency strings
df["Sales"] = df["Sales"].astype(object)
for i in idx(250): df.at[i, "Sales"] = f"${df.at[i, 'Sales']:,.2f}"
# missing values
for col, k in [("Customer Age", 220), ("Region", 90), ("Customer Rating", 380),
               ("Discount %", 140), ("Payment Method", 110), ("Gender", 60), ("Sales", 70), ("Unit Price", 40)]:
    df.loc[idx(k), col] = np.nan
# invalid / outlier values
df.loc[idx(12), "Customer Age"] = rng.choice([-5, 0, 150, 250, 999], 12)
df.loc[idx(15), "Quantity"] = rng.choice([-3, 0, 50, 100], 15)
df.loc[idx(14), "Unit Price"] = rng.choice([4999.99, 9999.0, 12000.0, -45.0], 14)
df.loc[idx(10), "Customer Rating"] = rng.choice([0, 7, 10], 10)
df.loc[idx(10), "Discount %"] = rng.choice([90, 100, -10], 10)
# duplicates
df = pd.concat([df, df.sample(110, random_state=3)], ignore_index=True)
df = df.sample(frac=1, random_state=9).reset_index(drop=True)
df.to_csv("data/raw_sales_data.csv", index=False)
print(df.shape)
