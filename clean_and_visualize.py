"""
Data Cleaning & Visualization Project
Retail Sales Dataset  ->  clean  ->  analyze  ->  visualize
Run:  python clean_and_visualize.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.0)
RAW, OUT = "data/raw_sales_data.csv", Path("outputs")
OUT.mkdir(exist_ok=True)
log = []

# ============================================================ 1. LOAD & INSPECT
raw = pd.read_csv(RAW)
raw_missing = raw.isna().sum()
print("RAW shape:", raw.shape, "| duplicates:", raw.duplicated().sum())
print(raw.dtypes, "\n", raw_missing[raw_missing > 0])

df = raw.copy()
df.columns = (df.columns.str.strip().str.lower()
              .str.replace("%", "pct").str.replace(" ", "_"))
log.append(f"Raw data: {raw.shape[0]} rows x {raw.shape[1]} columns")

# ============================================================ 2. DUPLICATES
n = len(df)
df = df.drop_duplicates().reset_index(drop=True)
log.append(f"Duplicates: removed {n - len(df)} exact duplicate rows")

# ============================================================ 3. TEXT CONSISTENCY
text_cols = ["gender", "region", "category", "product", "payment_method"]
for c in text_cols:
    df[c] = df[c].astype("string").str.strip()
    # placeholder strings that really mean "missing"
    df[c] = df[c].mask(df[c].str.lower().isin(["nan", "n/a", "na", "none", "null", ""]))
df["gender"] = df["gender"].str.lower().map({"m": "Male", "male": "Male", "f": "Female", "female": "Female"})
df["region"] = df["region"].str.title()
df["category"] = (df["category"].str.title().str.replace("Home And Kitchen", "Home & Kitchen"))
df["payment_method"] = (df["payment_method"].str.lower()
    .map({"upi": "UPI", "credit card": "Credit Card", "debit card": "Debit Card",
          "cod": "Cash on Delivery", "cash on delivery": "Cash on Delivery",
          "net banking": "Net Banking"}))
log.append("Text: standardised case/spacing/aliases in gender, region, category, payment_method; converted placeholder strings such as 'nan' to real missing values")

# ============================================================ 4. DATA TYPES
# sales stored as text like "$1,019.45" in some rows
df["sales"] = pd.to_numeric(df["sales"].astype("string").str.replace(r"[$,]", "", regex=True), errors="coerce")

def parse_date(s):
    if pd.isna(s): return pd.NaT
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y"):
        try: return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError): pass
    return pd.NaT
df["order_date"] = df["order_date"].map(parse_date)
bad_dates = df["order_date"].isna().sum()
df = df.dropna(subset=["order_date"]).reset_index(drop=True)
log.append(f"Types: converted sales '$' strings to numbers; parsed 3 date formats; "
           f"dropped {bad_dates} rows with unparseable dates")

# ============================================================ 5. INVALID VALUES -> NaN
rules = {"customer_age": (18, 80), "quantity": (1, 10), "unit_price": (1, 2000),
         "customer_rating": (1, 5), "discount_pct": (0, 50)}
for col, (lo, hi) in rules.items():
    bad = (~df[col].between(lo, hi)) & df[col].notna()
    df.loc[bad, col] = np.nan
    log.append(f"Invalid: {bad.sum()} impossible values in '{col}' (outside {lo}-{hi}) set to missing")

# ============================================================ 6. MISSING VALUES
before_missing = df.isna().sum()
# categorical -> mode ; numeric -> median (robust)
k = df["region"].isna().sum(); df["region"] = df["region"].fillna("Unknown")
log.append(f"Missing: {k} in 'region' labelled 'Unknown' (a location cannot be guessed, so no mode fill)")
for c in ["gender", "payment_method"]:
    k = df[c].isna().sum(); df[c] = df[c].fillna(df[c].mode()[0])
    log.append(f"Missing: {k} in '{c}' filled with mode")
df["customer_age"] = df["customer_age"].fillna(df["customer_age"].median())
df["customer_rating"] = df["customer_rating"].fillna(df["customer_rating"].median())
df["discount_pct"] = df["discount_pct"].fillna(0)
# unit price: median of the same product (prices differ a lot by product)
df["unit_price"] = df["unit_price"].fillna(df.groupby("product")["unit_price"].transform("median"))
df["quantity"] = df["quantity"].fillna(df["quantity"].median())
log.append("Missing: age/rating/quantity -> median, discount -> 0, unit_price -> product median")

# ============================================================ 7. OUTLIERS (IQR, per category)
price_before = df["unit_price"].copy()
def cap_iqr(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return s.clip(q1 - 1.5 * i, q3 + 1.5 * i)
df["unit_price"] = df.groupby("category")["unit_price"].transform(cap_iqr)
n_cap = (price_before != df["unit_price"]).sum()
log.append(f"Outliers: capped {n_cap} unit_price values using IQR rule within each category")

# ============================================================ 8. RECOMPUTE SALES
calc = (df["quantity"] * df["unit_price"] * (1 - df["discount_pct"] / 100)).round(2)
fixed = df["sales"].isna().sum()
mismatch = ((df["sales"] - calc).abs() > 1).sum()
df["sales"] = calc          # single source of truth after cleaning
log.append(f"Sales: {fixed} missing + {mismatch} inconsistent values recomputed = quantity x price x (1-discount)")

# ============================================================ 9. FEATURES
df["month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
df["age_group"] = pd.cut(df["customer_age"], [17, 25, 35, 45, 55, 80],
                         labels=["18-25", "26-35", "36-45", "46-55", "56+"])
df = df.sort_values("order_date").reset_index(drop=True)
assert df.isna().sum().sum() == 0 and df.duplicated().sum() == 0
log.append(f"Final clean data: {df.shape[0]} rows x {df.shape[1]} columns, 0 missing, 0 duplicates")
df.to_csv("data/cleaned_sales_data.csv", index=False)
Path(OUT / "cleaning_log.txt").write_text("\n".join(log))

# ============================================================ 10. INSIGHTS
total = df["sales"].sum()
by_cat = df.groupby("category")["sales"].sum().sort_values(ascending=False)
by_reg = df.groupby("region")["sales"].sum().sort_values(ascending=False)
monthly = df.groupby("month")["sales"].sum()
disc_grp = df.assign(disc=pd.cut(df["discount_pct"], [-1, 0, 10, 20, 100], labels=["0%", "1-10%", "11-20%", "21%+"]))
metrics = {
    "total_sales": round(total, 2), "orders": len(df), "aov": round(df["sales"].mean(), 2),
    "avg_rating": round(df["customer_rating"].mean(), 2),
    "top_category": by_cat.index[0], "top_category_share": round(by_cat.iloc[0] / total * 100, 1),
    "top_region": by_reg.index[0], "top_region_share": round(by_reg.iloc[0] / total * 100, 1),
    "best_month": str(monthly.idxmax().strftime("%B")), "best_month_sales": round(monthly.max(), 2),
    "worst_month": str(monthly.idxmin().strftime("%B")), "worst_month_sales": round(monthly.min(), 2),
    "nov_dec_share": round(monthly[monthly.index.month.isin([11, 12])].sum() / total * 100, 1),
    "avg_order_by_category": df.groupby("category")["sales"].mean().round(2).to_dict(),
    "sales_by_category": by_cat.round(0).to_dict(), "sales_by_region": by_reg.round(0).to_dict(),
    "avg_order_by_discount": disc_grp.groupby("disc", observed=True)["sales"].mean().round(2).to_dict(),
    "rating_by_category": df.groupby("category")["customer_rating"].mean().round(2).to_dict(),
    "sales_by_age_group": df.groupby("age_group", observed=True)["sales"].sum().round(0).to_dict(),
    "sales_by_payment": df.groupby("payment_method")["sales"].sum().round(0).to_dict(),
    "top_products": df.groupby("product")["sales"].sum().nlargest(5).round(0).to_dict(),
}
Path(OUT / "key_metrics.json").write_text(json.dumps(metrics, indent=2))

# ============================================================ 11. CHARTS
PAL = sns.color_palette("viridis", 6)
def save(fig, name): fig.tight_layout(); fig.savefig(OUT / name, dpi=150, bbox_inches="tight"); plt.close(fig)

# 11a data quality: before vs after
fig, ax = plt.subplots(figsize=(9, 4.5))
mm = pd.DataFrame({"Before cleaning": raw_missing, "After cleaning": 0}).rename(index=lambda x: x)
mm = mm[mm["Before cleaning"] > 0].sort_values("Before cleaning")
mm.plot.barh(ax=ax, color=["#e07a5f", "#3d9970"])
ax.set_title("Missing values per column - before vs after cleaning"); ax.set_xlabel("Missing cells")
save(fig, "01_missing_values_before_after.png")

# 11b outliers before/after (unit price)
raw_price = pd.to_numeric(raw["Unit Price"], errors="coerce")
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
sns.boxplot(y=raw_price, ax=ax[0], color="#e07a5f"); ax[0].set_title("Unit price - raw (extreme outliers)")
sns.boxplot(x="category", y="unit_price", data=df, ax=ax[1], palette="viridis", hue="category", legend=False)
ax[1].set_title("Unit price - cleaned, by category"); ax[1].tick_params(axis="x", rotation=25)
save(fig, "02_outliers_before_after.png")

# 11c monthly trend
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(monthly.index, monthly.values, marker="o", lw=2.5, color="#2a9d8f")
ax.fill_between(monthly.index, monthly.values, alpha=.15, color="#2a9d8f")
ax.set_title("Monthly sales trend, 2025"); ax.set_ylabel("Sales ($)")
ax.yaxis.set_major_formatter(lambda v, _: f"${v/1000:.0f}K")
save(fig, "03_monthly_sales_trend.png")

# 11d category + region
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
by_cat.sort_values().plot.barh(ax=ax[0], color=PAL[2]); ax[0].set_title("Sales by category")
ax[0].xaxis.set_major_formatter(lambda v, _: f"${v/1000:.0f}K")
ax[1].pie(by_reg, labels=by_reg.index, autopct="%1.0f%%", colors=PAL[:5], startangle=90,
          wedgeprops=dict(width=.45, edgecolor="white")); ax[1].set_title("Sales share by region")
save(fig, "04_category_and_region.png")

# 11e discount vs order value, age
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
disc_grp.groupby("disc", observed=True)["sales"].mean().plot.bar(ax=ax[0], color=PAL[3], rot=0)
ax[0].set_title("Avg order value by discount level"); ax[0].set_xlabel("Discount"); ax[0].set_ylabel("Avg order ($)")
sns.histplot(df["customer_age"], bins=20, kde=True, ax=ax[1], color=PAL[1]); ax[1].set_title("Customer age distribution")
save(fig, "05_discount_and_age.png")

# 11f heatmap category x month
pivot = df.pivot_table(index="category", columns=df["order_date"].dt.month, values="sales", aggfunc="sum")
fig, ax = plt.subplots(figsize=(11, 4.5))
sns.heatmap(pivot / 1000, cmap="YlGnBu", annot=True, fmt=".0f", cbar_kws={"label": "Sales ($K)"}, ax=ax)
ax.set_title("Sales ($K) by category and month"); ax.set_xlabel("Month")
save(fig, "06_category_month_heatmap.png")

# 11g DASHBOARD
fig = plt.figure(figsize=(18, 11)); fig.patch.set_facecolor("#f7f7f2")
gs = fig.add_gridspec(3, 3, height_ratios=[.55, 1.4, 1.4], hspace=.45, wspace=.28)
fig.suptitle("Retail Sales 2025 - Cleaned Data Dashboard", fontsize=22, fontweight="bold", y=.98)
kpis = [("Total Sales", f"${total:,.0f}"), ("Orders", f"{len(df):,}"),
        ("Avg Order Value", f"${metrics['aov']:,.0f}"), ("Avg Rating", f"{metrics['avg_rating']:.2f} / 5")]
kax = fig.add_subplot(gs[0, :]); kax.axis("off")
for i, (k, v) in enumerate(kpis):
    kax.text(.125 + i * .25, .6, v, ha="center", fontsize=26, fontweight="bold", color="#264653")
    kax.text(.125 + i * .25, .1, k, ha="center", fontsize=13, color="#555")
a = fig.add_subplot(gs[1, :2]); a.plot(monthly.index, monthly.values, marker="o", lw=2.5, color="#2a9d8f")
a.fill_between(monthly.index, monthly.values, alpha=.15, color="#2a9d8f"); a.set_title("Monthly sales trend")
a.yaxis.set_major_formatter(lambda v, _: f"${v/1000:.0f}K")
b = fig.add_subplot(gs[1, 2]); b.pie(by_reg, labels=by_reg.index, autopct="%1.0f%%", colors=PAL[:5], startangle=90,
        wedgeprops=dict(width=.45, edgecolor="white")); b.set_title("Sales share by region")
c = fig.add_subplot(gs[2, 0]); by_cat.sort_values().plot.barh(ax=c, color=PAL[2]); c.set_title("Sales by category")
c.xaxis.set_major_formatter(lambda v, _: f"${v/1000:.0f}K"); c.set_ylabel("")
d = fig.add_subplot(gs[2, 1]); disc_grp.groupby("disc", observed=True)["sales"].mean().plot.bar(ax=d, color=PAL[3], rot=0)
d.set_title("Avg order value by discount"); d.set_xlabel("")
e = fig.add_subplot(gs[2, 2]); df.groupby("payment_method")["sales"].sum().sort_values().plot.barh(ax=e, color=PAL[4])
e.set_title("Sales by payment method"); e.xaxis.set_major_formatter(lambda v, _: f"${v/1000:.0f}K"); e.set_ylabel("")
fig.savefig(OUT / "00_dashboard.png", dpi=150, bbox_inches="tight"); plt.close(fig)

print("\n".join(log)); print("\nDone -> outputs/ and data/cleaned_sales_data.csv")
