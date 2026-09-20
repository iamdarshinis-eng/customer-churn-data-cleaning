import pandas as pd

INPUT_FILE = "customer_churn_sample (1)(1).csv"
OUTPUT_FILE = "customer_churn_cleaned.csv"

# 1. Load dataset
df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# 2. Standardize column headers
df.columns = (
    df.columns.str.strip()
    .str.replace(r'(?<=[a-z0-9])(?=[A-Z])', '_', regex=True)
    .str.replace(r'[^A-Za-z0-9]+', '_', regex=True)
    .str.strip('_')
    .str.lower()
)

# 3. Clean string/categorical columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype("string").str.strip()

# 4. Validate numeric columns
numeric_cols = [
    "age", "tenure_months", "monthly_charges",
    "total_charges", "support_tickets"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 5. Handle duplicates
# Exact duplicate rows are removed. In this dataset there are none.
df = df.drop_duplicates().reset_index(drop=True)

# 6. Handle missing values
# Inspection showed zero missing values, so no rows/values were imputed or dropped.
# If missing values occur in a future version:
# numeric columns -> median; categorical columns -> mode.

# 7. Validate ranges
assert df["age"].between(0, 120).all(), "Invalid age found"
assert (df["tenure_months"] >= 0).all(), "Invalid tenure found"
assert (df["monthly_charges"] >= 0).all(), "Invalid monthly charges found"
assert (df["total_charges"] >= 0).all(), "Invalid total charges found"
assert (df["support_tickets"] >= 0).all(), "Invalid support ticket count found"

# 8. Validate categorical values
assert set(df["gender"].dropna().unique()) <= {"Female", "Male"}
assert set(df["subscription_type"].dropna().unique()) <= {"Basic", "Pro", "Enterprise"}
assert set(df["contract_type"].dropna().unique()) <= {"Month-to-Month", "One Year", "Two Year"}
assert set(df["payment_method"].dropna().unique()) <= {"Credit Card", "Bank Transfer", "UPI", "Debit Card"}
assert set(df["churn"].dropna().unique()) <= {"Yes", "No"}

# 9. Export
df.to_csv(OUTPUT_FILE, index=False)

print("\nCleaned shape:", df.shape)
print("Final missing values:", int(df.isnull().sum().sum()))
print("Final duplicate rows:", int(df.duplicated().sum()))
print(f"\nCleaned file saved as: {OUTPUT_FILE}")
