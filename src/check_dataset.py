import pandas as pd

df = pd.read_parquet("data/processed/features.parquet")

print(df.head().T)
print()
print(df.info())
print()
print(df.isna().sum())