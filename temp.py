import pandas as pd
df = pd.read_csv("data/raw/nba_2008-2025.csv")
print(df['date'].head(10))
print(df['date'].dtype)