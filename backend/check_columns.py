import pandas as pd

df = pd.read_csv(r"D:\Cognify\RestoSense\data\Cuisine_rating.csv", encoding="ISO-8859-1")

# Fix BOM
df.rename(columns={"ï»¿User ID": "User ID"}, inplace=True)

print(df.columns.tolist())
