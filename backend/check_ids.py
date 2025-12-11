import pandas as pd

df = pd.read_csv("models/restaurant_index.csv")
print("Total restaurants:", len(df))
print(df["Restaurant ID"].head(20))
print("Min ID:", df["Restaurant ID"].min())
print("Max ID:", df["Restaurant ID"].max())
