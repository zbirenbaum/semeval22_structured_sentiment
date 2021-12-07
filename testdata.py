import pandas as pd

df = pd.read_csv("data/training_tweets_processed.csv")
print(df["Target"].unique())
