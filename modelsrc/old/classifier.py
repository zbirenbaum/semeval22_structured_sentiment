from df_builder import build_df
import pandas as pd

df = build_df()
data_dict = df.to_dict()
print(data_dict)