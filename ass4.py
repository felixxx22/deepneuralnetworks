# import modules
import os
import pandas as pd
from scipy.stats import zscore

# import data
file='C:\\Users\\felix\\github\\deepneuralnetworks\\crx.csv'
df = pd.read_csv(file, na_values="?")

# aim: replace missing values in a2 and a14
# x data = 's3','a8','a9','a10','a11','a12','a13','a15'
# create dummy values

# drop unused columns
df = df.drop(['a1', 'a4', 'a5', 'a6', 'a7','a16'], axis=1)

# create x table
x_columns = df.columns.drop(['a2', 'a14'])
x = df[x_columns]

# create a2 table
y_a2 = df['a2']

# create a14 table
y_a14 = df['a14']

print(x)
print(y_a2)
print(y_a14)
