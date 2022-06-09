import os
import pandas as pd
import numpy as np
from scipy.stats import zscore

# import data from csv file
df = pd.read_csv("http://data.heatonresearch.com/data/t81-558/datasets/reg-33-data.csv")

# remove id column
df.drop(labels=['id','weight', 'target', 'code', 'power','usage','region','convention','country'],axis=1,inplace=True)

# add column "ratio" that is max divided by number
df['ratio'] = [x/y for x,y in zip(df['max'], df['number'])]

# replace missing length values with median values
length_med = df['length'].median()
df['length'] = df['length'].fillna(length_med)

# replace height replace missing with median and convert to zscore.
height_med = df['height'].median()
df['height'] = df['height'].fillna(length_med)
df['height'] = zscore(df['height'])

# replace cat2 column with dummy variables
dummies_cat2 = pd.get_dummies(df['cat2'], prefix="cat2")
df = pd.concat([df,dummies_cat2], axis=1)

# replace items column with dummy variables
dummies_item = pd.get_dummies(df['item'], prefix="item")
df = pd.concat([df,dummies_item], axis=1)

print(df)
