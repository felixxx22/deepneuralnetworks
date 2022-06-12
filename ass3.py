# import modules
import os
import pandas as pd
from scipy.stats import zscore
import datetime as dt

# import file as pandas dataframe
file = "https://data.heatonresearch.com/data/t81-558/datasets/series-31.csv"
df = pd.read_csv(file)

#convert time to datetime object
df['time'] = pd.to_datetime(df['time'], errors='coerce')

# add day column to allow data to be grouped by days
df['date'] = df['time'].apply(lambda t: t.replace(hour=0, minute=0,second=0))

# group data by days
days_grouped = df.groupby(by='date')

# sort grouped array to find max, min, start and end values for each group (day)
max = days_grouped['value'].max()
min = days_grouped['value'].min()

time_start = days_grouped.min()
time_end = days_grouped.max()

start = time_start['value']
end = time_end['value']

# create final array
df = pd.concat([start, max, min, end], axis=1, keys=['starting', 'max', 'min','ending'])

print(df)
