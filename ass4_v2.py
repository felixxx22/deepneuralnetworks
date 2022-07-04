# import modules
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import metrics
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridgex

# import data
file='C:\\Users\\felix\\github\\deepneuralnetworks\\crx.csv'
df = pd.read_csv(file, na_values=["?"])

# aim: replace missing values in a2 and a14
# x data = 's3','a8','a9','a10','a11','a12','a13','a15'
# create dummy values
# drop na values for modelling
df = df.dropna(axis = 0, how ='any', thresh = None, subset = None, inplace=False)

# drop unused columns
df = df.drop(['a1', 'a4', 'a5', 'a6', 'a7','a16'], axis=1)

# create dummy variables for dataframe
df = pd.get_dummies(df)

# standardise ranges
df['s3'] = zscore(df['s3'])
df['a8'] = zscore(df['a8'])
df['a11'] = zscore(df['a11'])
df['a15'] = zscore(df['a15'])

print(df)

# create to numpy - classification data
x_columns = df.columns.drop('a2').drop('a14')
x = df[x_columns].values
y_a2 = df['a2'].values
y_a14 = df['a14'].values

# create train/test
x_train, x_test, y_a2_train, y_a2_test = train_test_split(x, y_a2,test_size=0.20, random_state=42)
# x_train, x_val, y_a2_train, y_a2_val = train_test_split(x_train, y_a2_train,test_size=0.25, random_state=42)


#####################################################
# Build Model
'''
model = Sequential()
model.add(Dense(25, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3,patience=5,verbose=3, mode='auto', restore_best_weights=True)
model.fit(x_train, y_a2_train, validation_data=(x_val, y_a2_val),callbacks=[monitor], verbose=3,epochs=1000)
'''

model = Ridge(alpha=1)
model.fit(x_train, y_a2_train)

pred = model.predict(x_test)
score = np.sqrt(metrics.mean_squared_error(pred,y_a2_test))
print(f"Final score (RMSE): {score}")

# print regression chart
t = pd.DataFrame({'pred': pred.flatten(), 'y': y_a2_test.flatten()})
t.sort_values(by=['y'], inplace=True)
plt.plot(t['y'].tolist(), label='expected')
plt.plot(t['pred'].tolist(), label='prediction')
plt.ylabel('output')
plt.legend()
plt.show()
