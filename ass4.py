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

# import data
file='C:\\Users\\felix\\github\\deepneuralnetworks\\crx.csv'
df = pd.read_csv(file, na_values="?")

# aim: replace missing values in a2 and a14
# x data = 's3','a8','a9','a10','a11','a12','a13','a15'
# create dummy values
# drop na values for modelling
df = df.dropna(axis = 0, how ='any', thresh = None, subset = None, inplace=False)

# drop unused columns
df = df.drop(['a1', 'a4', 'a5', 'a6', 'a7','a16'], axis=1)

# create dummy variables for dataframe
df = pd.get_dummies(df)

# 'shuffle' data (reindex)
df = df.reindex(np.random.permutation(df.index))

# split data into testing, validation and training
mask = np.random.rand(len(df))<0.8
traindf = pd.DataFrame(df[mask])
testdf = pd.DataFrame(df[~mask])

mask = np.random.rand(len(traindf))<0.8
validationdf = pd.DataFrame(traindf[~mask])
traindf = pd.DataFrame(traindf[mask])


print(f"Training df: {len(traindf)}")
print(f"test df: {len(testdf)}")


# create x table
x_columns = traindf.columns.drop(['a2', 'a14'])
train_x = traindf[x_columns] # need to get dummy columns manually
test_x = testdf[x_columns]
validation_x = validationdf[x_columns]

# create a2 table
train_y_a2 = traindf['a2']
test_y_a2 = testdf['a2']
validation_y_a2 = validationdf['a2']

# create a14 table
train_y_a14 = df['a14']


#####################################################
# Build Model
model = Sequential()
model.add(Dense(10, input_dim=train_x.shape[1], activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3,patience=5,verbose=1, mode='auto', restore_best_weights=True)
model.fit(train_x, train_y_a2, validation_data=(validation_x, validation_y_a2),callbacks=[monitor], verbose=1,epochs=1000)

pred = model.predict(test_x)
score = np.sqrt(metrics.mean_squared_error(pred,test_y_a2))
print(f"Final score (RMSE): {score}")

print(len(pred))
print(test_y_a2)
