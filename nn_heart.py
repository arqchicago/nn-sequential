# -*- coding: utf-8 -*-
"""
@author: Ahmad Qadri
Sequential Neural Network on Heart Disease dataset

"""
import pandas as pd
import numpy as np
import sklearn.model_selection as skms
from time import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.python.keras.callbacks import TensorBoard

random_seed = 295471
target_var = 'target'
test_size = 0.10
train_size = 1-test_size

#----  reading data
heart_df = pd.read_csv('data\\heart2.csv')
rows, cols = heart_df.shape
target0_rows = heart_df[heart_df[target_var]==0].shape[0]
target1_rows = heart_df[heart_df[target_var]==1].shape[0]
print(f'> data rows = {rows}  data cols = {cols}')
print(f'> {target_var}==0 ({target0_rows})  {target_var}==1 ({target1_rows})')


#----  splitting into training & testing sets
y = heart_df.target
X = heart_df.drop(target_var, axis=1)
features = X.columns.tolist()
X_train, X_test, y_train, y_test = skms.train_test_split(X, y, test_size=test_size, random_state=random_seed)
X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
X_train_rows, y_train_rows = X_train.shape[0], y_train.shape[0]
X_test_rows, y_test_rows = X_test.shape[0], y_test.shape[0]
    
print(f'> features = {len(features)}')
print(f'> training set = {X_train_rows} ({round(X_train_rows*1.0/rows,3)})')
print(f'> testing set = {X_test_rows} ({round(X_test_rows*1.0/rows,3)}) \n')

#----  creating the model
model = keras.Sequential()
model.add(keras.Input(shape=(13,)))
model.add(Dense(26, activation='sigmoid', name='layer1'))
model.add(Dense(26, activation='sigmoid', name='layer2'))
model.add(Dense(1, activation='sigmoid', name='output_layer'))

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

print(model.summary())
print(f'input shape= {model.input_shape}')
print(f'output shape= {model.output_shape}')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=500, validation_split=0.15, verbose=1, callbacks=[tensorboard])

test_set_loss, test_set_accuracy = model.evaluate(X_test, y_test)
print(f'test set loss = {round(test_set_loss, 4)}  test set accuracy = {round(test_set_accuracy, 4)}')


