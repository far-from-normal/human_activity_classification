#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from keras.utils.io_utils import HDF5Matrix
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras import backend as K
K.set_image_dim_ordering('tf')

#%%
def PlotConfusionMatrix(y_test,y_pred,fig_name):

    cm = confusion_matrix(y_test, y_pred)
    cm_norm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm_norm)

    fig = plt.figure(figsize=(9, 3))
    ax = fig.add_subplot(1,2,1)
    sns.heatmap(cm, cmap='coolwarm_r',linewidths=0.5,annot=True,ax=ax)
    plt.title('Confusion Matrix')
    plt.ylabel('True churn type')
    plt.xlabel('Predicted churn type')

    ax = fig.add_subplot(1,2,2)
    sns.heatmap(cm_norm ,cmap='coolwarm_r',linewidths=0.5,annot=True,ax=ax)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True churn type')
    plt.xlabel('Predicted churn type')
    plt.show()
    
    plot_path = Path('plots')
    if not plot_path.exists():
        plot_path.mkdir()
    fig.savefig(str(plot_path.joinpath(fig_name + 'confusion_matrix.png')), bbox_inches='tight')
    
    return None

#%%

def data_to_hdf5(hdf5_path): 
    parent_data_dir = 'UCI HAR Dataset'
    test_dir = 'test'
    train_dir = 'train'
    sub_dir = 'Inertial Signals'
    
    data_link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
    
    train_names = ['body_acc_x_train.txt',
                   'body_acc_y_train.txt',
                   'body_acc_z_train.txt',
                   'body_gyro_x_train.txt',
                   'body_gyro_y_train.txt',
                   'body_gyro_z_train.txt',
                   'total_acc_x_train.txt',
                   'total_acc_y_train.txt',
                   'total_acc_z_train.txt']
    
    test_names = ['body_acc_x_test.txt',
                  'body_acc_y_test.txt',
                  'body_acc_z_test.txt',
                  'body_gyro_x_test.txt',
                  'body_gyro_y_test.txt',
                  'body_gyro_z_test.txt',
                  'total_acc_x_test.txt',
                  'total_acc_y_test.txt',
                  'total_acc_z_test.txt']
    
    ts_len = 128
    n_ts = len(train_names)
    
    y_train_path = os.path.join(parent_data_dir, train_dir, 'y_train.txt')
    y_test_path  = os.path.join(parent_data_dir, test_dir, 'y_test.txt')
    
    X_train_dir = os.path.join(parent_data_dir, train_dir, sub_dir)
    X_test_dir  = os.path.join(parent_data_dir, test_dir, sub_dir)
    
    # load labels
    y_train = np.loadtxt(y_train_path) - 1
    y_test  = np.loadtxt(y_test_path) - 1

    #get number of samples
    n_train = y_train.shape[0]
    n_test  = y_test.shape[0]
    
    # set shape of features
    train_shape = (n_train, ts_len, n_ts)
    test_shape = (n_test, ts_len, n_ts)
    
    # create hdf5 file
    hdf5_file = h5py.File(hdf5_path, mode='w')
    
    # create hdf5 data
    hdf5_file.create_dataset('y_train', (n_train,), np.int8)
    hdf5_file.create_dataset('y_test', (n_test,), np.int8)
    hdf5_file.create_dataset('X_train', train_shape, dtype=np.float64)
    hdf5_file.create_dataset('X_test', test_shape, dtype=np.float64)
    
    # load hdf5 labels
    hdf5_file['y_train'][...] = y_train
    hdf5_file['y_test'][...] = y_test
    
    # load hdf5 features
    meas = 0
    for (file_train, file_test) in zip(train_names, test_names):
        print('Loading ...', file_train, file_test)
        X_train_file = os.path.join(X_train_dir, file_train)
        X_test_file = os.path.join(X_test_dir, file_test)
        hdf5_file['X_train'][..., meas] = np.loadtxt(X_train_file)
        hdf5_file['X_test'][..., meas] = np.loadtxt(X_test_file)
        meas += 1
    print(hdf5_file['X_train'].shape)
    print(hdf5_file['X_test'].shape)
    
    hdf5_file.close()
    
    return None

#%%

def cnn_model(input_shape, num_class, conv_act):
    padding = 'valid'
    inputs = Input(shape=input_shape)
    x = Conv1D(filters= 1 * filt, 
               kernel_size=2, 
               strides=1,     
               kernel_initializer='uniform',      
               activation=conv_act,
               padding=padding)(inputs)
    x = MaxPooling1D(pool_size=2, 
                     strides=None, 
                     padding=padding)(x)
    x = Dropout(0.25)(x)
    x = Conv1D(filters=2 * filt, 
               kernel_size=2, 
               strides=1,     
               kernel_initializer='uniform',      
               activation=conv_act,
               padding=padding)(x)
    x = MaxPooling1D(pool_size=2, 
                     strides=None, 
                     padding=padding)(x)
    x = Dropout(0.25)(x)
    x = Conv1D(filters=4 * filt, 
               kernel_size=2, 
               strides=1,     
               kernel_initializer='uniform',      
               activation=conv_act,
               padding=padding)(x)
    x = MaxPooling1D(pool_size=2, 
                     strides=None, 
                     padding=padding)(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(units=1024, 
              activation='relu')(x)
    x = Dropout(0.25)(x)
    outputs = Dense(units=num_class, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    model.summary()
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    
    return model


#%%
hdf5_path = 'movement_data.h5'
weights_file = 'weights.h5'
batch_size = 128
n_epoch = 100
filt = 16
conv_act = 'relu'

# setting data
data_to_hdf5(hdf5_path)

hdf5_file = h5py.File(hdf5_path, mode='r')
X_train = HDF5Matrix(hdf5_path, 'X_train')
y_train = HDF5Matrix(hdf5_path, 'y_train')
y_train = to_categorical(y_train)

data_num = hdf5_file['X_train'].shape[0]
input_shape = hdf5_file['X_train'].shape[1:]
num_class = y_train.shape[-1]

# setting callbacks
checkpoint = ModelCheckpoint(weights_file, 
                             monitor='loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min')
callbacks_list = [checkpoint]

# initiate model
model = cnn_model(input_shape, num_class, conv_act)

# fit data to model
# Note: must use shuffle='batch' or False with HDF5Matrix
model.fit(X_train, y_train, 
          batch_size=batch_size, 
          shuffle='batch', 
          epochs=n_epoch, 
          verbose=2,
          callbacks=callbacks_list)

#%% get score from predicting test data

X_test = HDF5Matrix(hdf5_path, 'X_test')
y_test = HDF5Matrix(hdf5_path, 'y_test')

y_test = to_categorical(y_test)

model.load_weights(weights_file)
scores = model.evaluate(X_test, y_test, 
                        batch_size=batch_size, 
                        verbose=2)
print(scores)

y_pred = model.predict(X_test,
                       batch_size=batch_size)

PlotConfusionMatrix(y_test.argmax(axis=-1), y_pred.argmax(axis=-1), 'test_')


