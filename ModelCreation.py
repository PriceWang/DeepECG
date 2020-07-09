import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import keras
import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, Activation, BatchNormalization, MaxPooling1D, Flatten, Conv2D
from keras.layers import Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics

# Split data into training and test partitions
def createSet(dataset):
    x_cols = [col for col in dataset.columns if col != 'label']
    X_data = dataset[x_cols].values
    X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], -1))
    Y_data = dataset['label'].values

    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, random_state=0, test_size = 0.30, train_size = 0.7)

    num_classes = len(np.unique(Y_data))

    return num_classes, X_train, X_test, Y_train, Y_test

# Convert class vectors to binary class matrices
def binaryConvertion(num_classes, Y_train, Y_test):

    Y_train_encoder = sklearn.preprocessing.LabelEncoder()
    Y_train_num = Y_train_encoder.fit_transform(Y_train)
    Y_train_wide = np_utils.to_categorical(Y_train_num, num_classes)

    Y_test_num = Y_train_encoder.fit_transform(Y_test)
    Y_test_wide = np_utils.to_categorical(Y_test_num, num_classes)

    return Y_train_wide, Y_test_num, Y_test_wide

def show(X_train, Y_train):
    pltsize = 4
    row_images = 2
    col_images = 2
    plt.figure(figsize=(col_images*pltsize, row_images*pltsize))

    for i in range(row_images * col_images):
        i_rand = random.randint(0, X_train.shape[0]-1)
        plt.subplot(row_images, col_images, i+1)
        plt.plot(X_train[i_rand])
        plt.title(str(Y_train[i_rand]))

def modelling(X_train, Y_train_wide):
    # 1-D CNN
    input_shape = (X_train.shape[1], X_train.shape[2])

    model = Sequential()

    model.add(Conv1D(16, 7, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2))

    model.add(Conv1D(32, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2))

    model.add(Conv1D(64, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2))

    model.add(Conv1D(128, 7))
    model.add(Activation('relu'))

    model.add(Conv1D(256, 7))
    model.add(Activation('relu'))

    model.add(Conv1D(256, 8))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.summary()

    # training
    batch_size = 32
    epochs = 20

    # set up the callback to save the best model based on validaion data
    best_weights_filepath = './best_weights.hdf5'
    mcp = ModelCheckpoint(best_weights_filepath, monitor="val_accuracy",
                        save_best_only=True, save_weights_only=False)

    history = model.fit(X_train, Y_train_wide,
            batch_size=batch_size,
            epochs=epochs,
            verbose = 1,
            validation_split = 0.2,
            shuffle=True,
            callbacks=[mcp])

    # reload best weights
    model.load_weights(best_weights_filepath)

    # save model
    model.save('model.h5')

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss, 'blue', label='Training Loss')
    plt.plot(val_loss, 'green', label='Validation Loss')
    plt.xticks(range(0,epochs)[0::2])
    plt.legend()
    plt.show()

    return model

def evaluation(model, X_test, Y_test, Y_test_num):

    # make a set of predictions for the test data
    pred = model.predict_classes(X_test)

    # print performance details
    print(metrics.classification_report(Y_test_num, pred))

    # Draw some examples of correct classifications
    pltsize = 4
    row_images = 2
    col_images = 2
    plt.figure(figsize=(col_images*pltsize, row_images*pltsize))
    maxtoshow = row_images * col_images

    predictions = pred.reshape(-1)
    corrects = predictions == Y_test_num

    count = 0
    while range(len(corrects)):
        if count>=maxtoshow:
            break
        i_rand = random.randint(0, X_test.shape[0]-1)
        if corrects[i_rand]:        
            plt.subplot(row_images,col_images,count+1)
            plt.plot(X_test[i_rand])
            plt.title(Y_test[i_rand])
            count = count+1

    # Draw some examples of wrong classifications
    plt.figure(figsize=(col_images*pltsize, row_images*pltsize))

    count = 0
    while range(len(corrects)):
        if count>=maxtoshow:
            break
        i_rand = random.randint(0, X_test.shape[0]-1)
        if ~corrects[i_rand]:        
            plt.subplot(row_images,col_images,count+1)
            plt.plot(X_test[i_rand])
            plt.title(Y_test[i_rand])
            count = count+1

if __name__ == "__main__":
    dataset = pd.read_csv('PTB_dataset.csv')
    num_classes, X_train, X_test, Y_train, Y_test = createSet(dataset)
    Y_train_wide, Y_test_num, Y_test_wide = binaryConvertion(num_classes, Y_train, Y_test)
    show(X_train, Y_train)
    model = modelling(X_train, Y_train_wide)
    evaluation(model, X_test, Y_test, Y_test_num)
