import argparse
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Model
from keras.layers import (
    Input,
    Softmax,
    Conv1D,
    Dense,
    Dropout,
    ReLU,
    MaxPooling1D,
    Flatten,
)
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn import metrics

parser = argparse.ArgumentParser("Authentication", add_help=False)
parser.add_argument(
    "--save_path",
    default="model.h5",
    type=str,
    help="save path",
)
parser.add_argument(
    "--data_path",
    default="PTB_dataset.csv",
    type=str,
    help="dataset path",
)
args = parser.parse_args()


# Split data into training and test partitions
def createSet(dataset):
    x_cols = [col for col in dataset.columns if (col != "label" and col != "record")]
    X_data = dataset[x_cols].values
    X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], -1))
    Y_data = dataset["label"].values
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_data, Y_data, random_state=0, test_size=0.3, train_size=0.7
    )
    num_classes = len(np.unique(Y_data))
    return num_classes, X_train, X_test, Y_train, Y_test


# Convert class vectors to binary class matrices
def binaryConvertion(num_classes, Y_train, Y_test):
    Y_train_encoder = sklearn.preprocessing.LabelEncoder()
    Y_train_num = Y_train_encoder.fit_transform(Y_train)
    Y_train_wide = to_categorical(Y_train_num, num_classes)
    Y_test_num = Y_train_encoder.fit_transform(Y_test)
    Y_test_wide = to_categorical(Y_test_num, num_classes)
    return Y_train_wide, Y_test_num, Y_test_wide


def modelling(X_train, Y_train_wide, save_path):
    # 1-D CNN
    input_shape = (X_train.shape[1], X_train.shape[2])
    inputs = Input(shape=input_shape)
    x = Conv1D(16, 7)(inputs)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=3, strides=2)(x)
    x = Conv1D(32, 5)(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=3, strides=2)(x)
    x = Conv1D(64, 5)(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=3, strides=2)(x)
    x = Conv1D(128, 7)(x)
    x = ReLU()(x)
    x = Conv1D(256, 7)(x)
    x = ReLU()(x)
    x = Conv1D(256, 8)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(num_classes)(x)
    predictions = Softmax()(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    # training
    batch_size = 16
    epochs = 20
    # set up the callback to save the best model based on validaion data
    best_weights_filepath = "./best_weights.hdf5"
    mcp = ModelCheckpoint(
        best_weights_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
    )
    history = model.fit(
        X_train,
        Y_train_wide,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.2,
        shuffle=True,
        callbacks=[mcp],
    )
    # reload best weights
    model.load_weights(best_weights_filepath)
    # save model
    model.save(save_path)
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(loss, "blue", label="Training Loss")
    plt.plot(val_loss, "green", label="Validation Loss")
    plt.xticks(range(0, epochs)[0::2])
    plt.legend()
    plt.show()
    return model


def evaluation(model, X_test, Y_test_num):
    # make a set of predictions for the test data
    pred = np.argmax(model.predict(X_test, verbose=0), axis=-1)
    # print performance details
    print(metrics.classification_report(Y_test_num, pred))


if __name__ == "__main__":
    dataset = pd.read_csv(args.data_path)
    num_classes, X_train, X_test, Y_train, Y_test = createSet(dataset)
    Y_train_wide, Y_test_num, Y_test_wide = binaryConvertion(
        num_classes, Y_train, Y_test
    )
    model = modelling(X_train, Y_train_wide, args.save_path)
    evaluation(model, X_test, Y_test_num)
