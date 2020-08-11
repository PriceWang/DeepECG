import numpy as np
import pandas as pd
from progress.bar import Bar
import matplotlib.pyplot as plt
from matplotlib import ticker

import keras
import keras.backend as K


def weightTransform(W, mode = 1, n = 1):

    if mode == 1:
        W_new = W
    elif mode == 2:
        W_new = np.sign(W)
    else:
        W_new = np.round(W * (np.power(2, n))) / (np.power(2, n))
    return W_new

def rebuildModel(model_path):

    model = keras.models.load_model(model_path)

    layer_outputs = []
    for layer in model.layers:
        if layer.name.startswith('dropout'):
            break

        if layer.name.startswith('conv1d'):
            weights = layer.get_weights()
            for i in range(len(weights)):
                weights[i] = weightTransform(weights[i])
            layer.set_weights(weights)

        output = layer.output
        layer_outputs.append(output)

    model_template = keras.Model(inputs=model.input, outputs=layer_outputs)

    return model_template

def rebuildModelBNN(model_path):

    model = keras.models.load_model(model_path)

    layer_outputs = []
    for layer in model.layers:
        if layer.name.startswith('dropout'):
            break

        if layer.name.startswith('conv1d'):
            weights = layer.get_weights()
            for i in range(len(weights)):
                weights[i] = weightTransform(weights[i], 2)
            layer.set_weights(weights)

        output = layer.output
        layer_outputs.append(output)

    model_template = keras.Model(inputs=model.input, outputs=layer_outputs)

    return model_template

def rebuildModelENN(model_path, n):

    model = keras.models.load_model(model_path)

    layer_outputs = []
    for layer in model.layers:
        if layer.name.startswith('dropout'):
            break

        if layer.name.startswith('conv1d'):
            weights = layer.get_weights()
            for i in range(len(weights)):
                weights[i] = weightTransform(weights[i], 3, n)
            layer.set_weights(weights)

        output = layer.output
        layer_outputs.append(output)

    model_template = keras.Model(inputs=model.input, outputs=layer_outputs)

    return model_template

def dataProcessing(dataset_path):
    dataset = pd.read_csv(dataset_path)

    patients = pd.unique(dataset['label'])

    users = np.random.choice(patients, int(np.floor(len(patients) / 2)), replace=False)

    test_user = dataset.loc[dataset['label'].isin(users)]

    user_database = test_user.groupby('record').head(1)

    test_user = test_user.sample(n=1000, replace=False)
    test_intruder = (dataset.loc[~dataset['label'].isin(users)]).sample(n=1000, replace=False)

    return user_database, test_user, test_intruder

def databaseGeneration(model, user_database):
    col = [column for column in user_database.columns if column not in ['label', 'record']]
    
    # template, use the outputs from the last layer
    template = model.predict(user_database[col].values)[-1]

    return template

def authentication(model, database, login, threshold):
    login_data = model.predict(login)[-1]

    for login_part in login_data:
        for database_part in database:
            if np.linalg.norm(login_part - database_part) < threshold:
                return True

    return False

def login(model, database, test_user, test_intruder, threshold):
    user_number = len(test_user['record'].unique())
    intruder_number = len(test_intruder['record'].unique())

    test_number = user_number + intruder_number
    score = 0

    Bar.check_tty = False
    bar = Bar('Verifying Users', max=user_number, fill='#', suffix='%(percent)d%%')
    for user in test_user.groupby('record'):
        login = user[1].drop(columns=['label', 'record']).values        
        if authentication(model, database, login, threshold):
            score = score + 1
        bar.next()
    bar.finish()

    Bar.check_tty = False
    bar = Bar('Verifying Intruders', max=intruder_number, fill='#', suffix='%(percent)d%%')
    for user in test_intruder.groupby('record'):
        login = user[1].drop(columns=['label', 'record']).values        
        if not authentication(model, database, login, threshold):
            score = score + 1
        bar.next()
    bar.finish()

    accuracy = score / test_number

    print('Accuracy : {:.2%}'.format(accuracy))

    return accuracy


if __name__ == "__main__":

    model_path = 'model.h5'
    dataset_path = 'PTB_dataset.csv'

    user_database, test_user, test_intruder = dataProcessing(dataset_path)

    accuracy = []

    model = rebuildModel(model_path)

    database = databaseGeneration(model, user_database)

    normal_acc = login(model, database, test_user, test_intruder, 10.5)

    accuracy.append(normal_acc)

    model = rebuildModelBNN(model_path)

    database = databaseGeneration(model, user_database)

    bnn_acc = login(model, database, test_user, test_intruder, 135000)

    accuracy.append(bnn_acc)

    model = rebuildModelENN(model_path, 2)

    database = databaseGeneration(model, user_database)

    enn_acc_1 = login(model, database, test_user, test_intruder, 3.75)

    accuracy.append(enn_acc_1)

    model = rebuildModelENN(model_path, 3)

    database = databaseGeneration(model, user_database)

    enn_acc_2 = login(model, database, test_user, test_intruder, 6)

    accuracy.append(enn_acc_2)

    model = rebuildModelENN(model_path, 4)

    database = databaseGeneration(model, user_database)

    enn_acc_3 = login(model, database, test_user, test_intruder, 9)

    accuracy.append(enn_acc_3)

    net = ['original', 'bnn', 'enn(n=1)', 'enn(n=2)', 'enn(n=3)']
    fig, ax = plt.subplots()
    plt.plot(net, accuracy, color='b')
    plt.scatter(net, accuracy, color='r', marker='v')
    plt.ylabel('Accuracy')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    for a, b in zip(net, accuracy):
        plt.text(a, b+0.001, '{:.2%}'.format(b), ha='center', va= 'bottom',fontsize=9)
    plt.show()

