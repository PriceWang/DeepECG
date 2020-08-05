import time
import numpy as np
import pandas as pd
from progress.bar import Bar
from binary_layer.binary_ops import binarize

import keras

BNN = True

def rebuildModel(model_path):

    model = keras.models.load_model(model_path)

    layer_outputs = []
    for layer in model.layers:
        if layer.name.startswith('dropout'):
            break

        if BNN and layer.name.startswith('conv1d'):
            weights = layer.get_weights()
            for i in range(len(weights)):
                weights[i] = binarize(weights[i])
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

    test_user = test_user.sample(n=500, replace=False)
    test_intruder = (dataset.loc[~dataset['label'].isin(users)]).sample(n=500, replace=False)

    return user_database, test_user, test_intruder

def databaseGeneration(model, user_database):
    col = [column for column in user_database.columns if column not in ['label', 'record']]
    
    # template, use the outputs from the last layer
    template = model.predict(user_database[col].values)[-1]

    return template

def authentication(model, database, login):
    login_data = model.predict(login)[-1]

    if BNN:
        threshold = 200000
    else:
        threshold = 13.5

    for login_part in login_data:
        for database_part in database:
            if np.linalg.norm(login_part - database_part) < threshold:
                return True

    return False


if __name__ == "__main__":

    model_path = 'model.h5'
    dataset_path = 'PTB_dataset.csv'

    model = rebuildModel(model_path)
    user_database, test_user, test_intruder = dataProcessing(dataset_path)

    database = databaseGeneration(model, user_database)

    start_time = time.time()
    attempt_number = len(test_user['record'].unique())
    score = 0

    Bar.check_tty = False
    bar = Bar('Verifying', max=attempt_number, fill='#', suffix='%(percent)d%%')
    for user in test_user.groupby('record'):
        login = user[1].drop(columns=['label', 'record']).values        
        if authentication(model, database, login):
            score = score + 1
        bar.next()
    bar.finish()
    end_time = time.time()

    print('Accuracy : {:.2%}, Elapsed Time : {:.2f}s'.format(score / attempt_number, end_time - start_time))

    start_time = time.time()
    attempt_number = len(test_intruder['record'].unique())
    score = 0

    Bar.check_tty = False
    bar = Bar('Verifying', max=attempt_number, fill='#', suffix='%(percent)d%%')
    for user in test_intruder.groupby('record'):
        login = user[1].drop(columns=['label', 'record']).values        
        if not authentication(model, database, login):
            score = score + 1
        bar.next()
    bar.finish()
    end_time = time.time()

    print('Accuracy : {:.2%}, Elapsed Time : {:.2f}s'.format(score / attempt_number, end_time - start_time))