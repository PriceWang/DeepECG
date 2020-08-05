import os
import random
import numpy as np
import pandas as pd
from progress.bar import Bar

import keras

def rebuildModel(model_path):
    model = keras.models.load_model(model_path)

    layer_outputs = []
    for layer in model.layers:
        if layer.name.startswith('dropout'):
            break
        layer_outputs.append(layer.output)

    model_template = keras.Model(inputs=model.input, outputs=layer_outputs)
    model_template.summary()

    return model_template

def dataProcessing(dataset_path):
    dataset = pd.read_csv(dataset_path)

    patients = pd.unique(dataset['label'])

    users = np.random.choice(patients, int(np.floor(len(patients) / 2)), replace=False)

    test_user = dataset.loc[dataset['label'].isin(users)]

    user_database = test_user.groupby('record').head(1)

    test_user = test_user.sample(n=100, replace=False)
    test_intruder = (dataset.loc[~dataset['label'].isin(users)]).sample(n=100, replace=False)

    return user_database, test_user, test_intruder

def databaseGeneration(model, user_database):
    col = [column for column in user_database.columns if column not in ['label', 'record']]
    
    # template, use the outputs from the last layer
    template = model.predict(user_database[col].values)[-1]

    return template

def authentication(model, database, login):
    login_data = model.predict(login)[-1]

    threshold = 10
    score = 0

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

    print('Accuracy : {:.2%}'.format(score / attempt_number))

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

    print('Accuracy : {:.2%}'.format(score / attempt_number))