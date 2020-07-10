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

def dataProcessing(data_path, dataset_path):
    dataset = pd.read_csv('PTB_dataset.csv')

    patients = pd.unique(dataset['label'])

    users = np.random.choice(patients, int(np.floor(len(patients) / 2)), replace=False)

    test_user = dataset.loc[dataset['label'].isin(users)]
    test_intruder = dataset.loc[~dataset['label'].isin(users)]

    user_database = pd.DataFrame()
    for group in test_user.groupby('label'):
        records = list(group[1].groupby('record'))
        irand = random.randint(0, len(records))
        for i in range(irand):
            record = records[i][1]
            user_database = user_database.append(record, ignore_index=True, sort=False)

    return user_database, test_user, test_intruder

def databaseGeneration(model, user_database):
    col = [column for column in user_database.columns if column not in ['label', 'record']]
    
    # template, use the outputs from the last layer
    template = model.predict(user_database[col].values)[-1]

    return template

def authentication(model, database, login):
    login_data = model.predict(login)[-1]

    threshold = 15
    score = 0

    for login_part in login_data:
        for database_part in database:
            if np.linalg.norm(login_part - database_part) < threshold:
                score = score + 1
                break

    if score > len(login_data) * 0.7:
        return True
    else:
        return False


if __name__ == "__main__":
    model_path = 'model.h5'
    data_path = 'ptb-diagnostic-ecg-database-1.0.0/'
    dataset_path = 'PTB_dataset.csv'

    model = rebuildModel(model_path)
    user_database, test_user, test_intruder = dataProcessing(data_path, dataset_path)

    print(user_database.shape)
    print(test_user.shape)
    print(test_intruder.shape)


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