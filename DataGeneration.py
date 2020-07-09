import os
import wfdb
from wfdb import processing
import numpy as np
import pandas as pd
from progress.bar import Bar

def dataGeneration(data_path, window_size_half, max_bpm):

    patient_folders = [i for i in os.listdir(data_path) if (not i.startswith('.') and i.startswith('patient'))]

    # initialize dataset
    dataset = pd.DataFrame(columns=['label'])

    Bar.check_tty = False
    bar = Bar('Processing', max=len(patient_folders), fill='#', suffix='%(percent)d%%')

    # a loop for each patient
    for patient_name in patient_folders:
        detail_path = data_path + patient_name + '/'
        record_files = [i.split('.')[0] for i in os.listdir(detail_path) if i.endswith('.hea')]

        # a loop for each record
        for record_name in record_files:

            # load record
            signal, info = wfdb.rdsamp(detail_path + record_name, channel_names=['i'])

            # detect QRS peaks
            qrs_inds = processing.gqrs_detect(signal[:,0], fs=info['fs'])
            search_radius = int(info['fs']*60/max_bpm)
            corrected_qrs_inds = processing.correct_peaks(signal[:,0], peak_inds=qrs_inds, search_radius=search_radius, smooth_window_size=150)

            # a temp dataframe to store one record
            record_temp = pd.DataFrame(columns=['label'])

            # select 30 pieces, discard the first peak and the last peak
            if len(corrected_qrs_inds)<32:
                print('\noutlier detected, discard ' + record_name + ' of ' + patient_name)
                continue

            i_rand = np.random.choice(range(1, len(corrected_qrs_inds)-1), 30, replace=False)
            for i in i_rand:
                start_ind = corrected_qrs_inds[i] - window_size_half
                end_ind = corrected_qrs_inds[i] + window_size_half
                if start_ind<corrected_qrs_inds[i-1] or end_ind>corrected_qrs_inds[i+1]:
                    continue

                # normalization
                sig = processing.normalize_bound(signal[start_ind: end_ind], -1, 1)

                record_temp = record_temp.append(pd.DataFrame(sig.T), ignore_index=True, sort=False)
                record_temp.iloc[:, record_temp.columns.get_loc('label')] = patient_name

            # remove outliers
            if record_temp.shape[0]<3:
                print('\noutlier detected, discard ' + record_name + ' of ' + patient_name)
                continue


            # add it to final dataset
            dataset = dataset.append(record_temp, ignore_index=True, sort=False)
        
        bar.next()    

    bar.finish()

    # save for further use
    dataset.to_csv('PTB_dataset.csv', index=False)

    print('processing completed')

if __name__ == "__main__":
    # root path
    data_path = 'ptb-diagnostic-ecg-database-1.0.0/'

    # set some parameters
    window_size_half = 454
    max_bpm = 230

    dataGeneration(data_path, window_size_half, max_bpm)