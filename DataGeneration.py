import os
import random
import argparse
import itertools
import wfdb
import heapq
import numpy as np
import pandas as pd
from wfdb import processing
from tqdm import tqdm
from scipy.stats import pearsonr

parser = argparse.ArgumentParser("Authentication", add_help=False)
parser.add_argument(
    "--data_path",
    default="../storage/ssd/public/guoxin/physionet.org/files/ptbdb/1.0.0/",
    type=str,
    help="dataset path",
)
parser.add_argument(
    "--prefix",
    default="patient",
    type=str,
    help="dataset prefix",
)
parser.add_argument(
    "--output_path",
    default="PTB_dataset.csv",
    type=str,
    help="output path",
)
args = parser.parse_args()


def dataGeneration(data_path, csv_path, record_path):
    # initialize dataset
    dataset = pd.DataFrame(columns=["label", "record"])
    if record_path == None:
        # a loop for each patient
        detail_path = data_path + "/"
        record_files = [
            i.split(".")[0]
            for i in os.listdir(detail_path)
            if (not i.startswith(".") and i.endswith(".hea"))
        ]
        # a loop for each record
        for record_name in tqdm(record_files, desc="Processing"):
            # load record
            signal, info = wfdb.rdsamp(detail_path + record_name)
            fs = 200
            signal = processing.resample_sig(signal[:, 0], info["fs"], fs)[0]
            # set some parameters
            window_size_half = int(fs * 0.125 / 2)
            max_bpm = 230
            # detect QRS peaks
            qrs_inds = processing.gqrs_detect(signal, fs=fs)
            search_radius = int(fs * 60 / max_bpm)
            corrected_qrs_inds = processing.correct_peaks(
                signal,
                peak_inds=qrs_inds,
                search_radius=search_radius,
                smooth_window_size=150,
            )
            average_qrs = 0
            count = 0
            for i in range(1, len(corrected_qrs_inds) - 1):
                start_ind = corrected_qrs_inds[i] - window_size_half
                end_ind = corrected_qrs_inds[i] + window_size_half + 1
                if (
                    start_ind < corrected_qrs_inds[i - 1]
                    or end_ind > corrected_qrs_inds[i + 1]
                ):
                    continue
                average_qrs = average_qrs + signal[start_ind:end_ind]
                count = count + 1
            # remove outliers
            if count < 8:
                print("\noutlier detected, discard " + record_name)
                continue
            average_qrs = average_qrs / count
            corrcoefs = []
            for i in range(1, len(corrected_qrs_inds) - 1):
                start_ind = corrected_qrs_inds[i] - window_size_half
                end_ind = corrected_qrs_inds[i] + window_size_half + 1
                if (
                    start_ind < corrected_qrs_inds[i - 1]
                    or end_ind > corrected_qrs_inds[i + 1]
                ):
                    corrcoefs.append(-100)
                    continue
                corrcoef = pearsonr(signal[start_ind:end_ind], average_qrs)[0]
                corrcoefs.append(corrcoef)
            max_corr = list(map(corrcoefs.index, heapq.nlargest(8, corrcoefs)))
            index_corr = random.sample(list(itertools.permutations(max_corr, 8)), 100)
            for index in index_corr:
                # a temp dataframe to store one record
                record_temp = pd.DataFrame()
                signal_temp = []
                for i in index:
                    start_ind = corrected_qrs_inds[i + 1] - window_size_half
                    end_ind = corrected_qrs_inds[i + 1] + window_size_half + 1
                    sig = processing.normalize_bound(signal[start_ind:end_ind], -1, 1)
                    signal_temp = np.concatenate((signal_temp, sig))
                record_temp = record_temp._append(
                    pd.DataFrame(signal_temp.reshape(-1, signal_temp.shape[0])),
                    ignore_index=True,
                    sort=False,
                )
                record_temp["label"] = record_name
                record_temp["record"] = record_name
                # add it to final dataset
                dataset = dataset._append(record_temp, ignore_index=True, sort=False)
    else:
        patient_folders = [
            i
            for i in os.listdir(data_path)
            if (not i.startswith(".") and i.startswith(record_path))
        ]
        # a loop for each patient
        for patient_name in tqdm(patient_folders, desc="Processing"):
            detail_path = data_path + patient_name + "/"
            record_files = [
                i.split(".")[0] for i in os.listdir(detail_path) if i.endswith(".hea")
            ]
            # a loop for each record
            for record_name in record_files:
                # load record
                signal, info = wfdb.rdsamp(detail_path + record_name)
                fs = 200
                signal = processing.resample_sig(signal[:, 0], info["fs"], fs)[0]
                # set some parameters
                window_size_half = int(fs * 0.125 / 2)
                max_bpm = 230
                # detect QRS peaks
                qrs_inds = processing.gqrs_detect(signal, fs=fs)
                search_radius = int(fs * 60 / max_bpm)
                corrected_qrs_inds = processing.correct_peaks(
                    signal,
                    peak_inds=qrs_inds,
                    search_radius=search_radius,
                    smooth_window_size=150,
                )
                average_qrs = 0
                count = 0
                for i in range(1, len(corrected_qrs_inds) - 1):
                    start_ind = corrected_qrs_inds[i] - window_size_half
                    end_ind = corrected_qrs_inds[i] + window_size_half + 1
                    if (
                        start_ind < corrected_qrs_inds[i - 1]
                        or end_ind > corrected_qrs_inds[i + 1]
                    ):
                        continue
                    average_qrs = average_qrs + signal[start_ind:end_ind]
                    count = count + 1
                # remove outliers
                if count < 8:
                    print(
                        "\noutlier detected, discard "
                        + record_name
                        + " of "
                        + patient_name
                    )
                    continue
                average_qrs = average_qrs / count
                corrcoefs = []
                for i in range(1, len(corrected_qrs_inds) - 1):
                    start_ind = corrected_qrs_inds[i] - window_size_half
                    end_ind = corrected_qrs_inds[i] + window_size_half + 1
                    if (
                        start_ind < corrected_qrs_inds[i - 1]
                        or end_ind > corrected_qrs_inds[i + 1]
                    ):
                        corrcoefs.append(-100)
                        continue
                    corrcoef = pearsonr(signal[start_ind:end_ind], average_qrs)[0]
                    corrcoefs.append(corrcoef)
                max_corr = list(map(corrcoefs.index, heapq.nlargest(8, corrcoefs)))
                index_corr = random.sample(
                    list(itertools.permutations(max_corr, 8)), 100
                )
                for index in index_corr:
                    # a temp dataframe to store one record
                    record_temp = pd.DataFrame()
                    signal_temp = []
                    for i in index:
                        start_ind = corrected_qrs_inds[i + 1] - window_size_half
                        end_ind = corrected_qrs_inds[i + 1] + window_size_half + 1
                        sig = processing.normalize_bound(
                            signal[start_ind:end_ind], -1, 1
                        )
                        signal_temp = np.concatenate((signal_temp, sig))
                    record_temp = record_temp._append(
                        pd.DataFrame(signal_temp.reshape(-1, signal_temp.shape[0])),
                        ignore_index=True,
                        sort=False,
                    )
                    record_temp["label"] = patient_name
                    record_temp["record"] = record_name
                    # add it to final dataset
                    dataset = dataset._append(record_temp, ignore_index=True, sort=False)
    # save for further use
    dataset.to_csv(csv_path, index=False)
    print("processing completed")


if __name__ == "__main__":
    dataGeneration(args.data_path, args.output_path, args.prefix)
