import os

import wfdb
import numpy as np
from wfdb import processing
import json, csv, random

NON_USEFUL_ANNOTATIONS = ['~', '|', '+', 'Q']
ABNORMAL_BEATS = ['F', 'V', 'S']
FILENAMES = []
DATA_BASE_PATH = 'D:\\Desarrollo\\databases\\MIT-BIH\\'
SUMA_ACUMULADA = []
R_PEAKS = []

CURRENT_DIRECTORY = os.getcwd()


def find_r_peak_index(sample_close_to_peak, lower_i, upper_i):
    peaks = R_PEAKS[lower_i:upper_i]
    temp_lower_i, temp_upper_i = 0, len(peaks) - 1

    while temp_upper_i - temp_lower_i > 2:
        possible_peak = int((temp_upper_i - temp_lower_i) / 2)
        if peaks[temp_lower_i + possible_peak] < sample_close_to_peak:
            temp_lower_i = temp_lower_i + possible_peak

        elif peaks[temp_lower_i + possible_peak] > sample_close_to_peak:
            temp_upper_i = temp_lower_i + possible_peak

        elif peaks[possible_peak] == sample_close_to_peak:
            objective_r_peak = possible_peak
            return objective_r_peak

    if peaks[temp_lower_i] > sample_close_to_peak:
        return lower_i + temp_lower_i
    else:
        return lower_i + temp_upper_i


def temporal_features(current_r_peak_index):  # PROBAR
    pre_RR = R_PEAKS[current_r_peak_index] - R_PEAKS[current_r_peak_index - 1]
    post_RR = R_PEAKS[current_r_peak_index + 1] - R_PEAKS[current_r_peak_index]
    # local_RR_average = int(
    #     (SUMA_ACUMULADA[current_r_peak_index + 4] - SUMA_ACUMULADA[current_r_peak_index - 5 - 1]) / 10)
    local_RR_average = int(
        (SUMA_ACUMULADA[current_r_peak_index] - SUMA_ACUMULADA[current_r_peak_index - 10]) / 10)

    return pre_RR, post_RR, local_RR_average


def fill_RR_suma_acumulada():
    global SUMA_ACUMULADA

    i = 1
    end = len(R_PEAKS) - 1
    while i < end:
        try:
            SUMA_ACUMULADA.append(SUMA_ACUMULADA[-1] + R_PEAKS[i] - R_PEAKS[i - 1])
        except IndexError as identifier:
            SUMA_ACUMULADA.append(R_PEAKS[i] - R_PEAKS[i - 1])
        i += 1


def create_set_file(filename, set_folder_name):
    training_item = []
    filenames_used = []
    with open(f"{CURRENT_DIRECTORY}\\json_info\\{filename}.json") as patient_info_json_file:
        # wfdb_record = wfdb.rdrecord(f'{DATA_BASE_PATH}{filename}', channels=[0])

        patient_info_json = json.load(patient_info_json_file)

        indexes = []

        # normal beats
        while len(indexes) < 90: # total number of abnormal beats / number of files in dataset
            normal_r_peaks_info = patient_info_json["normal_beats"]
            rand_index = random.randint(0, len(normal_r_peaks_info) - 1)
            if rand_index not in indexes:
                indexes.append(rand_index)
                r_peak_info = normal_r_peaks_info[rand_index]

                beat_info = {"id": f"{filename}",
                             "pre_RR": r_peak_info[1],
                             "post_RR": r_peak_info[2],
                             "local_RR_average": r_peak_info[3],
                             "global_RR_average": r_peak_info[4],
                             "samples": r_peak_info[5],
                             "class": "N"
                             }

                training_item.append(beat_info)

        # abnormal beats
        abnormal_r_peaks_info = patient_info_json["abnormal_beats"]
        for r_peak_info in abnormal_r_peaks_info:
            beat_info = {"id": f"{filename}",
                         "pre_RR": r_peak_info[1],
                         "post_RR": r_peak_info[2],
                         "local_RR_average": r_peak_info[3],
                         "global_RR_average": r_peak_info[4],
                         "samples": r_peak_info[5],
                         "class": "A"
                         }
            training_item.append(beat_info)

    with open(f"{CURRENT_DIRECTORY}\\sets\\{set_folder_name}\\{filename}.json", "w+") as training_file:
        json.dump(training_item, training_file)

    filenames_used.append(patient_info_json_file)


# loading wfdb record #
classified_csv_beats_file = open(f'data.csv', newline='', mode='w+')
writer = csv.writer(classified_csv_beats_file, dialect='excel')

for filename in os.listdir(DATA_BASE_PATH):

    if (filename.endswith('.hea') or filename.endswith('.atr') or filename.endswith('.xws')):
        continue
    else:

        filename = filename.split(sep='.')
        # filename = ('108','lol')
        wfdb_record = wfdb.rdrecord(f'{DATA_BASE_PATH}{filename[0]}', channels=[0])
        wfdb_annotation = wfdb.rdann(f'{DATA_BASE_PATH}{filename[0]}', 'atr')

        record_p_signal = wfdb_record.p_signal
        signal_frecuency = wfdb_record.fs

        record_p_signal = [sample[0] for sample in record_p_signal]  # downgrading wfdb record to a simple float array

        # filtering signal #
        signal = np.array(record_p_signal)

        ### R_PEAKS

        qrs_detector = wfdb.processing.XQRS(signal, signal_frecuency)
        qrs_detector.detect()
        R_PEAKS = qrs_detector.qrs_inds
        filtered_signal = qrs_detector.sig_f

        ten_first_seconds_r_peak = find_r_peak_index(3600, 5, 20)

        fill_RR_suma_acumulada()
        GLOBAL_RR_AVERAGE = int(SUMA_ACUMULADA[-1] / len(SUMA_ACUMULADA))

        i = 7
        end = len(R_PEAKS) - 1

        beats = []

        with open(f"{CURRENT_DIRECTORY}\\samples_txt\\R_PEAKS vs samples {filename[0]}.txt", 'w+') as f:
            j, k = ten_first_seconds_r_peak, ten_first_seconds_r_peak
            peak_amount = len(wfdb_annotation.sample) if len(wfdb_annotation.sample) < len(R_PEAKS) else len(R_PEAKS)

            classified_beats = {
                'id': str(wfdb_record.record_name),
                'normal_beats': [],
                'abnormal_beats': []

            }

            while j < peak_amount - 2 and k < peak_amount:

                classified_beats_file = open(f'{CURRENT_DIRECTORY}\\json_info\\{filename[0]}.json', 'w+')

                if wfdb_annotation.symbol[k] in NON_USEFUL_ANNOTATIONS:
                    k += 1
                    continue

                if abs(R_PEAKS[j] - wfdb_annotation.sample[k]) > 11:
                    m = k
                    while R_PEAKS[j] - wfdb_annotation.sample[m] > 0 and abs(
                            R_PEAKS[j] - wfdb_annotation.sample[m]) > 11:
                        m += 1
                    if abs(R_PEAKS[j] - wfdb_annotation.sample[m]) > 11:
                        j += 1
                        # k += 1
                        continue
                    else:
                        k = m

                # print(str(R_PEAKS[j]) + " " + str(wfdb_annotation.sample[j]),file=f)
                neighbor_samples = filtered_signal[R_PEAKS[j] - 24:R_PEAKS[j] + 25]
                neighbor_samples = [sample for sample in neighbor_samples]

                pre_RR, post_RR, local_RR_average = temporal_features(j)

                if wfdb_annotation.symbol[k] in ABNORMAL_BEATS:
                    classified_beats['abnormal_beats'].append([str(R_PEAKS[j]),
                                                               str(pre_RR),
                                                               str(post_RR),
                                                               str(local_RR_average),
                                                               str(GLOBAL_RR_AVERAGE),
                                                               neighbor_samples])

                else:
                    classified_beats['normal_beats'].append([str(R_PEAKS[j]),
                                                             str(pre_RR),
                                                             str(post_RR),
                                                             str(local_RR_average),
                                                             str(GLOBAL_RR_AVERAGE),
                                                             neighbor_samples])

                print(str(abs(R_PEAKS[j] - wfdb_annotation.sample[k])) + " j=" + str(j) +
                      " " + str(R_PEAKS[j]) + " " + str(wfdb_annotation.sample[k]) + " " + wfdb_annotation.symbol[k],
                      file=f)
                j += 1
                k += 1

        json.dump(classified_beats, classified_beats_file)
        classified_beats_file.close()

        writer.writerow([classified_beats['id']
                            ,
                         2000 if len(classified_beats['normal_beats']) > 2000 else len(classified_beats['normal_beats'])
                            , len(classified_beats['abnormal_beats'])])

        print(f'done, {filename[0]}')

classified_csv_beats_file.close()

for filename in os.listdir(f"{CURRENT_DIRECTORY}\\json_info\\"):
    filename = filename.split('.')
    filename = filename[0]  # getting rid of extension
    FILENAMES.append(filename)

print('preprocessing complete')

# selecting set files #

cantidad_id_con_anormales_despreciables = 0
cantidad_items = 0
training_set_files = []
testing_set_files = []
total_files = len(FILENAMES)

while cantidad_items < total_files / 2:
    filename = FILENAMES[random.randint(0, len(FILENAMES)) - 1]
    # print(filename)

    with open(f"{CURRENT_DIRECTORY}\\json_info\\{filename}.json") as file:

        filename_json_info = json.load(file)
        # print(len(filename_json_info['abnormal_beats']))

        if len(filename_json_info['abnormal_beats']) < 5:
            if cantidad_id_con_anormales_despreciables < 12:
                cantidad_id_con_anormales_despreciables += 1
                training_set_files.append(filename)
                cantidad_items += 1
                FILENAMES.remove(filename)
                # print(FILENAMES)
        else:
            training_set_files.append(filename)
            cantidad_items += 1
            FILENAMES.remove(filename)
            # print(FILENAMES)

testing_set_files = FILENAMES

# print("training:" + f"{training_set_files}\n")

training_abnormal_beat_N = 0
training_normal_beat_N = 0
for filename in training_set_files:
    with open(f"{CURRENT_DIRECTORY}\\json_info\\{filename}.json") as file:
        json_info = json.load(file)
        training_abnormal_beat_N += len(json_info['abnormal_beats'])
        training_normal_beat_N += len(json_info['normal_beats'])

print(f"Training set abnormal beat number: {training_abnormal_beat_N}")
print(f"Training set normal beat number: {training_normal_beat_N}")

testing_abnormal_beat_N = 0
testing_normal_beat_N = 0
for filename in testing_set_files:
    with open(f"{CURRENT_DIRECTORY}\\json_info\\{filename}.json") as file:
        json_info = json.load(file)
        testing_abnormal_beat_N += len(json_info['abnormal_beats'])
        testing_normal_beat_N += len(json_info['normal_beats'])

print(f"Testing set abnormal beat number: {testing_abnormal_beat_N}")
print(f"Testing set normal beat number: {testing_normal_beat_N}")

# print("testing:" + f"{testing_set_files}\n")

# creating sets #

print("Creating sets")

for filename in training_set_files:
    create_set_file(filename, "training_set")

for filename in testing_set_files:
    create_set_file(filename, "testing_set")

print("Done creating sets")

### plotting ###
"""i = R_PEAKS[89]
times = np.arange(i+3600, dtype = 'float') #/ signal_frecuency
#plt.xlabel('Time [s]')
plt.xlabel('Sample')
plt.ylabel('mV')

while i < R_PEAKS[89]+ 3600:

    if i in R_PEAKS:
        plt.scatter(times[i],filtered_signal[i],c='red')
    elif i in wfdb_annotation.sample:
        plt.scatter(times[i],filtered_signal[i],c='blue')    
    else:
        plt.scatter(times[i],filtered_signal[i],c='black')
    i += 1
    
plt.show()
"""
# plt.plot(times, filtered_signal[:3600])
