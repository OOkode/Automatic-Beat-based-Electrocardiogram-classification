import os

import wfdb
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter
from scipy.signal import firwin, convolve
from wfdb import processing
from ecgdetectors import Detectors
import json, csv, random

NON_USEFUL_ANNOTATIONS = ['~', '|', '+', 'Q']
ABNORMAL_BEATS = ['F', 'V', 'S']
FILENAMES = []
CURRENT_DIRECTORY = os.getcwd()


class Beat:
    def __init__(self, qrs_samples, pre_RR, post_RR, local_RR_average, global_RR_average):
        self.qrs_samples = qrs_samples
        self.pre_RR = pre_RR
        self.post_RR = post_RR
        self.local_RR_average = local_RR_average
        self.global_RR_average = global_RR_average


def find_r_peak_index(r_peaks, sample_close_to_peak, lower_i, upper_i):
    peaks = r_peaks[lower_i:upper_i]
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


def temporal_features(r_peaks, current_r_peak_index, suma_acumulada):
    pre_RR = r_peaks[current_r_peak_index] - r_peaks[current_r_peak_index - 1]
    post_RR = r_peaks[current_r_peak_index + 1] - r_peaks[current_r_peak_index]
    local_RR_average = int((suma_acumulada[i + 4] - suma_acumulada[i - 5 - 1]) / 10)

    return pre_RR, post_RR, local_RR_average


def fill_RR_suma_acumulada(r_peaks, suma_acumulada):
    i = 1
    end = len(r_peaks) - 1
    while i < end:
        try:
            suma_acumulada.append(suma_acumulada[-1] + r_peaks[i] - r_peaks[i - 1])
        except IndexError as identifier:
            suma_acumulada.append(r_peaks[i] - r_peaks[i - 1])
        i += 1


# loading wfdb record #
classified_csv_beats_file = open(f'data.csv', newline='', mode='w')
writer = csv.writer(classified_csv_beats_file, dialect='excel')

for filename in os.listdir('D:\\Desarrollo\\databases\\MIT-BIH\\'):

    if (filename.endswith('.hea') or filename.endswith('.atr') or filename.endswith('.xws')):
        continue
    else:

        filename = filename.split(sep='.')
        # filename = ('108','lol')
        wfdb_record = wfdb.rdrecord(f'D:\\Desarrollo\\databases\\MIT-BIH\\{filename[0]}', channels=[0])
        wfdb_annotation = wfdb.rdann(f'D:\\Desarrollo\\databases\\MIT-BIH\\{filename[0]}', 'atr')

        record_p_signal = wfdb_record.p_signal
        signal_frecuency = wfdb_record.fs

        record_p_signal = [sample[0] for sample in record_p_signal]  # downgrading wfdb record to a simple float array



        # filtering signal #
        filtered_signal = np.array(record_p_signal)

        # detecting Beat features #
        qrs_detector = Detectors(signal_frecuency)

        # r_peaks = wfdb.processing.xqrs_detect(sig=filtered_signal, fs=signal_frecuency)



        # fifth_minute_r_peak = find_r_peak_index(r_peaks,108000,250,400)

        # qrs_sections = get_qrs_sections(r_peaks[fifth_minute_r_peak:],filtered_signal)

        # r_peak_counter = fifth_minute_r_peak # first after 5 minutes


#         suma_acumulada = []
#         fill_RR_suma_acumulada(r_peaks, suma_acumulada)
#
#         i = 7
#         end = len(r_peaks) - 1
#         GLOBAL_RR_AVERAGE = int(suma_acumulada[-1] / len(suma_acumulada))
#         beats = []
#
#         with open(f"{CURRENT_DIRECTORY}\\samples_txt\\r_peaks vs samples {filename[0]}.txt", 'w') as f:
#             j, k = 0, 0
#             peak_amount = len(wfdb_annotation.sample) if len(wfdb_annotation.sample) < len(r_peaks) else len(r_peaks)
#
#             classified_beats = {
#                 'id': str(wfdb_record.record_name),
#                 'normal_beats': [],
#                 'abnormal_beats': []
#
#             }
#
#             while j < peak_amount - 2 and k < peak_amount:
#
#                 classified_beats_file = open(f'{CURRENT_DIRECTORY}\\json_info\\{filename[0]}.json', 'w')
#
#                 if wfdb_annotation.symbol[k] in NON_USEFUL_ANNOTATIONS:
#                     k += 1
#                     continue
#
#                 if abs(r_peaks[j] - wfdb_annotation.sample[k]) > 11:
#                     m = k
#                     while r_peaks[j] - wfdb_annotation.sample[m] > 0 and abs(
#                             r_peaks[j] - wfdb_annotation.sample[m]) > 11:
#                         m += 1
#                     if abs(r_peaks[j] - wfdb_annotation.sample[m]) > 11:
#                         j += 1
#                         # k += 1
#                         continue
#                     else:
#                         k = m
#
#                 # print(str(r_peaks[j]) + " " + str(wfdb_annotation.sample[j]),file=f)
#                 if wfdb_annotation.symbol[k] in ABNORMAL_BEATS:
#                     classified_beats['abnormal_beats'].append(str(r_peaks[j]))
#
#                 else:
#                     classified_beats['normal_beats'].append(str(r_peaks[j]))
#
#                 print(str(abs(r_peaks[j] - wfdb_annotation.sample[k])) + " j=" + str(j) +
#                       " " + str(r_peaks[j]) + " " + str(wfdb_annotation.sample[k]) + " " + wfdb_annotation.symbol[k],
#                       file=f)
#                 j += 1
#                 k += 1
#
#         json.dump(classified_beats, classified_beats_file)
#         classified_beats_file.close()
#
#         writer.writerow([classified_beats['id']
#                             ,
#                          2000 if len(classified_beats['normal_beats']) > 2000 else len(classified_beats['normal_beats'])
#                             , len(classified_beats['abnormal_beats'])])
#
#         print(f'done, {filename[0]}')
#
# classified_csv_beats_file.close()

for filename in os.listdir(f"{CURRENT_DIRECTORY}\\json_info\\"):
    FILENAMES.append(filename)

print('preprocessing complete')

# selecting sets #

cantidad_id_con_anormales_despreciables = 0
cantidad_items = 0
training_set_files = []
testing_set_files = []


while cantidad_items < 24:
    filename = FILENAMES[random.randint(0,len(FILENAMES)) - 1]
    print(filename)

    with open(f"{CURRENT_DIRECTORY}\\json_info\\{filename}") as file:

        filename_json_info = json.load(file)
        print(len(filename_json_info['abnormal_beats']))

        if len(filename_json_info['abnormal_beats']) < 5:
            if cantidad_id_con_anormales_despreciables < 10:

                cantidad_id_con_anormales_despreciables += 1
                training_set_files.append(filename)
                cantidad_items += 1
                FILENAMES.remove(filename)
                print(FILENAMES)
        else:
            training_set_files.append(filename)
            cantidad_items += 1
            FILENAMES.remove(filename)
            print(FILENAMES)

testing_set_files = FILENAMES

print("trainig:" + f"{training_set_files}\n")

training_abnormal_beat_N = 0
for filename in training_set_files:
    with open(f"{CURRENT_DIRECTORY}\\json_info\\{filename}") as file:
        json_info = json.load(file)
        training_abnormal_beat_N += len(json_info['abnormal_beats'])

print(training_abnormal_beat_N)

print("testing:" + f"{testing_set_files}\n")





### detecting Beat features ###

### plotting ###
"""i = r_peaks[89]
times = np.arange(i+3600, dtype = 'float') #/ signal_frecuency
#plt.xlabel('Time [s]')
plt.xlabel('Sample')
plt.ylabel('mV')

while i < r_peaks[89]+ 3600:

    if i in r_peaks:
        plt.scatter(times[i],filtered_signal[i],c='red')
    elif i in wfdb_annotation.sample:
        plt.scatter(times[i],filtered_signal[i],c='blue')    
    else:
        plt.scatter(times[i],filtered_signal[i],c='black')
    i += 1
    
plt.show()
"""
# plt.plot(times, filtered_signal[:3600])
