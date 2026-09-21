import wfdb
from wfdb import processing
import glob
import numpy as np
import os


def read_csv(file_name):
    file_name_list = glob.glob(file_name)
    # for it in range(file_name_list.__len__()):
    #     file_name_list[it] = file_name_list[it].replace('.dat','')
    return file_name_list


def read_data_from_files(file_name_list, csv_folder_path):
    signal = []
    i = 1
    for file_name in file_name_list:
        # read_data(file_name, csv_folder_path, i)
        read_data_csv(file_name, csv_folder_path, i)
        i += 1
    return


def read_data(file_name, folder_name, file_index):
    contents = []
    with open(file_name, "r") as f:
        w = open(folder_name.format(file_index), 'w')
        for line in f.readlines():
            if '#' in line:
                continue
            w.write(line)
            # contents.append(float(line))
    return


def read_data_csv(file_name, folder_name, file_index):
    import csv
    with open(file_name, newline='') as f:
        csv_rows = csv.reader(f)
        w = open(folder_name.format(file_index), 'w')
        for line in csv_rows:
            if 'lead' in line[1]:
                continue
            w.write(line[1])
            w.write('\n')
            # contents.append(float(line))
    return


def convert_signal(folder_name):
    freq = 1200
    number_users = 2644
    i = 1
    os.chdir('src/data/converted/NeomedECGDataset-v0/new_data/')
    for i in range(1, number_users):
        each_file = '{}.txt'.format(i)
        command = 'wrsamp -F ' + \
            str(freq) + ' -i ' + each_file + ' -o ' + '\'' + str(i) + '\''
        i += 1
        print(command)
        os.system(command)
        record = wfdb.rdrecord(str(i - 1), channels=[0])


# file_name = 'src/data/CYBHi/data/long-term/*'
# file_name_list = read_csv(file_name)
# csv_folder_path = 'src/data/converted/CYBHi/csv/{}.txt'
# data = read_data_from_files(file_name_list, csv_folder_path)
# convert_signal('src/data/converted/CYBHi/csv/*')


# file_name = 'src/data/ecg-bg/data/ECG/*_1.csv'
# file_name_list = read_csv(file_name)
# csv_folder_path = 'src/data/converted/ecg_bg/data/{}.txt'
# data = read_data_from_files(file_name_list, csv_folder_path)
# convert_signal('src/data/converted/ecg_bg/data/*')


file_name = 'src/data/converted/NeomedECGDataset-v0/data/*.csv'
file_name_list = read_csv(file_name)
csv_folder_path = 'src/data/converted/NeomedECGDataset-v0/new_data/{}.txt'
data = read_data_from_files(file_name_list, csv_folder_path)
convert_signal('src/data/converted/NeomedECGDataset-v0/new_data/*')
