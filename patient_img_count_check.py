'''
Goals:
    1. check whether a patient has 50 slices in its each time point
    2. check whether a patient only in one of the three dataset: train, vald and test
    3. count the number of patients in each dataset and each category
'''

import os
import numpy as np
import pdb

def patient_num_check(root_path):
    '''
    Goals: 1. check whether a time sequence of patients has 50 slices
           2. count the number of patients in a specific category
    '''
    filenames = os.listdir(root_path)
    filenames.sort()

    patients_names = []
    patient_count = 0
    patient_name_old = ''
    time_str_old = ''
    for i, filename in enumerate(filenames):
        if 'ADNI_bet-B' in filename:
            continue
        # print i, filename
        # pdb.set_trace()
        # filename = 'ADNI_002_S_0685_MR_MT1__N3m_Br_20120424140811404_S114048_I300265_58'
        filename_strs = filename.split('_')
        time_str_new = filename_strs[-4]
        if i % 50 == 0:
            ''' check whether each time point has 50 slices'''
            assert time_str_new != time_str_old
            time_str_old = time_str_new

        '''count the number of patient'''
        patient_ind = filename.find(time_str_new)
        patient_name_new = filename[:patient_ind]
        if patient_name_new != patient_name_old:
            assert i % 50 == 0
            patient_count += 1
            patients_names.append(patient_name_new)
            patient_name_old = patient_name_new
    # print('patient_count: {}'.format(patient_count))
    return patient_count, patients_names

def main(root_dir, dataset_names, categry_names):
    patients_names_all_dataset = {}
    for dataset_name in dataset_names:
        print('------------{}-------------'.format(dataset_name))
        patients_names_all_categrys = {}
        for categry_name in categry_names:
            # pdb.set_trace()
            # dataset_key = dataset_name+categry_name
            root_path = os.path.join(root_dir, dataset_name, categry_name)
            patient_count, patients_names = patient_num_check(root_path)
            # patients_names_all[dataset_key] = patients_names
            print('{} {} patients in {}'.format(patient_count, categry_name, dataset_name))

            patients_names_all_categrys[categry_name] = patients_names

        '''check whether each category folder has the same patient or not'''
        categry_keys = patients_names_all_categrys.keys()
        categry_len = len(categry_keys)
        for i in xrange(categry_len -1):
            key1 = categry_keys[i]
            patients_names1 = patients_names_all_categrys[key1]
            for j in xrange(i+1, categry_len):
                key2 = categry_keys[j]
                patients_names2 = patients_names_all_categrys[key2]
                patient_inter = list((set(patients_names1)) & (set(patients_names2)))
                print ('Intersection of {} and {}: {}'.format(key1, key2, patient_inter))

        patients_names_all_list = []
        for key, value in patients_names_all_categrys.iteritems():
            patients_names_all_list.extend(value)
        patients_names_all_dataset[dataset_name] = patients_names_all_list

    '''check whether each dataset folder has the same patient or not'''
    print('\n========Intersection datasets==========')
    dataset_keys = patients_names_all_dataset.keys()
    dataset_len = len(dataset_names)
    for i in xrange(dataset_len -1):
        key1 = dataset_keys[i]
        patients_names1 = patients_names_all_dataset[key1]
        for j in xrange(i+1, categry_len):
            key2 = dataset_keys[j]
            patients_names2 = patients_names_all_dataset[key2]
            patient_inter = list((set(patients_names1)) & (set(patients_names2)))
            print ('Intersection of {} and {}: {}'.format(key1, key2, patient_inter))

if __name__ == '__main__':
    dataset_names = ['train', 'vald', 'test']
    # root_dir = '/data/data2/Linlin/GoogLeNet_input/Slice50/NCvsMCIvsAD/'
    # categry_names = ['AD', 'MCI', 'NC']
    root_dir = '/data/data2/Linlin/GoogLeNet_input/Slice50/NCvsMCI/'
    categry_names = [ '0', '1']
    main(root_dir, dataset_names, categry_names)
