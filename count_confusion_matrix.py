import os
import pdb
import numpy as np
import deepdish as dd
# from keras.preprocessing.image import ImageDataGenerator




'''
1. import model  2. extract features(512 units)  3. bow   4. SVM/LS/RF classification
'''


def count_confusion_matrix(file_preffix, folder_name):
    TP = 0   # the number of classify AD correctly
    TN = 0   # the number of classify NC correctly
    FP = 0   # the number of classify AD wrongly
    FN = 0   # the number of classify NC wrongly
    true_AD =0
    true_NC = 0
    slice_dir = os.path.join(os.path.dirname(__file__), 'output', folder_name)
    probs_path = os.path.join(slice_dir, file_preffix+'_probs.h5')
    filenames_path = os.path.join(slice_dir, file_preffix+'_filenames.h5')
    filenames = dd.io.load(filenames_path)
    slice_probs = dd.io.load(probs_path)

    patient_pred_prob = {}
    patient_label_gt = {}
    # print('Patient classify...')
    slice_probs_np = np.array(slice_probs)

    ## Get the true and predicted labels of subjects
    for i, filename in enumerate(filenames):
        ind = filename.find('_', -8, -1)
        key_filename = filename[:ind]

        value_label = slice_probs_np[i][np.newaxis,:]
        # print('subj_label_predict_prob is {}'.format(subj_label_predict_prob.keys()))
        if key_filename in patient_pred_prob.keys(): # calculate the probability of the same subjects
            assert int(filename[0]) == patient_label_gt[key_filename]
            patient_pred_prob[key_filename] = np.concatenate((patient_pred_prob[key_filename], value_label), axis = 0)  # a patient has a prob array
        else:
            patient_label_gt[key_filename] = int(filename[0])
            patient_pred_prob[key_filename] = value_label

    ## validate the accuracy of slices, and calculate the accuracy of patients
    slice_acc = .0
    patient_acc = .0
    print ('patient num is {}'.format(len(patient_pred_prob.keys())))

    for pid in patient_pred_prob.keys():

        probs = patient_pred_prob[pid]
        labels = np.argmax(probs, axis = 1)
        true_label = patient_label_gt[pid]
        true_slice = np.sum(patient_label_gt[pid] == labels)

        if true_label == 0:
            # pdb.set_trace()
            true_AD += len(labels)
            TP += true_slice
            FN += len(labels) - true_slice
        else:

            true_NC += len(labels)
            TN += true_slice
            FP += len(labels) - true_slice

        slice_acc += true_slice
        pred_patient_label = np.argmax(np.mean(probs, axis = 0))
        # pdb.set_trace()
        patient_acc += patient_label_gt[pid] == pred_patient_label

    patient_num = len(patient_label_gt)
    slice_num = len(filenames)
    slice_acc /= slice_num
    patient_acc /= patient_num
    print('slice_num: {}\ntrue_AD: {}\ttrue_NC: {}'.format(slice_num, true_AD, true_NC))
    print('TP: {}\tTN: {}\tFP: {}\tFN: {}'.format(TP, TN, FP, FN))
    print("slice_acc is {}    patient_acc is {}".format(slice_acc, patient_acc))

    return patient_acc

def read_images():
    print("Generating data...")
    datagen = ImageDataGenerator(preprocessing_function = preprocessing)
    generator = datagen.flow_from_directory(data_dir, target_size=(img_size, img_size), shuffle = False,
                                            batch_size=batch_size, class_mode='categorical')  ##Remember shuffle = False
    filenames = generator.filenames
    slice_num = len(filenames)

if __name__ == '__main__':
    folder_name = 'Category2_slice25_NCAD'
    # folder_name = 'Category2_slice50_patient'
    file_preffix = 'test'
    print('--------------------{}--------------------'.format(file_preffix))
    count_confusion_matrix(file_preffix, folder_name)

    file_preffix = 'vald'
    print('--------------------{}--------------------'.format(file_preffix))
    count_confusion_matrix(file_preffix, folder_name)
