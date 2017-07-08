import os, time, pdb, math
import numpy as np
import deepdish as dd
from keras import optimizers
from keras import backend as K
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import LearningRateScheduler
from keras.callbacks import Callback
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def read_file(category_path, method_folder, folder):
    googLeNet_feat_path = os.path.join(category_path, 'output', method_folder, folder+'_feat_googLeNet.h5')
    # pdb.set_trace()
    googLeNet_feats = dd.io.load(googLeNet_feat_path)
    googLeNet_feat_file_path = os.path.join(category_path, 'output', method_folder, folder+'_feat_filenames_googLeNet.h5')
    filenames = dd.io.load(googLeNet_feat_file_path)
    # pdb.set_trace()
    return googLeNet_feats, filenames

def transfer_rnn_input(googLeNet_feats, filenames, ):
    # filename = '0/ADNI_031_S_4721_MR_MT1__N3m_Br_20121001153406706_S164496_I337544_77.png'
    patients_feats = {}
    patients_labels = {}
    '''
        Count the info of a patient
    '''
    for i, filename in enumerate(filenames):
        # print('{} {} slice'.format(i, filename))

        if 'ADNI_bet-B' in filename:
            continue
        #ind = filename.find('_', -10, -3)
        ind = filename.rfind('_')
        patient_timepoint_name = filename[:ind]
        # patient_timepoint_name = filename[7:ind]
        feat = googLeNet_feats[i]
        if int(filename[0]) == 9:
            pdb.set_trace()
        # if patient_timepoint_name == '941_S_4255_MR_MT1__GradWarp__N3m_Br_20120626155048530_S151272_I312782':
        #     pdb.set_trace()
        # if patient_timepoint_name == '941_S_4376_MR_MT1__GradWarp__N3m_Br_20120416140131790_S145321_I297644':
        #     pdb.set_trace()
        # if patient_timepoint_name == '941_S_4066_MR_MT1__GradWarp__N3m_Br_20110623112737722_S112064_I241396':

        if patient_timepoint_name in patients_feats.keys():   # patient_timepoint_name exists
            timepoint_feat = patients_feats[patient_timepoint_name]  # get the timepoints_dict
            assert patients_labels[patient_timepoint_name] == int(filename[0])
            timepoint_feat_new = np.concatenate((timepoint_feat, feat[np.newaxis, :]), axis = 0)
            patients_feats[patient_timepoint_name] = timepoint_feat_new
        else:   # patient_timepoint_name does not exist
            patients_labels[patient_timepoint_name] = int(filename[0])
            patients_feats[patient_timepoint_name] = feat[np.newaxis, :]

    '''
        transfer dictionary format into list format
    '''
    patients_labels_list = []
    patients_feats_list = []
    patients_name_list = []
    for i, pid in enumerate(patients_labels.keys()):
        label = patients_labels[pid]
        # if i == 0:
        #     patients_labels_list = label
        #     patients_feats_list = patients_feats[pid].tolist()
        #     patients_name_list = pid
        # else:
        patients_labels_list.append(label)
        patients_feats_list.append(patients_feats[pid].tolist())
        patients_name_list.append(pid)
    # pdb.set_trace()
    patients_labels_list = np_utils.to_categorical(patients_labels_list, 2)
    return patients_feats_list, patients_labels_list, patients_name_list

def train_rnn_model(X_train, y_train, X_vald, y_vald, options):
    out_category = options['out_category']
    lrate = options['lrate']
    epoch_num = options['epoch_num']
    batch_size = options['batch_size']
    num_clusters = options['num_clusters']
    max_timesteps = options['max_timesteps']
    model = Sequential()   #The model needs to know what input shape it should expect, so the first layer in a Sequential model (and only the first) needs to receive information about its input shape.
    # model.add(Embedding(len(), output_dim = 256))
    model.add(LSTM(128, input_shape = (max_timesteps, num_clusters), activation='tanh', recurrent_activation='elu', return_sequences = False, stateful = False, name='lstm_layer'))
    model.add(Dropout(0.5, name = 'dropout_layer'))
    model.add(Dense(out_category, activation = 'softmax', name='dense_layer'))
    optimizer = optimizers.RMSprop(lr=lrate)
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [lrate]
    history = model.fit(X_train, y_train, validation_data=(X_vald, y_vald), epochs = epoch_num, batch_size = batch_size,
        shuffle = True, callbacks=callbacks_list,)
    draw_history(history)
    # since the loss is 'categorical_crossentropy', y_train and y_vald are one-hot vectors
    return model

def draw_history(history):
    print '-----------draw history-----------'
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # pdb.set_trace()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def majority_voting(patient_timepoint_names, labels_pred, labels_gt):
    TP, TN, FP, FN, PP, PN = .0, .0, .0, .0, .0, .0
    PPV = 0.0  # Positive predictive value (precision) = sum(TP)/sum(PP)
    FOR = 0.0  # False Omission Rate (FOR) = sum(FN)/sum(FN)
    FDR = 0.0  # False Discover Rate (FDR) = sum(FP)/sum(PP)
    NPV = 0.0  # Negative Predictive Value (NPV) = sum(TN)/sum(PN)
    # patient_name = '1/ADNI_123_S_4526_MR_MT1__GradWarp__N3m_Br_20120919120639394_S161113_I335389'
    patient_label_pred = {}
    patient_label_gt = {}
    for i, patient_timepoint_name in enumerate(patient_timepoint_names):
        inds = []
        index = patient_timepoint_name.find('_')
        while index != -1:
            inds.append(index)
            index = patient_timepoint_name.find('_', index+1)
        patient_name = patient_timepoint_name[: inds[3]]
        label_pred = labels_pred[i]
        if patient_name in patient_label_pred.keys():
            label_pred_list = patient_label_pred[patient_name]
            label_pred_list.append(label_pred)
            patient_label_pred[patient_name] = label_pred_list
        else:
            label_pred_list = []
            label_pred_list.append(label_pred)
            patient_label_pred[patient_name] = label_pred_list
            label_gt_list = []
            label_gt_list.append(labels_gt[i])
            patient_label_gt[patient_name] = label_gt_list

    # patients' label
    count = 0
    num_patients_0, num_patients_1 = 0, 0
    wrong_patients = []
    print('Wrong_classify filenames:')
    for pid in patient_label_pred.keys():
        label_gt = patient_label_gt[pid]
        labels_pred = patient_label_pred[pid]
        label_pred = np.argmax(np.bincount(labels_pred))
        if label_pred == 0:
            PP += 1.0
            if label_gt == label_pred:
                num_patients_0 += 1
                TP += 1.0
            else:
                num_patients_1 += 1
                FP += 1.0
                count += 1
                wrong_patients.append(pid)
                # print(pid)
        else:
            PN += 1.0
            if label_gt == label_pred:
                num_patients_1 += 1
                TN += 1.0
            else:
                num_patients_0 += 1
                FN += 1.0
                count += 1
                wrong_patients.append(pid)
                # print(pid)
    # pdb.set_trace()
    PPV = 1.0*TP/(PP+np.finfo(float).eps)
    FOR = 1.0*FN/(PN+np.finfo(float).eps)
    FDR = 1.0*FP/(PP+np.finfo(float).eps)
    NPV = 1.0*TN/(PN+np.finfo(float).eps)
    # pdb.set_trace()
    patient_acc = 1.0*(TP+TN)/len(patient_label_pred)

    print('Patient_acc {}'.format(patient_acc))
    print('Confusion matrix:\n{}  {}  \n{}  {}'.format(PPV, FOR, FDR, NPV))
    # num_patients_0 = np.sum(np.array(patient_label_gt.keys()) == 0)
    # num_patients_1 = np.sum(np.array(patient_label_gt.keys()) == 1)
    print('GT: {} patients {} MCI_patients {} NC_patients'.format(len(patient_label_gt.keys()), num_patients_0, num_patients_1))
    wrong_patients.sort()
    for wrong_patient in wrong_patients:
        print(wrong_patient)
    print('{} Wrong_classify'.format(count))

def check_pred(filenames, labels_pred):
    count = 0
    filenames_wrong = []
    for i, filename in enumerate(filenames):
        label_pred = labels_pred[i]
        label_gt = int(filename[0])
        if label_gt == 9:
            pdb.set_trace()
        if label_pred != label_gt:
            filenames_wrong.append(filename)
            count += 1
            # print('{} {}: gt: {} pred: {}'.format(i, filename, label_gt, label_pred))
    filenames_wrong.sort()
    print('Wrong filenames:')
    for filename in filenames_wrong:
        print (filename)
    print('There are {} incorrect prediction'.format(count))

def evaluate_no_majority_voting(labels_pred, patients_label_gt_in):
    TP, TN, FP, FN, PP, PN = .0, .0, .0, .0, .0, .0
    PPV = 0.0  # Positive predictive value (precision) = sum(TP)/sum(PP)
    FOR = 0.0  # False Omission Rate (FOR) = sum(FN)/sum(FN)
    FDR = 0.0  # False Discover Rate (FDR) = sum(FP)/sum(PP)
    NPV = 0.0  # Negative Predictive Value (NPV) = sum(TN)/sum(PN)
    for i, patient_label_pred in enumerate(labels_pred):
        # patient_label_pred = pid
        patient_label_gt = patients_label_gt_in[i]
        # patient_label_gt = np.argmax()
        if patient_label_pred == 0:
            PP += 1.0
            if patient_label_gt == patient_label_pred:
                TP += 1.0
            else:
                FP += 1.0
        else:
            PN += 1.0
            if patient_label_gt == patient_label_pred:
                TN += 1.0
            else:
                FN += 1.0

    PPV = 1.0*TP/(PP+np.finfo(float).eps)
    FOR = 1.0*FN/(PN+np.finfo(float).eps)
    FDR = 1.0*FP/(PP+np.finfo(float).eps)
    NPV = 1.0*TN/(PN+np.finfo(float).eps)
    patient_acc = 1.0*(TP+TN)/len(patients_label_gt_in)
    print('No_majority_voting Patient_acc {}'.format(patient_acc))
    print('Confusion matrix:\n{}  {}  \n{}  {}'.format(PPV, FOR, FDR, NPV))
    num_patients_0 = np.sum(np.array(patients_label_gt_in) == 0)
    num_patients_1 = np.sum(np.array(patients_label_gt_in) == 1)
    print('GT: {} patients {} MCI_patients {} NC_patients'.format(len(patients_label_gt_in), num_patients_0, num_patients_1))

def evaluate_classifier_majority_voting(rnn_model, patients_bow_in, patients_label_gt_in, patients_name_list, batch_size):
    TP, TN, FP, FN, PP, PN = .0, .0, .0, .0, .0, .0
    PPV = 0.0  # Positive predictive value (precision) = sum(TP)/sum(PP)
    FOR = 0.0  # False Omission Rate (FOR) = sum(FN)/sum(FN)
    FDR = 0.0  # False Discover Rate (FDR) = sum(FP)/sum(PP)
    NPV = 0.0  # Negative Predictive Value (NPV) = sum(TN)/sum(PN)

    labels_prob = rnn_model.predict(patients_bow_in, batch_size)
    labels_pred = np.argmax(labels_prob, axis = 1)
    patients_label_gt_in = np.argmax(patients_label_gt_in, axis = 1)
    evaluate_no_majority_voting(labels_pred, patients_label_gt_in)
    check_pred(patients_name_list, labels_pred)
    print '-----------------------------------'
    majority_voting(patients_name_list, labels_pred, patients_label_gt_in)


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 1e-5
    drop = 0.5
    epochs_drop = 5
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    print("lrate is {}".format(lrate))
    return lrate

def check_pred(filenames, labels_pred):
    count = 0
    filenames_wrong = []
    for i, filename in enumerate(filenames):
        label_pred = labels_pred[i]
        label_gt = int(filename[0])
        if label_gt == 9:
            pdb.set_trace()
        if label_pred != label_gt:
            filenames_wrong.append(filename)
            count += 1
            # print('{} {}: gt: {} pred: {}'.format(i, filename, label_gt, label_pred))
    filenames_wrong.sort()
    print('Wrong filenames:')
    for filename in filenames_wrong:
        print (filename)
    print('There are {} incorrect prediction'.format(count))


if __name__ == '__main__':

    options = {
        'GPU_id': '3',
        'out_category': 2,
        'lrate': 1e-5,
        'epoch_num': 20,
        'batch_size': 8,
        'num_clusters': 2048,   #60,
        'category_folder_name': 'NCvsMCI',
        'method_folder': 'baseline',
        'max_timesteps': 50,
        'root_dir': '/data/data2/Linlin/GoogLeNet_input/Slice50',
    }
    print '-------------Train RNN model-------------'
    method_folder = options['method_folder']
    folder = 'train'
    category_path = os.path.join(options['root_dir'], options['category_folder_name'])
    train_feats, train_filenames = read_file(category_path, method_folder, folder)
    train_feats_list,train_labels_list, train_patients_name_list = transfer_rnn_input(train_feats, train_filenames)
    folder = 'vald'
    category_path = os.path.join(options['root_dir'], options['category_folder_name'])
    vald_feats, vald_filenames = read_file(category_path, method_folder, folder)
    vald_feats_list,vald_labels_list, vald_patients_name_list = transfer_rnn_input(vald_feats, vald_filenames)
    # pdb.set_trace()
    rnn_model = train_rnn_model(train_feats_list, train_labels_list, vald_feats_list, vald_labels_list, options)

    print('Saving model and weights...')
    # save model(.json) and weights(.h5)
    out_dir = os.path.join(options['root_dir'], 'output', method_folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model_json = rnn_model.to_json()
    with open(os.path.join(out_dir, 'rnn_model.json'), 'w') as json_file:
        json_file.write(model_json)
    weight_savepath = os.path.join(out_dir, 'rnn_weights.h5')
    rnn_model.save_weights(weight_savepath)

    print '-------------Evaluate Vald-------------'
    evaluate_classifier_majority_voting(rnn_model, vald_feats_list, vald_labels_list, vald_patients_name_list, options['batch_size'])
    # vald_labels_pred = evaluate_classifier(rnn_model, vald_feats_list, vald_labels_list, options['batch_size'])
    # check_pred(vald_patients_name_list, vald_labels_pred)
    # pdb.set_trace()
    # vald_matrx = rnn_model.evaluate(vald_feats_list, vald_labels_list, batch_size = options['batch_size'], verbose = 0)
    # print('\nEvaluate Vald loss and acc')
    # print(vald_matrx)
    print '-------------Evaluate Test-------------'
    folder = 'test'
    category_path = os.path.join(options['root_dir'], options['category_folder_name'])
    test_feats, test_filenames = read_file(category_path, method_folder, folder)
    test_feats_list,test_labels_list, test_patients_name_list = transfer_rnn_input(test_feats, test_filenames)
    evaluate_classifier_majority_voting(rnn_model, test_feats_list, test_labels_list, test_patients_name_list, options['batch_size'])
    # test_labels_pred = evaluate_classifier(rnn_model, test_feats_list, test_labels_list, options['batch_size'])
    # check_pred(test_patients_name_list, test_labels_pred)
    # test_matrx = rnn_model.evaluate(test_feats_list, test_labels_list, batch_size = options['batch_size'] )
    # print('Test loss and acc')
    # print(test_matrx)
