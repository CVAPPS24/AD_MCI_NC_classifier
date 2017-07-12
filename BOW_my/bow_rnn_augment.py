import os, time, pdb, math
import numpy as np
import deepdish as dd
from numpy import linalg as LA
from keras import optimizers
from keras import backend as K
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.preprocessing.sequence  import pad_sequences
from sklearn.cluster import KMeans
import tensorflow as tf
from feat_generate_augmente import *

def config_GPU(GPU_id):
    os.environ["CUDA_VISIBLE_DEVICES"]=GPU_id
    # Session Setting
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    K.set_session(sess)

def normalize_feat(x):
    '''
        X: n*p, where n is # of samples, p is feat dim
        Here, only L2 is used to normalize feat,
        other normalized method should also be considered
    '''
    norm = LA.norm(x, ord = 2, axis = 1)
    return x / norm[:, np.newaxis]


def generate_patient_bow_feats(feats, filenames, kmeans_model, num_clusters):
    '''
        1. read files and normalize feat    2. produce slice bow features
        3. produce a list of patients' bow feature containing different timepoints
           the input of rnn is the patients' bow of different timepoints
    '''
    slices_bow_feats = kmeans_model.predict(feats)

    # 3. produce patient bow feature
    # print('Beginning Patient BOW Feat...')
    patients_bow = {}
    patients_label_gt = {}
    ## Get the true and predicted labels of subjects
    for i, filename in enumerate(filenames):
        # print('{} {} slice'.format(i, filename))
        if filename[2:12] == 'ADNI_bet-B':
            continue
            # patient_name = 'ADNI_bet-B_'
        patient_name = filename[7:17]
        filename_strs = filename.split('_')
        timepoint_name = filename_strs[-4]

        bow_feat = slices_bow_feats[i]
        if patient_name in patients_bow.keys():   # patient_name exists
            timepoints_dict = patients_bow[patient_name]  # get the timepoints_dict
            if timepoint_name in timepoints_dict.keys():  # timepoint exists or not
                timepoints_dict[timepoint_name][bow_feat] += 1.0
            else:
                timepoint_bow = np.zeros(num_clusters)
                timepoint_bow[bow_feat] += 1.0
                timepoints_dict[timepoint_name] = timepoint_bow
        else:   # patient_name does not exist
            patients_label_gt[patient_name] = int(filename[0])
            timepoint_bow = np.zeros(num_clusters)
            timepoint_bow[bow_feat] += 1.0
            timepoint_bow_dict = {}
            timepoint_bow_dict[timepoint_name] = timepoint_bow
            patients_bow[patient_name] = timepoint_bow_dict
    return patients_bow, patients_label_gt

def generate_patient_bow_feats_train(feats, filenames, labels, kmeans_model, num_clusters):
    slices_bow_feats = kmeans_model.predict(feats)
    # 3. produce patient bow feature
    patients_bow = {}
    patients_label_gt = {}
    for i, filename in enumerate(filenames):
        if filename[0:10] == 'ADNI_bet-B':
            continue
            # patient_name = 'ADNI_bet-B_'
        # pdb.set_trace()
        patient_name = filename[5:15]
        filename_strs = filename.split('_')
        timepoint_name = filename_strs[-4]

        bow_feat = slices_bow_feats[i]
        if patient_name in patients_bow.keys():   # patient_name exists
            timepoints_dict = patients_bow[patient_name]  # get the timepoints_dict
            if timepoint_name in timepoints_dict.keys():  # timepoint exists or not
                timepoints_dict[timepoint_name][bow_feat] += 1.0
            else:
                timepoint_bow = np.zeros(num_clusters)
                timepoint_bow[bow_feat] += 1.0
                timepoints_dict[timepoint_name] = timepoint_bow
        else:   # patient_name does not exist
            patients_label_gt[patient_name] = labels[i]
            timepoint_bow = np.zeros(num_clusters)
            timepoint_bow[bow_feat] += 1.0
            timepoint_bow_dict = {}
            timepoint_bow_dict[timepoint_name] = timepoint_bow
            patients_bow[patient_name] = timepoint_bow_dict
    return patients_bow, patients_label_gt

def transfer_rnn_input(patients_bow, patients_label_gt, max_timesteps):
    patients_bow_list = []  # list of lists of patients' different timepoint feats
    patients_label_gt_list = []  # list of patients' labels
    for pid in patients_bow.keys():
        patients_label_gt_list.append(patients_label_gt[pid])
        timepoint_bow_dict = patients_bow[pid]
        timepoint_bow_list = []
        for timepoint in timepoint_bow_dict.keys():
            bow_feat_tmp = timepoint_bow_dict[timepoint]
            timepoint_bow_list.append(bow_feat_tmp)
            # bow_feats.append(bow_feat_tmp)
        timepoint_bow_list = normalize_feat(timepoint_bow_list)
        patients_bow_list.append(timepoint_bow_list)
    patients_bow_list = pad_sequences(patients_bow_list, max_timesteps, value = -1)
    patients_label_gt_utils = np_utils.to_categorical(patients_label_gt_list, 2)
    return patients_bow_list, patients_label_gt_utils

def train_rnn_model(X_train, y_train, X_vald, y_vald, options):
    out_category = options['out_category']
    lrate = options['lrate']
    epoch_num = options['epoch_num']
    batch_size = options['batch_size']
    num_clusters = options['num_clusters']
    max_timesteps = options['max_timesteps']
    model = Sequential()   #The model needs to know what input shape it should expect, so the first layer in a Sequential model (and only the first) needs to receive information about its input shape.
    # tf.summary.histogram('weights', )
    model.add(LSTM(128, input_shape = (max_timesteps, num_clusters), activation='tanh', recurrent_activation='elu', return_sequences = False, stateful = False, name='lstm_layer'))
    model.add(Dropout(0.5, name = 'dropout_layer'))
    model.add(Dense(out_category, activation = 'softmax', name='dense_layer'))
    optimizer = optimizers.RMSprop(lr=lrate)
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_vald, y_vald), epochs = epoch_num, batch_size = batch_size,
        shuffle = True)
    # since the loss is 'categorical_crossentropy', y_train and y_vald are one-hot vectors
    return model

def evaluate_classifier(rnn_model, patients_bow_in, patients_label_gt_in, batch_size):
    TP, TN, FP, FN, PP, PN = .0, .0, .0, .0, .0, .0
    PPV = 0.0  # Positive predictive value (precision) = sum(TP)/sum(PP)
    FOR = 0.0  # False Omission Rate (FOR) = sum(FN)/sum(FN)
    FDR = 0.0  # False Discover Rate (FDR) = sum(FP)/sum(PP)
    NPV = 0.0  # Negative Predictive Value (NPV) = sum(TN)/sum(PN)
    matrx = rnn_model.evaluate(patients_bow_in, patients_label_gt_in, batch_size = options['batch_size'] )
    print('\nloss and acc')
    print(matrx)
    labels_prob = rnn_model.predict(patients_bow_in, batch_size)
    labels_pred = np.argmax(labels_prob, axis = 1)
    patients_label_gt_in = np.argmax(patients_label_gt_in, axis = 1)
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
    # pdb.set_trace()
    PPV = 1.0*TP/PP
    FOR = 1.0*FN/PN
    FDR = 1.0*FP/PP
    NPV = 1.0*TN/PN
    # pdb.set_trace()
    patient_acc = 1.0*(TP+TN)/len(patients_label_gt_in)

    print('Patient_acc {}'.format(patient_acc))
    print('Confusion matrix:\n{}  {}  \n{}  {}'.format(PPV, FOR, FDR, NPV))
    num_patients_0 = np.sum(np.array(patients_label_gt_in) == 0)
    num_patients_1 = np.sum(np.array(patients_label_gt_in) == 1)
    print('GT: {} patients {} MCI_patients {} NC_patients'.format(len(patients_label_gt_in), num_patients_0, num_patients_1))

def main_funct(options):
    '''
        1. extract the googLeNet feats of the augmented training imgs, testing imgs and vald imgs
        2. use the above feats to train k-means model
        3. calculate the bow feats of the augmented training images, testing imgs and vald imgs
        4. train RNN according the bow feats of the augmented training imgs
        5. test the trained RNN using the bow feats of the testing and vald imgs
    '''
    num_clusters = options['num_clusters']
    max_timesteps = options['max_timesteps']
    category_folder = options['category_folder']
    googLeNet_model_folder = options['googLeNet_model_folder']
    root_dir = options['root_dir']
    train_dir = options['train_dir']
    vald_dir = options['vald_dir']
    test_dir = options['test_dir']
    out_dir = options['out_dir']    # the directory of GoogLeNet outputs
    model_filename = options['model_filename']
    weight_filename = options['weight_filename']
    batch_size = options['batch_size']
    img_size = options['img_size']
    rg = options['rg']

    pre_dir = os.path.join(root_dir, category_folder, 'output', googLeNet_model_folder)
    googLeNet_model = generate_DCNN_feat_model(pre_dir, model_filename, weight_filename)
    # 1. generate kmeans_model based on the bow feats of training data
    print '----------Generate Kmeans_model-----------'
    train_googLeNet_feats, train_googLeNet_labels, train_googLeNet_filenames = generate_googlenet_feat_augmentated_images(googLeNet_model,
                    train_dir, rg, channel_axis=2, fill_mode='nearest', cval=0.)
    train_googLeNet_feats_path = os.path.join(out_dir, 'train_googLeNet_feats_augmentated.h5')
    dd.io.save(train_googLeNet_feats_path, train_googLeNet_feats)
    train_googLeNet_filenames_path = os.path.join(out_dir, 'train_googLeNet_filenames_augmentated.h5')
    dd.io.save(train_googLeNet_filenames_path, train_googLeNet_filenames)
    kmeans_model = generate_kmeans_model(train_googLeNet_feats, num_clusters)
    # 2. extract patients' BOW feats of train data and train svm model
    folder = 'train'
    train_patients_bow, train_patient_label_gt = generate_patient_bow_feats_train(train_googLeNet_feats, train_googLeNet_filenames, train_googLeNet_labels, kmeans_model, num_clusters)
    # pdb.set_trace()
    train_patients_bow_in, train_patient_label_gt_in = transfer_rnn_input(train_patients_bow, train_patient_label_gt, max_timesteps)
    folder = 'vald'
    vald_googLeNet_feats, vald_googLeNet_filenames = generate_googlenet_feat_images(googLeNet_model, vald_dir, img_size, batch_size, )
    vald_patients_bow, vald_patients_label_gt = generate_patient_bow_feats(vald_googLeNet_feats, vald_googLeNet_filenames, kmeans_model, num_clusters)
    vald_patients_bow_in, vald_patient_label_gt_in = transfer_rnn_input(vald_patients_bow, vald_patients_label_gt, max_timesteps)

    # 3. train RNN
    print '-------------Train RNN model-------------'
    rnn_model = train_rnn_model(train_patients_bow_in, train_patient_label_gt_in,
                vald_patients_bow_in, vald_patient_label_gt_in, options)
    print('Saving model and weights...')
    # save model(.json) and weights(.h5)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model_json = rnn_model.to_json()
    with open(os.path.join(out_dir, 'rnn_model.json'), 'w') as json_file:
        json_file.write(model_json)
    weight_savepath = os.path.join(out_dir, 'rnn_weights.h5')
    rnn_model.save_weights(weight_savepath)

    # 4. evaluate
    print '-------------Evaluate Vald Data-------------'
    vald_acc = evaluate_classifier(rnn_model, vald_patients_bow_in, vald_patient_label_gt_in, batch_size)
    # pdb.set_trace()
    # vald_matrx = rnn_model.evaluate(vald_patients_bow_in, vald_patient_label_gt_in, batch_size = options['batch_size'], verbose = 0)
    # print('\nEvaluate Vald loss and acc')
    # print(vald_matrx)
    print '-------------Evaluate Test Data-------------'
    folder = 'test'
    test_googLeNet_feats, test_googLeNet_filenames = generate_googlenet_feat_images(googLeNet_model, test_dir, img_size, batch_size, )
    test_patients_bow, test_patients_label_gt = generate_patient_bow_feats(test_googLeNet_feats, test_googLeNet_filenames, kmeans_model, num_clusters)
    test_patients_bow_in, test_patient_label_gt_in = transfer_rnn_input(vald_patients_bow, vald_patients_label_gt, max_timesteps)
    test_acc = evaluate_classifier(rnn_model, test_patients_bow_in, test_patient_label_gt_in, batch_size)
    # test_matrx = rnn_model.evaluate(test_patients_bow_in, test_patient_label_gt_in, batch_size = options['batch_size'] )
    # print('Test loss and acc')
    # print(test_matrx)

if __name__ == '__main__':
    options = {
        'GPU_id': '1',
        'out_category': 2,
        'lrate': 1e-6,
        'epoch_num': 3,
        'img_size': 299,
        'batch_size': 1,
        'num_clusters': 128,
        'category_folder': 'NCvsMCI_test',
        'max_timesteps':8,
        'googLeNet_model_folder': 'baseline',
        'out_folder': 'googLeNet_feats_augmentation_rnn',
        'model_filename': 'model_googLeNet.json',
        'weight_filename': 'weights_googLeNet.h5',
        'rg': 20,
    }
    root_dir = '/data/Linlin/GoogLeNet_inputs_patients/'
    # root_dir = '/data/data2/Linlin/GoogLeNet_input/Slice50'
    options['root_dir'] = root_dir
    # root_dir = '/media/linlin/bmeyanglab/Brain_research/GoogLeNet_inputs_patients'
    out_dir = os.path.join(root_dir, options['category_folder'], 'output', options['out_folder'])
    train_dir = os.path.join(root_dir, options['category_folder'], 'train')
    test_dir = os.path.join(root_dir, options['category_folder'], 'test')
    vald_dir = os.path.join(root_dir, options['category_folder'], 'vald')
    options['out_dir'] = out_dir
    options['train_dir'] = train_dir
    options['vald_dir'] = vald_dir
    options['test_dir'] = test_dir

    config_GPU(options['GPU_id'])
    main_funct(options)
