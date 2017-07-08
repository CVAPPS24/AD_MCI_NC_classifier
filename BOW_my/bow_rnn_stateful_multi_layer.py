import os, time, pdb, math
import numpy as np
import deepdish as dd
from numpy import linalg as LA
from keras import optimizers
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.preprocessing.sequence  import pad_sequences
from sklearn.cluster import KMeans
import tensorflow as tf


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

def generate_kmeans_model(category_folder_name, num_clusters):
    # 1. read files about slices
    folder = 'train'
    filename_path = os.path.join(os.path.dirname(__file__), 'output', category_folder_name, folder+'_filenames_bow.h5')
    feat_path = os.path.join(os.path.dirname(__file__), 'output', category_folder_name, folder+'_feat_bow.h5')
    filenames = dd.io.load(filename_path)
    feats = dd.io.load(feat_path)
    # do L2 normalization
    feats = normalize_feat(feats)

    # 2. slices' bow features
    print('Begining Kmeans...')
    # clustering features
    kmeans_model = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, n_jobs=-1)
    kmeans_model.fit(feats)

    return kmeans_model

def patient_feats(folder, category_folder_name, kmeans_model, num_clusters):
    '''
        1. read files and normalize feat    2. produce slice bow features
        3. produce a list of patients' bow feature containing different timepoints
    '''
    # 1. read files about slices
    filename_path = os.path.join(os.path.dirname(__file__), 'output', category_folder_name, folder+'_filenames_bow.h5')
    feat_path = os.path.join(os.path.dirname(__file__), 'output', category_folder_name, folder+'_feat_bow.h5')
    filenames = dd.io.load(filename_path)
    feats = dd.io.load(feat_path)
    # pdb.set_trace()
    # do L2 normalization
    feats = normalize_feat(feats)

    # 2. slices' bow features: predict the label that X belongs to which index of centers
    # print('Begining Slice BOW...')
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
        if patient_name in patients_bow.keys():   # patient_name exists or not
            timepoints_dict = patients_bow[patient_name]  # get the timepoints_dict
            if timepoint_name in timepoints_dict.keys():  # timepoint exists or not
                timepoints_dict[timepoint_name][bow_feat] += 1.0
            else:
                timepoint_bow = np.zeros(num_clusters)
                timepoint_bow[bow_feat] += 1.0
                timepoints_dict[timepoint_name] = timepoint_bow
        else:
            patients_label_gt[patient_name] = int(filename[0])
            timepoint_bow = np.zeros(num_clusters)
            timepoint_bow[bow_feat] += 1.0
            timepoint_bow_dict = {}
            timepoint_bow_dict[timepoint_name] = timepoint_bow
            patients_bow[patient_name] = timepoint_bow_dict
    return patients_bow, patients_label_gt

def transfer_rnn_input(patients_bow, patients_label_gt, max_timesteps, folder):
    patients_bow_list = []  # list of lists of patients' different timepoint feats
    patients_label_gt_list = []  # list of patients' labels
    for pid in patients_bow.keys():
        patients_label_gt_list.append(patients_label_gt[pid])
        # pdb.set_trace()
        timepoint_bow_dict = patients_bow[pid]
        timepoint_bow_list = []
        for timepoint in timepoint_bow_dict.keys():
            bow_feat_tmp = timepoint_bow_dict[timepoint]
            timepoint_bow_list.append(bow_feat_tmp)
            # bow_feats.append(bow_feat_tmp)
        timepoint_bow_list = normalize_feat(timepoint_bow_list)
        patients_bow_list.append(timepoint_bow_list)
        # if max_timesteps < len(timepoint_bow_list):
        #     max_timesteps = len(timepoint_bow_list)
    patients_bow_list = pad_sequences(patients_bow_list, max_timesteps, value = -1)
    # pdb.set_trace()
    patients_label_gt_np = np.zeros((len(patients_label_gt_list), 2))
    patients_label_gt_np[np.arange(len(patients_label_gt_list)), np.array(patients_label_gt_list)] = 1
    patient_bow_path = os.path.join(os.path.dirname(__file__), 'output', folder+'bow_feat_rnn.h5')
    dd.io.save(patient_bow_path, patients_bow_list)
    patient_label_path = os.path.join(os.path.dirname(__file__), 'output', folder+'label_rnn.h5')
    dd.io.save(patient_label_path, patients_label_gt_np)

    return patients_bow_list, patients_label_gt_np

def read_rnn_input(folder):
    patient_bow_path = os.path.join(os.path.dirname(__file__), 'output', folder+'bow_feat_rnn.h5')
    patients_bow_list = dd.io.load(patient_bow_path)
    patient_label_path = os.path.join(os.path.dirname(__file__), 'output', folder+'label_rnn.h5')
    patients_label_gt_np = dd.io.load(patient_label_path)

    return patients_bow_list, patients_label_gt_np

def train_rnn_model(X_train, y_train, X_vald, y_vald, options):
    out_category = options['out_category']
    lrate = options['lrate']
    epoch_num = options['epoch_num']
    batch_size = options['batch_size']
    num_clusters = options['num_clusters']
    max_timesteps = options['max_timesteps']
    optimizer = options['optimizer']
    stateful_flag = options['stateful_flag']
    multi_lstm_flag = options['multi_lstm_flag']
    optimizer = options['optimizer']

    if optimizer == 'RMSprop':
        optimizer = optimizers.RMSprop(lr=lrate)
    elif optimizer == 'Adam':
        optimizer = optimizers.Adam(lr=lrate)

    # pdb.set_trace()
    model = Sequential()   #The model needs to know what input shape it should expect, so the first layer in a Sequential model (and only the first) needs to receive information about its input shape.
    if (stateful_flag == True) & (multi_lstm_flag == False):
        model.add(LSTM(128, batch_input_shape = (batch_size, max_timesteps, num_clusters), activation='tanh', recurrent_activation='sigmoid', return_sequences = False, stateful = True, name='lstm_layer'))
    elif ((stateful_flag == False) & (multi_lstm_flag == True)):
        model.add(LSTM(512, input_shape = (max_timesteps, num_clusters), activation='tanh', recurrent_activation='sigmoid', return_sequences = True, stateful = False, name='lstm_layer'))
        model.add(LSTM(128, input_shape = (max_timesteps, num_clusters), activation='tanh', recurrent_activation='sigmoid', return_sequences = True, stateful = False, name='lstm_layer'))
        model.add(LSTM(64, input_shape = (max_timesteps, num_clusters), activation='tanh', recurrent_activation='sigmoid', return_sequences = False, stateful = False, name='lstm_layer'))
    elif (stateful_flag == True) & (multi_lstm_flag == True):
        pass
    else:  # single lstm layer
        model.add(LSTM(128, input_shape = (max_timesteps, num_clusters), activation='tanh', recurrent_activation='sigmoid', return_sequences = False, stateful = False, name='lstm_layer'))

    # model.add(Dropout(0.5, name = 'dropout_layer'))
    model.add(Dense(out_category, activation = 'softmax', name='dense_layer'))
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_vald, y_vald), epochs = epoch_num, batch_size = batch_size,
        shuffle = True)
    # since the loss is 'categorical_crossentropy', y_train and y_vald are one-hot vectors

    return model

def main_funct(in_folder_name, options):
    '''
        1. generate kmeans_model  2. generate patients' bow feats at different timesteps
        3. train RNN model  4. evaluate RNN_model based on the bow feats of train, vald and test data
    '''
    num_clusters = options['num_clusters']
    max_timesteps = options['max_timesteps']

    print '-------------Read bow feats-------------'
    # 1. generate kmeans_model
    kmeans_model = generate_kmeans_model(in_folder_name, num_clusters)

    # 2. extract patients' BOW feats at differnt timesteps
    folder = 'train'
    train_patients_bow_in, train_patient_label_gt_in = read_rnn_input(folder)   # read features from files
    # train_patients_bow, train_patient_label_gt = patient_feats(folder, in_folder_name, kmeans_model, num_clusters)
    # train_patients_bow_in, train_patient_label_gt_in = transfer_rnn_input(train_patients_bow, train_patient_label_gt, max_timesteps, folder)
    folder = 'vald'
    vald_patients_bow_in, vald_patient_label_gt_in = read_rnn_input(folder)   # read features from files
    # vald_patients_bow, vald_patient_label_gt = patient_feats(folder, in_folder_name, kmeans_model, num_clusters)
    # vald_patients_bow_in, vald_patient_label_gt_in = transfer_rnn_input(vald_patients_bow, vald_patient_label_gt, max_timesteps, folder)
    folder = 'test'
    test_patients_bow_in, test_patient_label_gt_in = read_rnn_input(folder)  # read features from files
    # vald_patients_bow, vald_patient_label_gt = patient_feats(folder, in_folder_name, kmeans_model, num_clusters)
    # test_patients_bow_in, test_patient_label_gt_in = transfer_rnn_input(vald_patients_bow, vald_patient_label_gt, max_timestep, folder)

    # 3. train RNN model
    rnn_model = train_rnn_model(train_patients_bow_in, train_patient_label_gt_in,
                vald_patients_bow_in, vald_patient_label_gt_in, options)

    print('Saving model and weights...')
    # save model(.json) and weights(.h5)
    out_dir = os.path.join(os.path.dirname(__file__), 'output', in_folder_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model_json = rnn_model.to_json()
    with open(os.path.join(out_dir, 'rnn_model.json'), 'w') as json_file:
        json_file.write(model_json)
    weight_savepath = os.path.join(out_dir, 'rnn_weights.h5')
    rnn_model.save_weights(weight_savepath)

    # 4. evaluate
    print '-------------Evaluate Vald and Test Data-------------'
    vald_matrx = rnn_model.evaluate(vald_patients_bow_in, vald_patient_label_gt_in, batch_size = options['batch_size'], verbose=0 )
    print('\nVALD LOSS and ACC')
    print(vald_matrx)
    test_matrx = rnn_model.evaluate(test_patients_bow_in, test_patient_label_gt_in, batch_size = options['batch_size'], verbose=0 )
    print('TEST LOSS and ACC')
    print(test_matrx)

if __name__ == '__main__':
    options = {
        'GPU_id': '0',
        'out_category': 2,
        'lrate': 1e-4,
        'epoch_num': 80,
        'batch_size': 1,
        'num_clusters': 128,
        'category_folder_name': 'NCvsMCI',
        'suffix': '_normslice',
        'max_timesteps':10,
        'optimizer':'RMSprop',
        'stateful_flag': False,
        'multi_lstm_flag': True,
    }

    config_GPU(options['GPU_id'])
    in_folder_name = options['category_folder_name']+options['suffix']

    main_funct(in_folder_name, options)
    print('STATEFUL={}, MULTI_LSTM={}'.format(options['stateful_flag'], options['multi_lstm_flag']))
