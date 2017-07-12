import os, time, pdb, math
import numpy as np
import deepdish as dd
from numpy import linalg as LA
from keras import optimizers
from keras import backend as K
from keras.models import Sequential
from keras.utils import np_utils
from keras.callbacks import TensorBoard
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
    filename_path = os.path.join(os.path.dirname(__file__), 'output', category_folder_name, folder+'_filenames_googLeNet.h5')
    feat_path = os.path.join(os.path.dirname(__file__), 'output', category_folder_name, folder+'_feat_googLeNet.h5')
    filenames = dd.io.load(filename_path)
    feats = dd.io.load(feat_path)
    # do L2 normalization
    feats = normalize_feat(feats)

    # 2. slices' bow features
    print('Beginning Kmeans training...')
    # clustering features
    kmeans_model = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, n_jobs=-1)
    kmeans_model.fit(feats)

    return kmeans_model

def generate_patient_bow_feats(folder, category_folder_name, kmeans_model, num_clusters):
    '''
        1. read files and normalize feat    2. produce slice bow features
        3. produce a list of patients' bow feature containing different timepoints
           the input of rnn is the patients' bow of different timepoints
    '''
    # 1. read files about slices
    filename_path = os.path.join(os.path.dirname(__file__), 'output', category_folder_name, folder+'_filenames_googLeNet.h5')
    feat_path = os.path.join(os.path.dirname(__file__), 'output', category_folder_name, folder+'_feat_googLeNet.h5')
    filenames = dd.io.load(filename_path)
    feats = dd.io.load(feat_path)
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
    # patients_label_gt_np = np.zeros((len(patients_label_gt_list), 2))   #
    # patients_label_gt_np[np.arange(len(patients_label_gt_list)), np.array(patients_label_gt_list)] = 1
    patients_label_gt_utils = np_utils.to_categorical(patients_label_gt_list, 2)

    return patients_bow_list, patients_label_gt_utils

def train_rnn_model(X_train, y_train, X_vald, y_vald, options):
    out_category = options['out_category']
    lrate = options['lrate']
    epoch_num = options['epoch_num']
    batch_size = options['batch_size']
    num_clusters = options['num_clusters']
    max_timesteps = options['max_timesteps']
    category_folder_name = options['category_folder_name']
    model = Sequential()   #The model needs to know what input shape it should expect, so the first layer in a Sequential model (and only the first) needs to receive information about its input shape.
    print('batch_size is {}'.format(batch_size))
    tb_callback = TensorBoard(log_dir='./graph/'+category_folder_name, histogram_freq=5, write_graph=True,  write_images=True)
    model.add(LSTM(128, input_shape = (max_timesteps, num_clusters), activation='tanh', recurrent_activation='elu', return_sequences = False, stateful = False, name='lstm_layer'))
    model.add(Dropout(0.5, name = 'dropout_layer'))
    model.add(Dense(out_category, activation = 'softmax', name='dense_layer'))
    optimizer = optimizers.RMSprop(lr=lrate)
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_vald, y_vald), epochs = epoch_num, batch_size = batch_size,
        shuffle = True, callbacks = [tb_callback])
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

def main_funct(in_folder_name, options):
    '''
        1. generate kmeans_model  2. generate patients' bow feats of train and vald data
        3. train RNN model  4. evaluate RNN_model based on the bow feats of train, vald and test data
    '''
    num_clusters = options['num_clusters']
    max_timesteps = options['max_timesteps']

    # 1. generate kmeans_model based on the bow feats of training data
    kmeans_model = generate_kmeans_model(in_folder_name, num_clusters)

    # 2. extract patients' BOW feats of train data and train svm model
    folder = 'train'
    train_patients_bow, train_patient_label_gt = generate_patient_bow_feats(folder, in_folder_name, kmeans_model, num_clusters)
    train_patients_bow_in, train_patient_label_gt_in = transfer_rnn_input(train_patients_bow, train_patient_label_gt, max_timesteps)
    folder = 'vald'
    vald_patients_bow, vald_patient_label_gt = generate_patient_bow_feats(folder, in_folder_name, kmeans_model, num_clusters)
    vald_patients_bow_in, vald_patient_label_gt_in = transfer_rnn_input(vald_patients_bow, vald_patient_label_gt, max_timesteps)

    # 3. train RNN
    print '-------------Train RNN model-------------'
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
    print '-------------Evaluate Vald Data-------------'
    vald_acc = evaluate_classifier(rnn_model, vald_patients_bow_in, vald_patient_label_gt_in, options['batch_size'])
    # pdb.set_trace()
    # vald_matrx = rnn_model.evaluate(vald_patients_bow_in, vald_patient_label_gt_in, batch_size = options['batch_size'], verbose = 0)
    # print('\nEvaluate Vald loss and acc')
    # print(vald_matrx)
    print '-------------Evaluate Test Data-------------'
    folder = 'test'
    vald_patients_bow, vald_patient_label_gt = generate_patient_bow_feats(folder, in_folder_name, kmeans_model, num_clusters)
    test_patients_bow_in, test_patient_label_gt_in = transfer_rnn_input(vald_patients_bow, vald_patient_label_gt, max_timesteps)
    test_acc = evaluate_classifier(rnn_model, test_patients_bow_in, test_patient_label_gt_in, options['batch_size'])
    # test_matrx = rnn_model.evaluate(test_patients_bow_in, test_patient_label_gt_in, batch_size = options['batch_size'] )
    # print('Test loss and acc')
    # print(test_matrx)

if __name__ == '__main__':
    options = {
        'GPU_id': '0',
        'out_category': 2,
        'lrate': 1e-6,
        'epoch_num': 250,
        'batch_size': 1,
        'num_clusters': 128,
        'category_folder_name': 'NCvsAD',
        'suffix': '_normslice',
        'max_timesteps':8,
    }

    config_GPU(options['GPU_id'])
    in_folder_name = options['category_folder_name']+options['suffix']
    # score_max = []  # classification acc under diff 'num_clusters'
    main_funct(in_folder_name, options)
