import os
import pdb
import time
import math
import numpy as np
import deepdish as dd
import tensorflow as tf
from keras import optimizers
from keras.models import Model
from keras.applications import InceptionV3
from keras.layers import Input, Dense, Dropout
from keras.models import Sequential, model_from_json
from keras.layers.pooling import GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.callbacks import Callback
from keras import backend as K
from PIL import Image
from image_augmentation import *

def config_GPU(GPU_id):
    os.environ["CUDA_VISIBLE_DEVICES"]=GPU_id
    # Session Setting
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    K.set_session(sess)

def build_model(options):
    learning_rate = options['learning_rate']
    momentum = options['momentum']
    out_category = options['out_category']

    # load InceptionV3 without fully-connected layers
    base_model = InceptionV3(include_top = False, weights = 'imagenet')
    # Add networks
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(out_category, activation='softmax')(x)
    model = Model(input = base_model.input, output = predictions)
    # compile the model with a SGD/momentum optimizer and a very slow learning rate
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum), metrics=['accuracy'])
    return model

def preprocessing(x):
    x /= x.max()*1.0
    mean_float = 0.3

    return x-mean_float

def lrate_update(options, epoch_ite):
    initial_lrate = options['lrate']
    lr_drop = options['lrate_drop']
    lr_update_epoch_num = options['lr_update_epoch_num']
    lrate = initial_lrate * math.pow(lr_drop, math.floor((1+epoch_ite)/lr_update_epoch_num))
    options['lrate'] = lrate
    return lrate

def train_model(model, options):
    epoch_num = options['epoch_num']
    img_size = options['img_size']
    batch_size = options['batch_size']
    train_dir = options['train_dir']
    vald_dir = options['vald_dir']
    lrate = options['learning_rate']
    # lr_update_epoch_num = options['lr_update_epoch_num']
    # file_rm_epoch_num = options['file_rm_epoch_num']
    out_dir = options['out_dir']
    rg = options['rg']
    save_to_dir = options['save_to_dir']

    print("--------- Generating data---------")
    train_datagen = ImageDataGenerator(preprocessing_function = preprocessing, rotation_range = rg)
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_size, img_size),
                      batch_size = batch_size, class_mode = 'categorical', save_to_dir = save_to_dir, save_format = 'png')
                    #   , save_to_dir = save_to_dir
    train_num = len(train_generator.filenames)
    datagen = ImageDataGenerator(preprocessing_function = preprocessing)
    vald_generator = datagen.flow_from_directory(vald_dir, target_size=(img_size, img_size),
                      batch_size = batch_size, class_mode = 'categorical')
    vald_num = len(vald_generator.filenames)

    print("--------- Training model Total ---------")
    # tb = time.time()
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [lrate]
    pdb.set_trace()
    model.fit_generator(train_generator, samples_per_epoch = int(math.ceil(1.0*train_num/batch_size)), epochs=epoch_num,
          validation_data=vald_generator, nb_val_samples=int(math.ceil(1.0*vald_num/batch_size)),
          callbacks=callbacks_list, verbose=2)
    # te = time.time()

    print('Saving model and weights...')
    # save model(.json) and weights(.h5)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model_json = model.to_json()
    with open(os.path.join(out_dir, 'model_googLeNet_augmentation_keras.json'), 'w') as json_file:
        json_file.write(model_json)
    # model_weights = model.get_weights()
    weight_savepath = os.path.join(out_dir, 'weights_googLeNet_augmentation_keras.h5')
    model.save_weights(weight_savepath)
    return model

def predict_funct(model, data_dir, img_size, batch_size):
    print("Generating data...")
    datagen = ImageDataGenerator(preprocessing_function = preprocessing)
    generator = datagen.flow_from_directory(data_dir, target_size=(img_size, img_size), shuffle = False,
                batch_size=batch_size, class_mode='categorical')  ##Remember shuffle = False
    filenames = generator.filenames
    slice_num = len(filenames)
    print('Evaluating data...')
    score = model.evaluate_generator(generator, int(math.ceil(1.0*slice_num/batch_size)),)
    print("slice_score:\n{}\n{}".format(model.metrics_names, score))
    print('Predict data...')
    probs = model.predict_generator(generator, steps = int(math.ceil(1.0*slice_num/batch_size)),
            max_q_size=10, workers=1, pickle_safe=False, verbose=0)
    return probs, filenames

def patient_classify(slice_probs, filenames, ):
    #slice_record
    TP, TN, FP, FN, PP, PN = .0, .0, .0, .0, .0, .0
    PPV = 0.0  # Positive predictive value (precision) = sum(TP)/sum(PP)
    FOR = 0.0  # False Omission Rate (FOR) = sum(FN)/sum(FN)
    FDR = 0.0  # False Discover Rate (FDR) = sum(FP)/sum(PP)
    NPV = 0.0  # Negative Predictive Value (NPV) = sum(TN)/sum(PN)
    #patient_record
    pTP, pTN, pFP, pFN, pPP, pPN = .0, .0, .0, .0, .0, .0
    pPPV = 0.0  # Positive predictive value (precision) = sum(TP)/sum(PP)
    pFOR = 0.0  # False Omission Rate (FOR) = sum(FN)/sum(FN)
    pFDR = 0.0  # False Discover Rate (FDR) = sum(FP)/sum(PP)
    pNPV = 0.0  # Negative Predictive Value (NPV) = sum(TN)/sum(PN)

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
        true_label = patient_label_gt[pid]
        probs = patient_pred_prob[pid]
        labels = np.argmax(probs, axis = 1)
        true_slice = np.sum(true_label == labels)
        slice_acc += true_slice
        patient_label = np.argmax(np.mean(probs, axis = 0))
        patient_acc += np.sum(true_label == patient_label)
        # count slice confusion matrix
        PP += np.sum(labels == 0)
        PN += np.sum(labels == 1)
        if true_label == 0:
            TP += true_slice
            FN += (probs.shape[0]-true_slice)
        else:
            TN += true_slice
            FP += (probs.shape[0]-true_slice)
        # count patient confusion matrix
        pPP += (patient_label == 0)
        pPN += (patient_label == 1)
        if true_label == 0:
            if true_label == patient_label:
                pTP += 1
            else:
                pFN += 1
        else:
            if true_label == patient_label:
                pTN += 1
            else:
                pFP += 1

    PPV = 1.0*TP/PP
    FOR = 1.0*FN/PN
    FDR = 1.0*FP/PP
    NPV = 1.0*TN/PN
    pPPV = 1.0*pTP/pPP
    pFOR = 1.0*pFN/pPN
    pFDR = 1.0*pFP/pPP
    pNPV = 1.0*pTN/pPN
    patient_num = len(patient_label_gt)
    slice_num = len(filenames)
    slice_acc /= len(filenames)
    patient_acc /= len(patient_label_gt)

    print("Slice_acc is {}   Confusion matrix:\n{}\t{}\n{}\t{}".format(slice_acc, PPV, FOR, FDR, NPV))
    print("patient_acc is {}".format(patient_acc, pPPV, pFOR, pFDR, pNPV))
    # return patient_acc

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 1e-4
    drop = 0.5
    epochs_drop = 20
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    print("lrate is {}".format(lrate))

    return lrate

if __name__ == '__main__':

    options = {
        'GPU_id':'0',          # GPU id
        'img_size': 299,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'momentum': 0.90,
        'out_category': 2,
        'epoch_num': 40,
        'in_folder': 'NCvsMCI_test',
        'out_folder': 'augmentation_keras',
        'rg': 30,
    }
    # pre_dir = '/data/Linlin/GoogLeNet_inputs_patients_50vs20vs30'
    pre_dir = '/data/data2/Linlin/GoogLeNet_input/Slice50'
    in_dir = os.path.join(pre_dir, options['in_folder'])   # input directory
    out_dir = os.path.join(pre_dir, options['in_folder'], 'output', options['out_folder'])   # output directory
    train_dir = os.path.join(in_dir, 'train')
    vald_dir = os.path.join(in_dir, 'vald')
    test_dir = os.path.join(in_dir, 'test')
    options['train_dir'] = train_dir
    options['vald_dir'] = vald_dir
    options['test_dir'] = test_dir
    options['out_dir'] = out_dir
    options['save_to_dir'] = train_dir+'_augmented'

    config_GPU(options['GPU_id'])
    print('============================Train model============================')
    model = build_model(options)   # build model
    trained_model = train_model(model, options)   # train_model

    # print('\n=======================Evaluate and predict=======================')
    # print('------------------  Train data  ------------------')
    # train_probs, train_filenames = predict_funct(trained_model, train_dir, options['img_size'], options['batch_size'])
    # dd.io.save(os.path.join(out_dir, 'test_probs_googLeNet.h5'), train_probs)
    # dd.io.save(os.path.join(out_dir, 'test_filenames_googLeNet.h5'), train_filenames)
    # patient_classify(train_probs, train_filenames)
    # print('--------- Validation data ---------')
    # vald_probs, vald_filenames = predict_funct(trained_model, vald_dir, options['img_size'], options['batch_size'])
    # patient_classify(vald_probs, vald_filenames)
    # dd.io.save(os.path.join(out_dir, 'vald_probs_googLeNet.h5'), vald_probs)
    # dd.io.save(os.path.join(out_dir, 'vald_filenames_googLeNet.h5'), vald_filenames)
    # print('------------------  Test data  ------------------')
    # test_probs, test_filenames = predict_funct(trained_model, test_dir, options['img_size'], options['batch_size'])
    # patient_classify(test_probs, test_filenames)
    # dd.io.save(os.path.join(out_dir, 'test_probs_googLeNet.h5'), test_probs)
    # dd.io.save(os.path.join(out_dir, 'test_filenames_googLeNet.h5'), test_filenames)

    print("MCINC_lr0.0001_mmtm0.90: total time is {}s".format(te-tb))
    # print('vald_patient_acc: {}\ntest_patient_acc: {}\n'.format(vald_patient_acc, test_patient_acc))
