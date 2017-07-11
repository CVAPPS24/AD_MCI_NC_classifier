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
    lr_update_epoch_num = options['lr_update_epoch_num']
    # file_rm_epoch_num = options['file_rm_epoch_num']
    out_dir = options['out_dir']
    rg = options['rg']
    lr_update_epoch_num = options['lr_update_epoch_num']

    print("--------- Generating data---------")
    datagen = ImageDataGenerator(preprocessing_function = preprocessing)
    vald_generator = datagen.flow_from_directory(vald_dir, target_size=(img_size, img_size),
                      batch_size = batch_size, class_mode = 'categorical')
    vald_num = len(vald_generator.filenames)

    print("--------- Training model Total {} epochs---------".format(epoch_num))
    # 1. generate augmentated data
    # 2. produce image_generator
    # 3. train data
    for epoch_ite in xrange(epoch_num):

        if (epoch_ite != 0) & ((epoch_ite % lr_update_epoch_num) == 0):   # update learning rate
            lrate = lrate_update(options, epoch_ite)
        tb = time.time()
        train_dir_new = image_augmentation(train_dir, rg, slice_num=50, row_axis=0, col_axis=1, channel_axis=2,
                            fill_mode='nearest', cval=0.)
        te = time.time()
        print('image_augmentation {}s'.format(te-tb))
        tb = time.time()
        train_generator = datagen.flow_from_directory(train_dir_new, target_size=(img_size, img_size),
                          batch_size = batch_size, class_mode = 'categorical')
        train_num = len(train_generator.filenames)
        histry = model.fit_generator(train_generator, samples_per_epoch = int(math.ceil(1.0*train_num/batch_size)), epochs=1,
              validation_data=vald_generator, nb_val_samples=int(math.ceil(1.0*vald_num/batch_size)), verbose=0)
        te = time.time()
        print('[Epoch {}: {}S acc {} loss {} val_acc {} val_loss {} lrate {}]'.format(epoch_ite, int(te-tb),
             histry.history['acc'], histry.history['loss'], histry.history['val_acc'],
             histry.history['val_loss'], lrate))

    print('Saving model and weights...')
    # save model(.json) and weights(.h5)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model_json = model.to_json()
    with open(os.path.join(out_dir, 'model_googLeNet.json'), 'w') as json_file:
        json_file.write(model_json)
    # model_weights = model.get_weights()
    weight_savepath = os.path.join(out_dir, 'weights_googLeNet.h5')
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
        true_slice = np.sum(patient_label_gt[pid] == labels)
        slice_acc += true_slice
        patient_acc += patient_label_gt[pid] == np.argmax(np.mean(probs, axis = 0))

    patient_num = len(patient_label_gt)
    slice_num = len(filenames)
    slice_acc /= len(filenames)
    patient_acc /= len(patient_label_gt)
    print("slice_acc is {}    patient_acc is {}".format(slice_acc, patient_acc))

    return patient_acc

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.5
    epochs_drop = 50
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    print("lrate is {}".format(lrate))

    return lrate

if __name__ == '__main__':

    options = {
        'GPU_id':'1',          # GPU id
        'img_size': 299,
        'batch_size': 1,
        'learning_rate': 1e-3,
        'momentum': 0.90,
        'out_category': 2,
        'epoch_num': 3,
        'in_out_folder': 'NCvsMCI_test',
        'lr_update_epoch_num': 50,
        'rg': 30,
    }
    # pre_dir = '/data/Linlin/GoogLeNet_inputs_patients_50vs20vs30'
    pre_dir = '/data/data2/Linlin/GoogLeNet_input'
    in_dir = os.path.join(pre_dir, options['in_out_folder'])   # input directory
    out_dir = os.path.join(pre_dir, 'output', options['in_out_folder'])   # output directory
    train_dir = os.path.join(in_dir, 'train')
    vald_dir = os.path.join(in_dir, 'vald')
    test_dir = os.path.join(in_dir, 'test')
    options['train_dir'] = train_dir
    options['vald_dir'] = vald_dir
    options['test_dir'] = test_dir
    options['out_dir'] = out_dir

    config_GPU(options['GPU_id'])
    print('============================Train model============================')
    model = build_model(options)   # build model
    trained_model = train_model(model, options)   # train_model

    print('\n=======================Evaluate and predict=======================')
    print('--------- Validation data ---------')
    vald_probs, vald_filenames = predict_funct(trained_model, vald_dir, options['img_size'], options['batch_size'])
    vald_patient_acc = patient_classify(vald_probs, vald_filenames)
    dd.io.save(os.path.join(out_dir, 'vald_probs_googLeNet.h5'), vald_probs)
    dd.io.save(os.path.join(out_dir, 'vald_filenames_googLeNet.h5'), vald_filenames)
    print('------------------  Test data  ------------------')
    test_probs, test_filenames = predict_funct(trained_model, test_dir, options['img_size'], options['batch_size'])
    test_patient_acc = patient_classify(test_probs, test_filenames)
    dd.io.save(os.path.join(out_dir, 'test_probs_googLeNet.h5'), test_probs)
    dd.io.save(os.path.join(out_dir, 'test_filenames_googLeNet.h5'), test_filenames)
    te = time.time()
    reslt_path = os.path.join(out_dir, 'results_googLeNet.txt')
    print('reslt_path is {}'.format(reslt_path))
    with open(reslt_path, 'w') as reslt_file:
        reslt_str = 'vald_patient_acc: '+str(vald_patient_acc)+'\ntest_patient_acc: '+str(test_patient_acc)
        reslt_file.write(reslt_str)
    tb = time.time()
    print("MCINC_lr0.0001_mmtm0.98: total time is {}s".format(te-tb))
    # print('vald_patient_acc: {}\ntest_patient_acc: {}\n'.format(vald_patient_acc, test_patient_acc))
