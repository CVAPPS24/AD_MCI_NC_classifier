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

def config_GPU(GPU_id):
    os.environ["CUDA_VISIBLE_DEVICES"]=GPU_id
    # Session Setting
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    K.set_session(sess)

def build_model(learning_rate, momentum, out_category):
    # load InceptionV3 without fully-connected layers
    base_model = InceptionV3(include_top = False, weights = 'imagenet')
    # Add networks
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation = 'relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(256, activation = 'relu')(x)
    # x = Dropout(0.5)(x)
    predictions = Dense(out_category, activation='softmax')(x)
    model = Model(input = base_model.input, output = predictions)
    # compile the model with a SGD/momentum optimizer and a very slow learning rate
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum), metrics=['accuracy'])
    return model

def preprocessing(x):
    x /= x.max()*1.0
    mean_float = 0.3
    return x-mean_float

def train_model(model, train_dir, vald_dir, img_size, batch_size, epoch_num, out_dir):
    print("Generating data...")
    datagen = ImageDataGenerator(preprocessing_function = preprocessing)
    train_generator = datagen.flow_from_directory(train_dir, target_size=(img_size, img_size),
                                                        batch_size=batch_size, class_mode='categorical')
    train_num = len(train_generator.filenames)
    vald_generator = datagen.flow_from_directory(vald_dir, target_size=(img_size, img_size),
                                                batch_size=batch_size, class_mode='categorical')
    vald_num = len(vald_generator.filenames)

    print("Training model...")
    # learning schedule callback
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [lrate]
    # Fit the model
    model.fit_generator(train_generator, samples_per_epoch = int(math.ceil(1.0*train_num/batch_size)), epochs=epoch_num,
                        validation_data=vald_generator, nb_val_samples=int(math.ceil(1.0*vald_num/batch_size)),
                        callbacks=callbacks_list, verbose=2)

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
    # print("Generating data...")
    datagen = ImageDataGenerator(preprocessing_function = preprocessing)
    generator = datagen.flow_from_directory(data_dir, target_size=(img_size, img_size), shuffle = False,
                                            batch_size=batch_size, class_mode='categorical')  ##Remember shuffle = False
    filenames = generator.filenames
    slice_num = len(filenames)
    # print('Evaluating data...')
    # score = model.evaluate_generator(generator, int(math.ceil(1.0*slice_num/batch_size)),)
    # print("slice_score:\n{}\n{}".format(model.metrics_names, score))

    print('Predict data...')
    probs = model.predict_generator(generator, steps = int(math.ceil(1.0*slice_num/batch_size)),
            max_q_size=10, workers=1, pickle_safe=False, verbose=0)

    return probs, filenames

def majority_voting_patient_classify(slice_probs, filenames, ):
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
        # pdb.set_trace()
        ind = filename.find('_', 4, 9)
        key_filename = filename[ind+1:ind+11]
        # print(key_filename, filename)
        # pdb.set_trace()
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
        # predict patient's label
        patient_label = np.argmax(np.bincount(np.argmax(probs, axis = 1)))
        # patient_label = np.argmax(np.mean(probs, axis = 0))  # predict the label of patient
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

    PPV = 1.0*TP/(PP+np.finfo(float).eps)
    FOR = 1.0*FN/(PN+np.finfo(float).eps)
    FDR = 1.0*FP/(PP+np.finfo(float).eps)
    NPV = 1.0*TN/(PN+np.finfo(float).eps)
    pPPV = 1.0*pTP/(pPP+np.finfo(float).eps)
    pFOR = 1.0*pFN/(pPN+np.finfo(float).eps)
    pFDR = 1.0*pFP/(pPP+np.finfo(float).eps)
    pNPV = 1.0*pTN/pPN+np.finfo(float).eps)
    patient_num = len(patient_label_gt)
    slice_num = len(filenames)
    slice_acc /= len(filenames)
    patient_acc /= len(patient_label_gt)

    print("Slice_acc is {}   Confusion matrix:\n{}\t{}\n{}\t{}".format(slice_acc, PPV, FOR, FDR, NPV))
    print("Patient_acc is {}   Confusion matrix:\n{}\t{}\n{}\t{}".format(patient_acc, pPPV, pFOR, pFDR, pNPV))

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 1e-3
    drop = 0.5
    epochs_drop = 50
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    print("lrate is {}".format(lrate))
    return lrate

def load_model(in_dir, options):
    method_categry = options['out_folder']
    googLeNet_model = options['googLeNet_model']
    weights_googLeNet = options['weights_googLeNet']
    in_path = os.path.join(in_dir, 'output', method_categry)
    model_path = os.path.join(in_path, googLeNet_model)
    json_file = open(model_path, 'r')
    model_json = json_file.read()
    json_file.close()
    trained_model = model_from_json(model_json)
    weights_path = os.path.join(in_path, weights_googLeNet)
    trained_model.load_weights(weights_path)
    return trained_model

if __name__ == '__main__':

    options = {
        'GPU_id':'0',          # GPU id
        'img_size': 299,
        'batch_size': 16,
        'lrate': 1e-4,
        'momentum': 0.90,
        'out_category': 2,
        'epoch_num': 80,
        'in_folder': 'NCvsAD',
        'out_folder': 'baseline',
        'googLeNet_model': 'model_googLeNet.json',
        'weights_googLeNet': 'weights_googLeNet.h5',
    }
    # pre_dir = '/media/linlin/bmeyanglab/Brain_research/GoogLeNet_inputs'
    pre_dir = '/data/data2/Linlin/GoogLeNet_input/Slice50'
    in_dir = os.path.join(pre_dir, options['in_folder'])   # input directory
    out_dir = os.path.join(in_dir, 'output', options['out_folder'])   # output directory
    train_dir = os.path.join(in_dir, 'train')
    vald_dir = os.path.join(in_dir, 'vald')
    test_dir = os.path.join(in_dir, 'test')
    options['out_dir'] = out_dir
    options['train_dir'] = train_dir
    options['vald_dir'] = vald_dir
    options['test_dir'] = test_dir

    config_GPU(options['GPU_id'])
    print('============================Train model============================')
    model = build_model(options['lrate'], options['momentum'], options['out_category'])   # build model
    trained_model = train_model(model, train_dir, vald_dir,
                options['img_size'], options['batch_size'], options['epoch_num'],
                out_dir)   # train_model

    # print('\n=======================Load Model=======================')
    # trained_model = load_model(in_dir, options)
    print('\n=======================Evaluate and predict=======================')
    print('------------------  Train data  ------------------')
    train_probs, train_filenames = predict_funct(trained_model, train_dir, options['img_size'], options['batch_size'])
    dd.io.save(os.path.join(out_dir, 'train_probs_googLeNet.h5'), train_probs)
    dd.io.save(os.path.join(out_dir, 'train_probs_filenames_googLeNet.h5'), train_filenames)
    majority_voting_patient_classify(train_probs, train_filenames)
    print('--------- Validation data ---------')
    vald_probs, vald_filenames = predict_funct(trained_model, vald_dir, options['img_size'], options['batch_size'])
    majority_voting_patient_classify(vald_probs, vald_filenames)
    dd.io.save(os.path.join(out_dir, 'vald_probs_googLeNet.h5'), vald_probs)
    dd.io.save(os.path.join(out_dir, 'vald_probs_filenames_googLeNet.h5'), vald_filenames)
    print('------------------  Test data  ------------------')
    test_probs, test_filenames = predict_funct(trained_model, test_dir, options['img_size'], options['batch_size'])
    majority_voting_patient_classify(test_probs, test_filenames)
    dd.io.save(os.path.join(out_dir, 'test_probs_googLeNet.h5'), test_probs)
    dd.io.save(os.path.join(out_dir, 'test_probs_filenames_googLeNet.h5'), test_filenames)
