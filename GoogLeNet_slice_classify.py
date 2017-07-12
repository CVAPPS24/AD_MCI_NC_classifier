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
    x = Dense(1024, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(out_category, activation='softmax')(x)
    model = Model(input = base_model.input, output = predictions)
    # compile the model with a SGD/momentum optimizer and a very slow learning rate
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum), metrics=['accuracy'])

    return model

def preprocessing(x):
    x /= x.max()*1.0
    mean_float = 0.3

    return x-mean_float

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 20
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    print("lrate is {}".format(lrate))

    return lrate

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
    with open(os.path.join(out_dir, 'model.json'), 'w') as json_file:
        json_file.write(model_json)
    # model_weights = model.get_weights()
    weight_savepath = os.path.join(out_dir, 'weights.h5')
    model.save_weights(weight_savepath)

    return model

def predict_funct(model, data_dir, img_size, batch_size):
    print("Generating data...")
    datagen = ImageDataGenerator(preprocessing_function = preprocessing)
    generator = datagen.flow_from_directory(data_dir, target_size=(img_size, img_size), shuffle = false,
                                            batch_size=batch_size, class_mode='categorical')  ##Remember shuffle = False
    filenames = generator.filenames
    slice_num = len(filenames)
    score = model.evaluate_generator(generator, int(math.ceil(1.0*slice_num/batch_size)),)
    print("vald_score:\n{}\n{}".format(model.metrics_names, score))

    print('predict...')
    probs = model.predict_generator(generator, steps = int(math.ceil(1.0*slice_num/batch_size)),
                                    max_q_size=10, workers=1, pickle_safe=False, verbose=0)

    return probs, filenames

def patient_classify(slice_probs, filenames, ):
    patient_pred_prob = {}
    patient_label_gt = {}
    print('patient classify...')
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
            subj_label_gt[key_filename] = int(filename[0])
            patient_pred_prob[key_filename] = value_label

    ## validate the accuracy of slices, and calculate the accuracy of patients
    slice_acc = .0
    patient_acc = .0
    print ('patient num is {}'.format(len(patient_pred_prob.keys())))
    for pid in patient_pred_prob.keys():
        probs = patient_pred_prob[pid]
        labels = np.argmax(probs, axis = 1)
        true_slice = np.sum(subj_label_gt[pid] == labels)
        slice_acc += true_slice
        patient_acc += subj_label_gt[pid] == np.argmax(np.mean(probs, axis = 0))

    patient_num = len(subj_label_gt)
    slice_num = len(filenames)
    slice_acc /= len(filenames)
    patient_acc /= len(subj_label_gt)

    print("slice_acc is {}\npatient_acc is {}".format(slice_acc, patient_acc))

    return patient_acc


if __name__ == '__main__':

    options = {
        'GPU_id':'0',          # GPU id
        'img_size': 299,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'momentum': 0.9,
        'out_category': 3,
        'epoch_num': 100,
        'in_out_folder': 'Category3_slice50',
    }

    out_dir = os.path.join(os.path.dirname(__file__), 'output', options['in_out_folder'])   # output directory
    in_dir = os.path.join('/data/Linlin/GoogLeNet_inputs', options['in_out_folder'])   # input directory
    train_dir = os.path.join(in_dir, 'train')
    vald_dir = os.path.join(in_dir, 'vald')
    test_dir = os.path.join(in_dir, 'test')

    config_GPU(options['GPU_id'])

    model = build_model(options['learning_rate'], options['momentum'], options['out_category'])   # build model
    trained_model = train_model(model, train_dir, vald_dir,
                                options['img_size'], options['batch_size'], options['epoch_num'],
                                out_dir)   # train_model

    print('Predicting validation...')
    vald_probs, vald_filenames = predict_funct(trained_model, vald_dir, options['img_size'], options['batch_size'])
    vald_patient_acc = patient_classify(vald_probs, vald_filenames)
    print('Predicting test...')
    test_probs, test_filenames = predict_funct(trained_model, test_dir, options['img_size'], options['batch_size'])
    test_patient_acc = patient_classify(test_probs, test_filenames)
    te = time.time()

    reslt_path = os.path.join(out_dir, 'results.txt')
    with open(reslt_path, 'w') as reslt_file:
        reslt_str = 'vald_patient_acc: '+str(vald_patient_acc)+'\ntest_patient_acc: '+test_patient_acc
        reslt_file.write(reslt_str)
    print("50Slice_mean_database_epoch100: total time is {}h".format((te-tb)/3600))
    # print('vald_patient_acc: {}\ntest_patient_acc: {}\n'.format(vald_patient_acc, test_patient_acc))
