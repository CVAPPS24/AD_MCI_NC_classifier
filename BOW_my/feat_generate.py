import os, time, pdb, math
import numpy as np
from numpy import linalg as LA
import deepdish as dd
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Flatten
from sklearn.cluster import KMeans
from sklearn import svm
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator


def config_GPU(GPU_id):
    os.environ["CUDA_VISIBLE_DEVICES"]=GPU_id
    # Session Setting
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    K.set_session(sess)

def preprocessing(x):
    x /= x.max()*1.0
    mean = 0.3
    return x - mean

def generate_DCNN_feat_model( options):
    # loaded the well-trained DCNN based on the files of model.json and weights.h5 from the folder of 'output'
    model_filename = options['model_filename']
    weight_filename = options['weight_filename']
    category_folder_name = options['category_folder_name']
    root_dir = options['root_dir']

    in_dir = os.path.join(root_dir, category_folder_name, 'output', options['method_folder'])
    model_path = os.path.join(in_dir,model_filename)
    weight_path = os.path.join(in_dir, weight_filename)

    with open(model_path, 'r') as json_file:
        model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(weight_path)
    print('Loaded model from disk')

    # Create a intermediate model of a trained model
    output = model.layers[-2].output
    mediate_layer_model = Model(inputs = model.input, outputs = output)
    return mediate_layer_model

def generate_feats(model, options):
    root_dir = options['root_dir']
    category_folder_name = options['category_folder_name']
    method_folder = options['method_folder']
    img_size = options['img_size']
    batch_size = options['batch_size']
    out_dir = os.path.join(root_dir, category_folder_name, 'output', method_folder)

    datagen = ImageDataGenerator(preprocessing_function = preprocessing)
    print '-------------Vald feats-------------'
    vald_dir = os.path.join(root_dir, category_folder_name, 'vald')
    vald_generator = datagen.flow_from_directory(vald_dir, target_size=(img_size, img_size),
                    batch_size=batch_size, class_mode='categorical', shuffle = False)
    vald_slice_filenames = vald_generator.filenames
    vald_num = len(vald_slice_filenames)
    vald_slice_feats = model.predict_generator(vald_generator, steps=int(math.ceil(1.0*vald_num/batch_size)), verbose = 1)
    # pdb.set_trace()
    print('Writing filenames...')
    vald_filenames_path = os.path.join(out_dir, 'vald_feat_filenames_googLeNet.h5')
    dd.io.save(vald_filenames_path, vald_slice_filenames)
    print('Writing features...')
    vald_feat_path = os.path.join(out_dir, 'vald_feat_googLeNet.h5')
    dd.io.save(vald_feat_path, vald_slice_feats)
    print('Vald Writing finished ')

    print '-------------Train feats-------------'
    train_dir = os.path.join(root_dir, category_folder_name, 'train')
    train_generator = datagen.flow_from_directory(train_dir, target_size=(img_size, img_size),
                      batch_size=batch_size, class_mode='categorical', shuffle = False)
    train_slice_filenames = train_generator.filenames
    train_num = len(train_slice_filenames)
    train_slice_feats = model.predict_generator(train_generator, steps=int(math.ceil(1.0*train_num/batch_size)), verbose = 1)
    # print('Writing filenames...')
    train_filenames_path = os.path.join(out_dir, 'train_feat_filenames_googLeNet.h5')
    dd.io.save(train_filenames_path, train_slice_filenames)
    # print('Writing features...')
    train_feat_path = os.path.join(out_dir, 'train_feat_googLeNet.h5')
    dd.io.save(train_feat_path, train_slice_feats)
    print('Train Writing finished ')

    print '-------------Test feats-------------'
    test_dir = os.path.join(root_dir, category_folder_name, 'test')
    test_generator = datagen.flow_from_directory(test_dir, target_size=(img_size, img_size),
                      batch_size=batch_size, class_mode='categorical', shuffle = False)
    test_slice_filenames = test_generator.filenames
    test_num = len(test_slice_filenames)
    test_filenames_path = np.array(test_slice_filenames)
    # print('Writing filenames...')
    test_filenames_path = os.path.join(out_dir, 'test_feat_filenames_googLeNet.h5')
    dd.io.save(test_filenames_path, test_slice_filenames)
    # print('Writing features...')
    test_slice_feats = model.predict_generator(test_generator, steps=int(math.ceil(1.0*test_num/batch_size)), verbose = 1)
    test_feat_path = os.path.join(out_dir, 'test_feat_googLeNet.h5')
    dd.io.save(test_feat_path, test_slice_feats)
    print('Test Writing finished ')

if __name__ == '__main__':
    '''
        1. construct DCNN for feat extraction
        2. predict feats of training, validation and testing data
    '''
    options = {
        'GPU_id': '0',
        'category_folder_name': 'NCvsAD',
        'method_folder': 'baseline',
        'model_filename': 'model_googLeNet.json',
        'weight_filename': 'weights_googLeNet.h5',
        'img_size': 299,
        'batch_size': 32,
        'root_dir': '/data/data2/Linlin/GoogLeNet_input/Slice50',   # the dir of train, vald and test images
    }

    config_GPU(options['GPU_id'])
    print('==================== {} ===================='.format(options['category_folder_name']))
    # 1.construct DCNN for feat extraction
    mediate_layer_model = generate_DCNN_feat_model(options)
    # 2. predict feats
    generate_feats(mediate_layer_model, options)
