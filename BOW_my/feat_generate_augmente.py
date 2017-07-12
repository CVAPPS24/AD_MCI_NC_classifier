import os, time, math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy import linalg as LA
import scipy.misc
import scipy.ndimage as ndi
from sklearn.cluster import KMeans
from keras.models import Model
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import shutil
import pdb

def preprocessing(x):
    x /= x.max()*1.0
    mean = 0.3
    return x - mean

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x, transform_matrix, channel_axis=2, fill_mode='nearest', cval=0.0):
    '''Apply the image transformation specified by a matrix
       x: 2D numpy array, single image
       transform_matrix: numpy array specifying the geometric transformation
       channel_axis: index of axis for channel in the input tensor
       fill_model: points outside the boundaries of the input are filled according to the given mode
       cval: value used for points outside the boundaries
     Returns: the transformed version of the input
    '''
    x = np.rollaxis(x, channel_axis, 0)
    final_offine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_offine_matrix, final_offset, order=0, mode=fill_mode,cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)  # what is the purpose
    x = np.rollaxis(x, 0, channel_axis+1)
    return x

def random_rotation(x, rg, row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='nearest', cval=0.):
    """Performs a random rotation of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Rotated Numpy image tensor.
    """
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def generate_DCNN_feat_model(in_dir, model_filename, weight_filename):
    # loaded the well-trained DCNN based on the files of model.json and weights.h5 from the folder of 'output'
    model_path = os.path.join(in_dir, model_filename)
    weight_path = os.path.join(in_dir, weight_filename)

    with open(model_path, 'r') as json_file:
        model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(weight_path)
    print('Model has been loaded from disk')
    # Create a intermediate model of a trained model
    output = model.layers[-2].output
    mediate_layer_model = Model(inputs = model.input, outputs = output)
    return mediate_layer_model

def generate_googlenet_feat_augmentated_images(model, train_dir, rg, channel_axis=2,
                    fill_mode='nearest', cval=0.):
    tb = time.time()
    flag = True
    img_filename_org = []
    img_filename_aug = []
    img_label_org = []
    img_label_aug = []
    categry_folders = os.listdir(train_dir)
    for categry_folder in categry_folders:
        categry_path = os.path.join(train_dir, categry_folder)
        # produce a random rotation angle each 50 slices, and then rotate these 50 slices
        filenames = os.listdir(categry_path)
        filenames.sort()
        # pdb.set_trace()
        print('Category {} has a total of {} files'.format(categry_folder, len(filenames)))
        tb = time.time()
        filename_pref_old = ''
        for i, filename in enumerate(filenames):
            if i%500 == 0:
                print(i),
            ind_ = filename.rfind('_')
            filename_pref_new = filename[:ind_]
            filepath = os.path.join(categry_path, filename)
            img = scipy.misc.imread(filepath)
            img = np.stack((img,img,img), axis=2)

            if filename_pref_new != filename_pref_old:
                ang = np.random.uniform(-rg, rg)
                theta = np.pi / 180 * ang
                rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                            [np.sin(theta), np.cos(theta), 0],
                                            [0, 0, 1]])
                # h, w = x.shape[row_axis], x.shape[col_axis]
                h, w = img.shape[:2]
                transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
            img_aug = apply_transform(img, transform_matrix, channel_axis, fill_mode, cval)

            img = img[np.newaxis, :]
            img_aug = img_aug[np.newaxis, :]
            img_feat_org_tmp = model.predict(img, verbose = 0)
            img_feat_aug_tmp = model.predict(img_aug, verbose = 0)
            label_tmp = int(categry_folder)
            if flag == True:
                flag = False
                img_feat_org = img_feat_org_tmp
                img_feat_aug = img_feat_aug_tmp
                img_filename_org.append(filename)
                img_filename_aug.append(filename_pref_new+'_aug.png')
                img_label_org.append(label_tmp)
                img_label_aug.append(label_tmp)
            else:
                img_feat_org = np.concatenate((img_feat_org, img_feat_org_tmp), axis = 0)
                img_feat_aug = np.concatenate((img_feat_aug, img_feat_aug_tmp), axis = 0)
                img_filename_org.append(filename)
                img_filename_aug.append(filename_pref_new+'_aug.png')
                img_label_org.append(label_tmp)
                img_label_aug.append(label_tmp)

    img_feats = np.concatenate((img_feat_org, img_feat_aug), axis = 0)
    img_labels = np.concatenate((np.array(img_label_org), np.array(img_label_aug)), axis = 0)
    img_filenames = np.concatenate((np.array(img_filename_org), np.array(img_filename_aug)), axis = 0)

    return img_feats, img_labels, img_filenames

def generate_googlenet_feat_images(model, vald_dir, img_size, batch_size, ):
    datagen = ImageDataGenerator(preprocessing_function = preprocessing)
    vald_generator = datagen.flow_from_directory(vald_dir, target_size=(img_size, img_size),
                    batch_size=batch_size, class_mode='categorical', shuffle = False)
    vald_slice_filenames = vald_generator.filenames
    vald_num = len(vald_slice_filenames)
    vald_slice_feats = model.predict_generator(vald_generator, steps=int(math.ceil(1.0*vald_num/batch_size)), verbose = 1)
    return vald_slice_feats, vald_slice_filenames

def generate_kmeans_model(img_feats, num_clusters):
    # 1.do L2 normalization
    feats = normalize_feat(img_feats)
    # 2. slices' bow features
    print('Beginning Kmeans training...')
    # 3. clustering features
    kmeans_model = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, n_jobs=-1)
    kmeans_model.fit(feats)
    return kmeans_model

def normalize_feat(x):
    '''
        X: n*p, where n is # of samples, p is feat dim
        Here, only L2 is used to normalize feat,
        other normalized method should also be considered
    '''
    norm = LA.norm(x, ord = 2, axis = 1)
    return x / norm[:, np.newaxis]
