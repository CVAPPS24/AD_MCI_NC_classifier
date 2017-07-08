import os, time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
import scipy.ndimage as ndi
import shutil
import pdb

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

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

def extract_googlenet_feat_augmentated_images(train_dir, rg, row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='nearest', cval=0.):
    tb = time.time()
    categry_folders = os.listdir(train_dir)
    for categry_folder in categry_folders:
        categry_path = os.path.join(train_dir, categry_folder)
        # produce a random rotation angle each 50 slices, and then rotate these 50 slices
        filenames = os.listdir(categry_path)
        filenames.sort()
        print('{} {} files'.format(categry_folder, len(filenames)))
        tb = time.time()
        filename_pref_old = ''
        files_len = int(len(filenames)/3)
        flag_count = 1  # indicate whether to execute GoogLeNet_feat_extract or not
        for i, filename in enumerate(filenames):
            # tb = time.time()
            if i%500 == 0:
                print(i),
            ind_ = filename.rfind('_')
            filename_pref_new = filename[:ind_]
            filepath = os.path.join(categry_path, filename)
            img = scipy.misc.imread(filepath)
            img = np.stack((img,img,img), axis=2)

            if filename_pref_new != filename_pref_old:
                '''when the slice belongs to another patient, execute the following steps:
                1. when the slice # arrives to a num, extract their Googlenet feat and initial some variable
                2. generate a random angle
                '''
                if i > (flag_count*files_len):
                    tmp_feats = model.predict(x, batch_size, verbose = 1)
                    if  flag_count == 1:
                        googlenet_feats = tmp_feats
                    else:
                        googlenet_feats.concatenate((googlenet_feats, tmp_feats), axis = 0)
                    flag_count += 1
                    patient_imgs_org = []
                    patient_imgs_aug = []
                ang = np.random.uniform(-rg, rg)
                theta = np.pi / 180 * ang
                rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                            [np.sin(theta), np.cos(theta), 0],
                                            [0, 0, 1]])
                # h, w = x.shape[row_axis], x.shape[col_axis]
                h, w = img.shape[:2]
                transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
            x = apply_transform(img, transform_matrix, channel_axis, fill_mode, cval)
            if i == 0 || (i > ((flag_count-1)*files_len)):
                patient_imgs_org = img[np.newaxis,:]
                patient_imgs_aug = x[np.newaxis, :]
            else:
                img = img[np.newaxis,:]
                img_aug = x[np.newaxis, :]
                patient_imgs_org = np.concatenate((patient_imgs_org, img), axis=0)
                patient_imgs_aug = np.concatenate((patient_imgs_aug, img_aug), axis=0)
    return googlenet_feats
