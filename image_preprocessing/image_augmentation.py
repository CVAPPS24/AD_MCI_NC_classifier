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

def image_augmentation(train_dir, rg, slice_num=50, row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='nearest', cval=0.):
    tb = time.time()
    train_dir_new = train_dir+'_rotate'
    if os.path.exists(train_dir_new):
        shutil.rmtree(train_dir_new)
    shutil.copytree(train_dir, train_dir_new)
    te = time.time()
    print('Remove and Copy files {}s'.format(te-tb))

    old_folder = train_dir.split('/')[-1]
    new_folder = old_folder+'_rotate/'
    old_folder += '/'

    categry_folders = os.listdir(train_dir_new)
    for categry_folder in categry_folders:
        categry_path = os.path.join(train_dir_new, categry_folder)
        # produce a random rotation angle each 50 slices, and then rotate these 50 slices
        filenames = os.listdir(categry_path)
        filenames.sort()
        print('{} {} files'.format(categry_folder, len(filenames)))
        tb = time.time()
        filename_pref_old = ''
        for i, filename in enumerate(filenames):
            # tb = time.time()
            if i%500 == 0:
                print(i),
            ind_ = filename.rfind('_')
            filename_pref_new = filename[:ind_]
            filepath = os.path.join(categry_path, filename)
            img = scipy.misc.imread(filepath)

            x = np.stack((img,img,img), axis=2)
            if filename_pref_new != filename_pref_old:
                ang = np.random.uniform(-rg, rg)
                theta = np.pi / 180 * ang
                rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                            [np.sin(theta), np.cos(theta), 0],
                                            [0, 0, 1]])

                h, w = x.shape[row_axis], x.shape[col_axis]
                #h, w = x.shape[:2]
                transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
            x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
            filename_pref_old = filename_pref_new
            img_new_path = filepath.replace(old_folder, new_folder)

            img_path_strs = img_new_path.split('.')
            img_new_path = img_path_strs[0]+'_'+str(ang)+'rotated.png'
            scipy.misc.imsave(img_new_path, x)
        te = time.time()
        print(' Augmentated img {}s'.format(te-tb))

    return train_dir_new
