import os
import pdb
import numpy as np
import nibabel as nib
import random as rdm
import skimage.io
# import scipy.misc
from skimage import transform
from scipy import ndimage as ndi
import shutil

'''
'''
def normalized_select_slice(nii_prepath, filename, out_dir, bw_th,
                            slct_num, margin_num, img_size):
    # read nitfi images after BET
    filename_strs = filename.split('.')
    filename_str = filename_strs[0]
    img_org = nib.load(os.path.join(nii_prepath, filename))
    img_np_arr = img_org.get_data()   # access to the image data as a Numpy array
    # print('\nSize of img_np_arr is {}'.format(img_np_arr.shape))
    # calculate the brain size
    brain_area = []
    for j in xrange(img_np_arr.shape[0]):
        # print(j),
        img_out_tmp = img_np_arr[j,...]
        binary_global = img_out_tmp > bw_th
        fill_coins = ndi.binary_fill_holes(binary_global)  # fill the holes
        # Count the area of brain
        non_zero_count = np.count_nonzero(fill_coins)
        brain_area.append(non_zero_count)
    brain_area = np.array(brain_area)
    indice = np.argsort(brain_area)[::-1]  #sort the np.array in ascending way

    # select maximum slct_num slices and crop them
    count_tmp = 0
    for index in indice:
        #crop images and normalized into 0 and 1
        img_slice_name = filename_str+"_"+str(index)+".png"
        count_tmp += 1
        # pdb.set_trace()
        if count_tmp > slct_num:
            break
        else:
            img_out_tmp = img_np_arr[index,...]
            indices_non_zeros = np.nonzero(img_out_tmp)
            if len(indices_non_zeros) == 0:
                print('only background: {}'.format(img_slice_name))
                continue;
            ind_min = np.amin(indices_non_zeros, axis=1)
            ind_max = np.amax(indices_non_zeros, axis=1)

            # print('ind_min is {}\nind_max is {}'.format(ind_min, ind_max))
            # print('brain size is {}\t'.format(ind_max-ind_min)),
            min_margin = ind_min.min()
            max_margin = min(img_out_tmp.shape - ind_max)
            margin_num = min(margin_num, min_margin, max_margin)
            # pdb.set_trace()

            # crop image
            img_crop = img_out_tmp[ind_min[0]-margin_num: ind_max[0]+margin_num, ind_min[1]-margin_num: ind_max[1]+margin_num]

            img_crop_size = max(img_crop.shape)
            img_crop_initialize = np.zeros((img_crop_size, img_crop_size))
            img_crop_initialize[:img_crop.shape[0], :img_crop.shape[1]] = img_crop
            # pdb.set_trace()

            img_max_value = img_crop_initialize.max()
            if img_crop.max != 0:
                img_crop_initialize /= img_max_value
            # pdb.set_trace()
            img_resize = transform.resize(img_crop_initialize, (img_size, img_size))
            # pdb.set_trace()
            out_directory = os.path.join(out_dir, filename_str)
            if not os.path.exists(out_directory):
                os.makedirs(out_directory)

            # pdb.set_trace()
            out_img_path = os.path.join(out_directory, img_slice_name)
            skimage.io.imsave(out_img_path, img_resize)

def normalized_select_slice_main(in_path, patient_names, out_dir):
    for patient in patient_names:
        # print('Normalizing patient: {}'.format(patient))
        # if patient == '018_S_4696':
        #     pdb.set_trace()
        patient_prepath = os.path.join(in_path, patient)
        mid_folder = os.listdir(patient_prepath)
        time_folders = os.listdir(os.path.join(patient_prepath, mid_folder[0]))
        for time_folder in time_folders:
            # pdb.set_trace()
            time_path = os.path.join(patient_prepath, mid_folder[0], time_folder)
            seq_folders = os.listdir(time_path)
            nii_prepath = os.path.join(time_path,seq_folders[0])
            # print('nii_prepath {}'.format(nii_prepath))
            filenames = os.listdir(nii_prepath)
            if len(filenames) == 0:
                print('None: {}'.format(nii_prepath))
                continue
            filename = filenames[0]
            if filename.endswith('_bet-B.nii.gz'):
                # print('filename: {}'.format(filename))
                normalized_select_slice(nii_prepath, filename, out_dir, bw_th=30, slct_num=50, margin_num = 5, img_size = 299)
            else:
                print('filename: {}'.format(filename))

def img_prepare(in_path, out_predir, categry_name, train_patient_name,
                vald_patient_name, test_patient_name, slct_num):
    print('normalized slice...')
    # in_path = os.path.join(img_prepath, categry_name, in_folder_name)
    print('in_path is {}'.format(in_path))

    out_folder = 'train'
    print('{}...'.format(out_folder))
    out_dir = os.path.join(out_predir, out_folder, categry_name)
    normalized_select_slice_main(in_path, train_patient_name, out_dir)
    out_folder = 'vald'
    print('{}...'.format(out_folder))
    out_dir = os.path.join(out_predir, out_folder, categry_name)
    normalized_select_slice_main(in_path, vald_patient_name, out_dir)
    out_folder = 'test'
    print('{}...'.format(out_folder))
    out_dir = os.path.join(out_predir, out_folder, categry_name)
    normalized_select_slice_main(in_path, test_patient_name, out_dir)

def generate_train_vald_test(img_prepath, in_folder_name, categry_name, test_pct, vald_pct):
    # generate training, validation and testing images
    print("Generating training and test files...")
    in_path = os.path.join(img_prepath, categry_name, in_folder_name)
    # print('in_path is {}'.format(in_path))
    # pdb.set_trace()
    patient_names = os.listdir(in_path)
    patient_num = len(patient_names)
    print('categry_name: {}'.format(patient_num))
    vald_patient_num = int(patient_num*vald_pct)
    test_patient_num = int(patient_num*test_pct)

    rdm.shuffle(patient_names)
    vald_patient_names = patient_names[:vald_patient_num]
    test_patient_names = patient_names[vald_patient_num:(vald_patient_num + test_patient_num)]
    train_patient_names = patient_names[(vald_patient_num + test_patient_num):patient_num]

    # print(len(test_patient_name), len(vald_patient_name), len(train_patient_name))

    return train_patient_names, vald_patient_names, test_patient_names


if __name__ == '__main__':

    test_pct=0.2
    vald_pct= 0.2
    slct_num= 80   # the number of selected slices of each patient
    in_folder_name= 'After_BET'
    out_folder_name= 'ADvsMCIvsNC_slice80_60vs20vs20'
    img_prepath= '/media/linlin/bmeyanglab/Brain_research/ADNI_images'
    out_prepath= '/media/linlin/bmeyanglab/Brain_research/GoogLeNet_inputs'

    categry_name = 'AD'
    print("Begining {}...".format(categry_name))
    train_patient_name, vald_patient_name, test_patient_name = generate_train_vald_test(img_prepath,
                                                in_folder_name, categry_name, test_pct, vald_pct)
    out_dir = os.path.join(out_prepath, out_folder_name)
    in_dir = os.path.join(img_prepath, categry_name, in_folder_name)
    img_prepare(in_dir, out_dir, categry_name, train_patient_name,
                vald_patient_name, test_patient_name, slct_num)

    categry_name = 'MCI'
    print("Begining {}...".format(categry_name))
    train_patient_name, vald_patient_name, test_patient_name = generate_train_vald_test(img_prepath,
                                             in_folder_name, categry_name, test_pct, vald_pct)
    out_dir = os.path.join(out_prepath, out_folder_name)
    in_dir = os.path.join(img_prepath, categry_name, in_folder_name)
    img_prepare(in_dir, out_dir, categry_name, train_patient_name,
                vald_patient_name, test_patient_name, slct_num)

    categry_name = 'NC'
    print("Begining {}...".format(categry_name))
    train_patient_name, vald_patient_name, test_patient_name = generate_train_vald_test(img_prepath,
                                                in_folder_name, categry_name, test_pct, vald_pct)
    out_dir = os.path.join(out_prepath, out_folder_name)
    in_dir = os.path.join(img_prepath, categry_name, in_folder_name)
    img_prepare(in_dir, out_dir, categry_name, train_patient_name,
                vald_patient_name, test_patient_name, slct_num)
