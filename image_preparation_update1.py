import os
import pdb
import time
# import shutil
import numpy as np
import nibabel as nib
from PIL import Image
from matplotlib import pyplot as plt
import skimage.io
import scipy.misc
from skimage import transform
from scipy import ndimage as ndi

def check_wrong_subj(prepath, org_bet_folder,in_folder_name):
    subj_name = 'ADNI_014_S_4080_MR_MT1__GradWarp__N3m_Br_20120125124936182_S133443_I280544'
    ext = '._bet-B.nii.gz'
    subj_path = os.path.join(prepath, org_bet_folder,in_folder_name, subj_name+ext)
    print(subj_path)
    # pdb.set_trace()
    img_nii = nib.load(subj_path)
    img_arr = img_nii.get_data()
    slice_num = img_arr.shape[0]
    img_tmp = img_arr[slice_num//2, ...]
    plt.imshow(img_tmp)
    plt.show()

def slice_select_crop(in_bet_path, out_slice_path, in_folder_name, bw_th=30, slct_num=50, margin_num = 8, img_size = 299):

    # select slices from nitfi images,
    # then crop the brain and resize into img_size
    # at last, sum the pixel values and save the resized image
    pixelvalue_count = 0
    pixelvalue_count_fl = 0.0
    in_img_path = os.path.join(in_bet_path, in_folder_name)  #the path of nitfi images
    subj_count =  len(os.listdir(in_img_path))
    for i, filename in enumerate(os.listdir(in_img_path)):
        filename_strs = filename.split('.')
        filename_str = filename_strs[0]
        print("{} subjects: {}".format(i, filename_str))

        seq_path = os.path.join(in_img_path, filename)
        # read nitfi images after BET
        img_org = nib.load(seq_path)
        img_np_arr = img_org.get_data()   # access to the image data as a Numpy array
        # select maximum 20(slct_num) slices and crop them
        brain_area = []
        for j in xrange(img_np_arr.shape[0]):
            # print(j),
            img_out_tmp = img_np_arr[j,...]
            binary_global = img_out_tmp > bw_th
            fill_coins = ndi.binary_fill_holes(binary_global)  # fill the holes
            # pdb.set_trace()
            # Count the area of brain
            non_zero_count = np.count_nonzero(fill_coins)
            brain_area.append(non_zero_count)
        brain_area = np.array(brain_area)
        indice = np.argsort(brain_area)[::-1]  #sort the np.array in ascending way
        # print brain_area[indice]
        # pdb.set_trace()
        count_tmp = 0
        for index in indice:
            #crop images and normalized into 0 and 1
            img_slice_name = filename_str+"_"+str(index)+".png"
            count_tmp += 1
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

                if ind_min.min() < margin_num:
                    margin_num = ind_min.min()
                    print(margin_num, filename_str)

                img_crop = img_out_tmp[ind_min[0]-margin_num: ind_max[0]+margin_num, ind_min[1]-margin_num: ind_max[1]+margin_num]
                img_max_value = img_crop.max()
                if img_crop.max != 0:
                    img_crop /= img_max_value
                # pdb.set_trace()
                img_resize = transform.resize(img_crop, (img_size, img_size))
                # pdb.set_trace()
                out_directory = os.path.join(out_slice_path, in_folder_name, filename_str)
                if not os.path.exists(out_directory):
                    os.makedirs(out_directory)

                pixelvalue_count += ((np.sum(img_resize))*(img_max_value))
                pixelvalue_count_fl += np.sum(img_resize)
                # pdb.set_trace()
                out_img_path = os.path.join(out_directory, img_slice_name)
                skimage.io.imsave(out_img_path, img_resize)

    return pixelvalue_count, pixelvalue_count_fl, subj_count

def average_pixel_stack_img(in_path, folder_name, ava_pixelvalue):
    for dir, folders, files in os.walk(os.path.join(in_path, folder_name)):
        for i,file in enumerate(files):
            filename_strs = file.split('.')
            filename_str = filename_strs[0]
            print("{} subjects: {}".format(i,filename_str))

            img_path = os.path.join(dir, file)
            img_org = skimage.io.imread(img_path)
            img_org /= img_org.max()
            # pdb.set_trace()
            img_stack = np.stack((img_org, img_org, img_org), axis = 2)

            # pdb.set_trace()
            img_ava = img_stack - ava_pixelvalue

            img_out_path = img_path.replace('Select_slices', 'Select_slices_ava')
            print('in_path is {}\nout_path is {}'.format(img_path, img_out_path))
            img_out_dir = img_out_path[:(len(img_out_path)-(len(filename_str)+7))]
            if not os.path.exists(img_out_dir):
                os.makedirs(img_out_dir)
            skimage.io.imsave(img_out_path, img_ava)


if __name__ == "__main__":

    img_size = 299
    slct_num = 80
    bw_th=30
    margin_num = 8
    subj_num_NC = 0
    subj_num_MCI = 0
    subj_num_AD = 0
    pixelvalue_count = 0
    prepath = '/data/Linlin/GoogLeNet_inputs'
    # org_bet_folder = 'Org_bet'
    # prepath = '/data/Linlin/GoogLeNet_inputs'
    org_bet_folder = 'NII_Folders'
    # org_slice_folder = 'Org_slices'
    slct_slice_folder = 'Select_slices'

    # Check the unexcepted file
    # check_wrong_subj(prepath, org_bet_folder, 'NC')

    in_folder_name = '0'  #'AD'
    print("\n{} begin...".format(in_folder_name))
    in_bet_path = os.path.join(prepath, org_bet_folder)
    out_slice_path = os.path.join(prepath, slct_slice_folder)
    pixelvalue_count_AD, pixelvalue_count_fl_AD, subj_num_AD = slice_select_crop(in_bet_path, out_slice_path, in_folder_name, bw_th=bw_th, slct_num=slct_num,
                                         margin_num = margin_num, img_size = img_size)

    in_folder_name = '1'   #'MCI'
    print("\n{} begin...".format(in_folder_name))
    in_bet_path = os.path.join(prepath, org_bet_folder)
    out_slice_path = os.path.join(prepath, slct_slice_folder)
    pixelvalue_count_MCI, pixelvalue_count_fl_MCI, subj_num_MCI = slice_select_crop(in_bet_path, out_slice_path, in_folder_name, bw_th=bw_th, slct_num=slct_num,
                                         margin_num = margin_num, img_size = img_size)

    in_folder_name = '2'   #'NC'
    print("\n{} begin...".format(in_folder_name))
    in_bet_path = os.path.join(prepath, org_bet_folder)
    out_slice_path = os.path.join(prepath, slct_slice_folder)
    pixelvalue_count_NC, pixelvalue_count_fl_NC, subj_num_NC = slice_select_crop(in_bet_path, out_slice_path, in_folder_name, bw_th=bw_th, slct_num=slct_num,
                                         margin_num = margin_num, img_size = img_size)

    pixelvalue_count = pixelvalue_count_AD + pixelvalue_count_MCI + pixelvalue_count_NC
    pixelvalue_count_fl = pixelvalue_count_fl_AD + pixelvalue_count_fl_MCI + pixelvalue_count_fl_NC
    pixel_num = (subj_num_AD + subj_num_MCI + subj_num_NC)*slct_num*img_size*img_size
    print("NC: {}, pixelvalues: {}\nMCI: {}, pixelvalues: {}\nAD: {}, pixelvalues: {}".format(subj_num_NC, pixelvalue_count_NC, subj_num_MCI, pixelvalue_count_MCI, subj_num_AD, pixelvalue_count_AD))
    print("sum of pixels' values is {}\ntatal pixel number is {}".format(pixelvalue_count, pixel_num))
    ava_pixelvalue = pixelvalue_count/pixel_num
    ava_pixelvalue_fl = pixelvalue_count_fl/pixel_num
    print("average pixel value is {}\naverage pixel value float is {}".format(ava_pixelvalue, ava_pixelvalue_fl))
    # pdb.set_trace()

    # folder_name = 'Select_slices'
    # in_folder_name = 'AD'
    # print("\n{} begin...".format(in_folder_name))
    # in_bet_path = os.path.join(prepath, folder_name)
    # average_pixel_stack_img(prepath,  folder_name, ava_pixelvalue)

    # in_folder_name = 'MCI'
    # print("\n{} begin...".format(in_folder_name))
    # in_bet_path = os.path.join(prepath, folder_name)
    # average_pixel_stack_img(in_path, out_path, folder_name, ava_pixelvalue)
    #
    # in_folder_name = 'NC'
    # print("\n{} begin...".format(in_folder_name))
    # in_bet_path = os.path.join(prepath, folder_name)
    # average_pixel_stack_img(in_path, out_path, folder_name, ava_pixelvalue)
