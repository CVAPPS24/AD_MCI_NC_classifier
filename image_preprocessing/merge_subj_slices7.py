import os
import pdb
import shutil

def merge_subjs(in_prepath, out_prepath, folder_name):
    in_path = os.path.join(in_prepath, folder_name)
    for dir, folders, filenames in os.walk(in_path):
        for filename in filenames:
            file_src = os.path.join(dir, filename)
            dir_dst = os.path.join(out_prepath, folder_name)
            if not os.path.exists(dir_dst):
                os.makedirs(dir_dst)

            file_dst = os.path.join(dir_dst, filename)
            # print('src: {}\ndst: {}'.format(file_src, file_dst))
            # pdb.set_trace()
            shutil.copy(file_src, file_dst)

def merge_main(prepath, in_folder, out_folder, folder):
    in_prepath = os.path.join(prepath, in_folder, folder)
    out_prepath = os.path.join(prepath, out_folder, folder)

    folder_name = 'AD'
    print("begin {}...".format(folder_name))
    merge_subjs(in_prepath, out_prepath, folder_name)

    folder_name = 'MCI'
    print("begin {}...".format(folder_name))
    merge_subjs(in_prepath, out_prepath, folder_name)

    folder_name = 'NC'
    print("begin {}...".format(folder_name))
    merge_subjs(in_prepath, out_prepath, folder_name)



if __name__ == '__main__':
    prepath = '/media/linlin/bmeyanglab/Brain_research/GoogLeNet_inputs'
    in_folder = 'ADvsMCIvsNC_slice50_50vs20vs30'
    out_folder ='ADvsMCIvsNC_slice50_50vs20vs30_merge'


    folder = 'train'
    print('begin {}.........'.format(folder))
    merge_main(prepath, in_folder, out_folder, folder)

    folder = 'test'
    print('begin {}.........'.format(folder))
    merge_main(prepath, in_folder, out_folder, folder)

    folder = 'vald'
    print('begin {}.........'.format(folder))
    merge_main(prepath, in_folder, out_folder, folder)
