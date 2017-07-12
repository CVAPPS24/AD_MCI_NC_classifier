import os
import shutil


def select_25slices():
    #select 25 slices from 50 ones
    root_dir = '/media/linlin/bmeyanglab/Brain_research/GoogLeNet_inputs'
    in_folder = 'Category3_slice50_patient'
    out_folder = 'Category3_slice25_patient'

    in_path = os.path.join(root_dir, in_folder)
    out_path = os.path.join(root_dir, out_folder)

    for mid_folder in os.listdir(in_path):
        mid_path = os.path.join(in_path, mid_folder)
        for categry_folder in os.listdir(mid_path):
            categry_path = os.path.join(mid_path, categry_folder)
            for subj_folder in os.listdir(categry_path):
                subj_path = os.path.join(categry_path, subj_folder)
                slice_files = os.listdir(subj_path)
                for i in xrange(0, len(slice_files), 2):
                    slice_file = slice_files[i]
                    slice_src = os.path.join(subj_path, slice_file)
                    slice_dst = slice_src.replace(in_folder, out_folder)
                    slice_dir_dst = subj_path.replace(in_folder, out_folder)
                    if not os.path.exists(slice_dir_dst):
                        os.makedirs(slice_dir_dst)
                    shutil.copyfile(slice_src, slice_dst)


if __name__ == '__main__':
    select_25slices()
