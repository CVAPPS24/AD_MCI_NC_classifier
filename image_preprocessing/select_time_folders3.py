import os
import pdb
import shutil


def select_time_folder():
    root_dir = '/media/linlin/bmeyanglab/Brain_research/ADNI_images/AD'
    in_folder = 'Unselected_subjs_bet_NC'
    out_folder = 'Unselected_subjs_time_NC'

    in_path = os.path.join(root_dir, in_folder)
    out_path = os.path.join(root_dir, out_folder)

    subj_folders = os.listdir(in_path)
    for subj_folder in subj_folders:
        subj_path = os.path.join(in_path, subj_folder)
        mid_folders = os.listdir(subj_path)
        for mid_folder in mid_folders:
            mid_path = os.path.join(subj_path, mid_folder)
            time_folders = os.listdir(mid_path)
            f_old = time_folders[0]

            if len(time_folders) == 1:
                path_src = os.path.join(mid_path, f_old)
                path_dst = path_src.replace(in_folder, out_folder)
                print('path_src: {}\npath_dst{}'.format(path_src, path_dst))
                # pdb.set_trace()
                shutil.copytree(path_src, path_dst)
                continue

            for i, f_new in enumerate(time_folders):
                if f_old[0:7] != f_new[0:7]:
                    path_src = os.path.join(mid_path, f_old)
                    path_dst = path_src.replace(in_folder, out_folder)
                    shutil.copytree(path_src, path_dst)
                f_old = f_new

            path_src = os.path.join(mid_path, f_old)
            path_dst = path_src.replace(in_folder, out_folder)
            shutil.copytree(path_src, path_dst)

def check_bet_files():
    flag = True
    root_dir = '/media/linlin/bmeyanglab/Brain_research/ADNI_images/AD/Unselected_subjs_bet'
    subj_folders = os.listdir(root_dir)
    for subj_folder in subj_folders:
        subj_path = os.path.join(root_dir, subj_folder)
        mid_folders = os.listdir(subj_path)
        for mid_folder in mid_folders:
            mid_path = os.path.join(subj_path, mid_folder)
            time_folders = os.listdir(mid_path)
            for time_folder in time_folders:
                time_path = os.path.join(mid_path, time_folder)
                seq_folders = os.listdir(time_path)
                for seq_folder in seq_folders:
                    seq_path = os.path.join(time_path, seq_folder)
                    bet_folderes = os.listdir(seq_path)
                    if not bet_folderes[0].endswith('._bet-B.nii.gz'):
                        flag = False
                        print('Not bet file: {}'.format(seq_path))

    if flag:
        print('All the folders have bet files')


if __name__ == '__main__':
    # check_bet_files()
    select_time_folder()
