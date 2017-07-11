import subprocess
import os
# import pdb; pdb.set_trace()

def bet_nitfi(root_dir, in_folder, out_folder, categry_name):
    count = 0
    in_predir = os.path.join(root_dir, categry_name, in_folder)
    for in_dir, folders, filepaths in os.walk(in_predir):
        for filepath in filepaths:
            count = count + 1
            print("count is {}".format(count))
            out_dir = in_dir.replace(in_folder, out_folder)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            in_path = os.path.join(in_dir, filepath)
            out_path = os.path.join(out_dir, filepath[:len(filepaths)-4]+'_bet-B.nii')
            # out_path_tmp = in_path_tmp[:len(in_path_tmp)-4] + '_bet-B.nii'
            print ("input_path is {}\noutput_path is {}\n\n".format(in_path, out_path))
            subprocess.call(['bet', in_path, out_path, '-B'])


if __name__ == "__main__":
    root_dir = '/media/linlin/bmeyanglab/Brain_research/ADNI_images'
    in_folder = 'Unselected_subjs_NC'
    out_folder = 'Unselected_subjs_bet_NC'

    categry_name = 'AD'
    print("Beginning {}...".format(categry_name))
    bet_nitfi(root_dir, in_folder, out_folder, categry_name)
