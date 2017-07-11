import subprocess
import pdb
import os

def GM_extract(in_root, ext):
    count = 0
    for in_dir, folders, filepaths in os.walk(in_root):
        for filepath in filepaths:
            if filepath.endswith(ext):
                count = count + 1
                in_path = os.path.join(in_dir, filepath)
                print ("{} {}".format(count, in_path))
                subprocess.call(['fast', '-t', '1', '-P', in_path])


if __name__ == "__main__":
    root_dir = '/media/linlin/bmeyanglab/Brain_research/ADNI_images/AD'
    # root_dir = '/media/linlin/bmeyanglab/Brain_research/ADNI_images'
    in_folder = 'After_BET'
    ext = '._bet-B.nii.gz'
    out_folder = 'GM'

    categry_name = 'AD'
    print("Beginning {}...".format(categry_name))
    GM_extract(root_dir, ext)
