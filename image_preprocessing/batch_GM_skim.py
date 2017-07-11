import subprocess
import pdb
import os

def GM_extract(root_dir, in_folder, categry_name, ext, ext_change):
    count = 0
    in_root = '/media/linlin/bmeyanglab/Brain_research/ADNI_images/AD/After_BET'
    # in_root = os.path.join(root_dir, categry_name, in_folder)
    # patients_folders = os.listdir(in_root)
    for patient_folder in os.listdir(in_root):
        patient_path = os.path.join(in_root, patient_folder)
        mid_folder = os.listdir(patient_path)[0]
        for time_folder in os.listdir(os.path.join(patient_path, mid_folder)):
            time_path = os.path.join(patient_path, mid_folder, time_folder)
            for serial_folder in os.listdir(time_path):
                count += 1
                print(count),
                files = os.listdir(os.path.join(time_path, serial_folder))
                files.sort()
                if (len(files)> 5):
                    if files[4].endswith(ext_change):
                        continue
                    else:
                        print(files[0])
                else:
                    # pdb.set_trace()
                    # print(files[0])
                    in_path = os.path.join(time_path, files[0])
                    print(in_path)
                    subprocess.call(['fast', '-t', '1', '-P', in_path])

if __name__ == "__main__":

    root_dir = '/media/linlin/bmeyanglab/Brain_research/ADNI_images/'
    in_folder = 'After_BET'
    ext = '._bet-B.nii.gz'
    ext_change = '._bet-B_pve_1.nii.gz'

    categry_name = 'AD'
    print("Beginning {}...".format(categry_name))
    GM_extract(root_dir, in_folder, categry_name, ext, ext_change)

    # categry_name = 'MCI'
    # print("\nBeginning {}...".format(categry_name))
    # GM_extract(root_dir, in_folder, out_folder, categry_name, ext)

    # categry_name = 'NC'
    # print("\nBeginning {}...".format(categry_name))
    # GM_extract(root_dir, in_folder, out_folder, categry_name, ext)
