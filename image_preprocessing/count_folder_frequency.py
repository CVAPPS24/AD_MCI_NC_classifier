import os
import pdb

def folder_frequency(root_path, categry_name, in_folder, folder_dct):
    path = os.path.join(root_path, categry_name, in_folder)
    subj_folders = os.listdir(path)
    print('subjs: {}'.format(len(subj_folders)))
    # pdb.set_trace()
    # folder_dct = {}
    for subj_folder in subj_folders:
        folder_name_lst = os.listdir(os.path.join(path, subj_folder))

        for folder_name in folder_name_lst:
            if folder_name in folder_dct:
                value = folder_dct[folder_name]
                value += 1
            else:
                value = 1
            folder_dct[folder_name] = value

    return folder_dct


if __name__ == '__main__':
    folder_dct = {}
    root_path = '/media/linlin/bmeyanglab/Brain_research/ADNI_images'
    in_folder = 'Original'
    categry_name = 'AD'
    print('Begining {}...'.format(categry_name))
    folder_dct = folder_frequency(root_path, categry_name, in_folder, folder_dct)

    categry_name = 'MCI'
    print('Begining {}...'.format(categry_name))
    folder_dct = folder_frequency(root_path, categry_name, in_folder, folder_dct)
    pdb.set_trace()

    categry_name = 'NC'
    print('Begining {}...'.format(categry_name))
    folder_dct = folder_frequency(root_path, categry_name, in_folder, folder_dct)
    pdb.set_trace()
    for key, value in folder_dct.items():
        print(key, value)

    # print('folder_dct is {}\n\n'.format(folder_dct))
    values = folder_dct.values()
    value_max = max(values)
    for key, value in folder_dct.items():
        print(key, value)
        if value == value_max:
            print('folder_name with maximum number is {}'.format(key))
