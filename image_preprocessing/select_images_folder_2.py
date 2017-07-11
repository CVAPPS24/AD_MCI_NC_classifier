import os
from shutil import copytree

def move_img_folder(category_name):
    count = 0;
    prepath = '/media/linlin/My Passport/Brain_research/ADNI_category'
    in_folder = 'Original'
    out_folder = 'Select_Img_Folder'
    input_path = os.path.join(prepath, category_name, in_folder)
    output_prepath = os.path.join(prepath, category_name, out_folder)

    # input_path = os.path.join(input_prepath, category_name, 'ADNI')
    print ("input_path is {}".format(input_path))
    folder_objs = os.walk(input_path)
    input_path,subj_folders, subj_files = folder_objs.next()
    # print ('subj_folders.shape is {}'.format(subj_folders))

    for subj_folder in subj_folders:   # different subjects
        subfolder_objs = os.walk(os.path.join(input_path, subj_folder))
        subpath,subfolders, subfiles = subfolder_objs.next()
        folder = 'MT1__GradWarp__N3m'
        if folder in subfolders:
            pass
        else:
            folder = 'MT1__N3m'

        if folder not in subfolders:
            print ("subj_id is {}".format(subj_folder))
            continue

        # print ("source path is {}".format(os.path.join(subpath, folder)))
        sub_subfolder_objs = os.walk(os.path.join(subpath, folder))
        susb_subpath, sub_subfolders,sub_subfiles = sub_subfolder_objs.next()

        for sub_subfolder in sub_subfolders:
            folder_path_src = os.path.join(subpath, folder, sub_subfolder)
            folder_path_dst = os.path.join(output_prepath, category_name, subj_folder, folder, sub_subfolder)
            copytree(folder_path_src, folder_path_dst)
            count = count +1
            print (count)
        # folder_path_src = os.path.join(subpath, folder, sub_subfolders[len(sub_subfolders)-1])
        # folder_path_dst = os.path.join(output_prepath, category_name, subj_folder, folder, sub_subfolders[len(sub_subfolders)-1])
        # os.makedirs(folder_path_dst)

        # print("src is {}".format(folder_path_src))
        # print ("dst is {}\n\n".format(folder_path_dst))

        # copytree(folder_path_src, folder_path_dst)



if __name__ == "__main__":
    print("Beginning NC...")
    move_img_folder('NC')
    print("finish NC\n")
    # print("Beginning MCI...")
    # move_img_folder('MCI')
    # print("finish MCI\n")
    # print("Beginning AD...")
    # move_img_folder('AD')
    # print("finish AD\n")
