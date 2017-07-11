

def generate_vald_test_subjs(img_prepath, in_folder_name, out_folder_name, folder_name, test_pct, vald_pct, slct_num):

    print("Generating training and test files...")
    in_path = os.path.join(img_prepath, in_folder_name, folder_name)
    subj_folders_obj = os.wakl(in_path)
    subj_folders = subj_folders_obj.next()
    subj_folders_num = len(subj_folders)

    vald_subj_num = int(subj_folders_num*vald_pct)
    vald_num = rdm.sample(list(xrange(subj_folders_num)), vald_subj_num))
    # Move validation data
    for i in vald_num:
        vald_foldername = subj_folders[i]
        vald_path_src = os.path.join(in_path, vald_foldername)
        vald_path_dst = os.path.join(img_prepath, out_folder_name, 'vald', folder_name)
        print("src is {}\ndst is {}".format(vald_path_src, vald_path_dst))
        pdb.set_trace()
        shutil.move(vald_path_src, vald_path_dst)
    print('vald finished.')
    # Copy training data manually
    subj_folders_obj = os.wakl(in_path)
    subj_folders = subj_folders_obj.next()
    subj_folders_num = len(subj_folders)
    test_subj_num = int(subj_folders_num*test_pct)
    test_num = rdm.sample(list(xrange(subj_folders_num)), int(test_subj_num))
    # Move validation data
    for i in test_num:
        test_foldername = subj_folders[i]
        test_path_src = os.path.join(in_path, test_foldername)
        vald_path_dst = os.path.join(img_prepath, out_folder_name, 'test', folder_name)
        print("src is {}\ndst is {}".format(vald_path_src, vald_path_dst))
        pdb.set_trace()
        shutil.move(test_path_src, test_path_dst)
    print("test finished.")
    # dd.io.save(os.path.join(prepath, folder_file, folder_name+'_test_filenames.h5'), test_filenames)
    print("Finish {}...".format(folder_name))

if __name__ == '__main__':

    options = {
        'test_pct':0.2
        'vald_pct': 0.3
        'slct_num': 50
        'in_folder_name': 'Select_slices'
        'out_folder_name': 'Train_vald_test_subj'
        'img_prepath': '/data/Linlin/GoogLeNet_inputs'
    }

    folder_name = 'AD'
    generate_vald_test_subjs(options['img_prepath'], options['in_folder_name'], options['out_folder_name'],...
    folder_name, options['test_pct'], options['vald_pct'], options['slct_num'])
