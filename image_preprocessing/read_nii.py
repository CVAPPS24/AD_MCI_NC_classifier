import os
import pdb
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # root_dir = '/media/linlin/bmeyanglab/Brain_research/ADNI_axial/After_BET3'
    # categry_name = 'AD'
    # categry_path = os.path.join(root_dir, categry_name)
    #
    # root_dir = '/media/linlin/bmeyanglab/Brain_research/ADNI_axial/After_BET3/AD/002_S_5018/3_Plane_Localizer/2013-05-16_12_20_33.0/S189763'
    # filename = 'ADNI_002_S_5018_MR_3_Plane_Localizer__br_raw_20130517102118272_9_S189763_I372813._bet-B.nii.gz'

    root_dir = '/media/linlin/bmeyanglab/Brain_research/ADNI_axial/Select_Folders2/AD/002_S_5018/3_Plane_Localizer/2013-05-16_12_20_33.0/S189763'
    filename = 'ADNI_002_S_5018_MR_3_Plane_Localizer__br_raw_20130517102118272_9_S189763_I372813.nii'
    filepath = os.path.join(root_dir, filename)
    pdb.set_trace()
    nii_file = nib.load(filepath)
    nii_np = nii_file.get_data()
    print('nii_np.shape: {}'.format(nii_np.shape))

    slice_tmp = nii_np[40,...]
    plt.imshow(slice_tmp)
    plt.show()
