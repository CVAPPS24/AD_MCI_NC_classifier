import os
import pdb

root_dir = ''
root_dir = '/data/data2/Linlin/Codes/ADNIProcess/image_preprocess/output_doc'
root_dir = os.path.join(os.path.dirname(__file__), 'ouput_doc')
folder ='NC'
filename = folder+'.csv'
file = open(os.path.join(root_dir, filename), 'r')
mci_patients_names_csv = file.read()
mci_patients_names_csv = mci_patients_names_csv.split(',')
print(len(mci_patients_names_csv))  # 237
root_dir = '/media/linlin/bmeyanglab/Brain_research/ADNI_images/'+folder+'/Select_Time_Folder'
mci_paients_names_folder = os.listdir(root_dir)
print('mci_patients_names_folder: {}'.format(len(mci_paients_names_folder)))  # 150

pdb.set_trace()
count = 0
for patient_folder in mci_paients_names_folder:
    if not patient_folder in mci_patients_names_csv:
        print ('{} not {}'.format(patient_folder, folder))
        count += 1
print ('{} not in {}'.format(count, folder))

# folder = '135_S_5275'
# if folder in mci_patients_names_csv:
#     print 'Y'
