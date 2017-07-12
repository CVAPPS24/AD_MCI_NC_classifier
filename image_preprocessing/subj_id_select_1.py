import os
import numpy as np
import pandas as pd
import pdb;
from shutil import copytree

def read_csv():
    ## read the table of Diagnostic_summary.csv
    filepath_sum = os.path.join(os.path.dirname(__file__),'input_doc/DXSUM_PDXCONV_ADNIALL-Diagnostic Summary.csv')
    doc_sum = pd.read_csv(filepath_sum)
    # read the table of ARM.csv
    filepath_arm = os.path.join(os.path.dirname(__file__),'input_doc/ARM.csv')
    doc_arm = pd.read_csv(filepath_arm)
    ## obtain subject ID based Roster.csv
    filepath_rid_subj = os.path.join(os.path.dirname(__file__),'input_doc/Roster.csv')
    doc_rid_subj = pd.read_csv(filepath_rid_subj)
    rid_subj = doc_rid_subj[["RID","PTID"]]
    # print("There are {} rows and column in doc_sum.".format(doc_sum.shape))
    # print("There are {} rows and column in doc_arm.".format(doc_arm.shape[0]))
    # print ("There are {} subjects in tatal".format(rid_subj.shape[0]))

    return doc_sum, doc_arm,doc_rid_subj

def table_join(doc_sum, doc_arm, doc_rid_subj, dxch, arm):

    rid_sum = doc_sum["RID"][doc_sum.DXCHANGE == dxch]
    rid_arm = doc_arm['RID'][doc_arm.ARM==arm]
    # print ("\n\n\nrid_sum is \n{}".format(rid_sum))
    # print ("rid_arm.shape is {}, rid_arm.shape is {}".format(rid_sum.shape,rid_arm.shape))

    ## assign name to the column
    rid_sum = pd.DataFrame(rid_sum, columns=['RID'])
    rid_arm = pd.DataFrame(rid_arm, columns=['RID'])
    rid_ind_same = rid_sum['RID'].isin(rid_arm['RID'])  # isin is to obtain the intersection of two sets
    rid_same = rid_sum[rid_ind_same]
    # print (rid_same)
    subj_ind = []
    subj_ind = doc_rid_subj["RID"].isin(rid_same["RID"])
    subj = doc_rid_subj["PTID"][subj_ind]
    subj = pd.DataFrame(subj, columns=['PTID'])
    # print (subj)

    # remove duplicates
    # print(subj.shape)
    # print (subj['PTID'])
    subj_without_dupl = subj.drop_duplicates('PTID')
    # subj_without_dupl.drop(subj_without_dupl.index[0])
    # print (subj_without_dupl)
    subj_row = subj_without_dupl.transpose()
    # print ("subj_row is {}".format(subj_row))
    return subj_row

def write_result(filename, prepath, subj_without_dupl):
    if not os.path.exists(prepath):
        os.makedirs(prepath)
    filepath = os.path.join(prepath,filename)
    subj_without_dupl.to_csv(filepath, sep=',', header = False, index = False)

def img_categories(prepath_src, prepath_dst, folder_name_src, folder_name_dst, subj_NC):
    no_exist = []
    for subj in subj_NC.values:
        # print ("subj is {}".format(subj))
        subj = "".join(subj)
        path_src = os.path.join(prepath_src, subj, folder_name_src)
        path_dst = os.path.join(prepath_dst, folder_name_dst, subj, folder_name_src)

        if os.path.exists(path_src):
            copytree(path_src, path_dst)
        else:
            # print ("{} does not exist".format(path_src))
            no_exist.append(subj)

    no_exist_dataframe = pd.DataFrame(no_exist)
    path_writing = os.path.join(os.path.dirname(__file__),'output_doc', folder_name_dst+'_missing.csv')
    no_exist_dataframe.to_csv(path_writing)

if __name__ == '__main__':
    # in doc_sum,  DXCHANGE=1  Stable: NL to NL;  2  Stable: MCI to MCI;
    #                       3  Stable: Dementia to Dementia;
    dxch = [1, 2, 3]
    dxch_arr = np.array(dxch, dtype = np.int32)
    # in doc_arm,  ARM=1 NL - 1.5T only; 2=MCI - 1.5T only; 3=AD - 1.5T only
    # 7=NL - 3T+1.5T;  8=MCI - 3T+1.5T;  9=AD - 3T+1.5T;    10=EMCI
    arm = [1, 2, 3, 7, 8, 9, 10]
    arm_arr = np.array(arm, dtype = np.int32)
    # file name of each category
    name_nc = 'NC.csv'
    name_emci = 'EMCI.csv'
    name_mci = 'MCI.csv'
    name_ad = 'AD.csv'

    prepath = os.path.join(os.path.dirname(__file__), 'ouput_doc')

    doc_sum, doc_arm, doc_rid_subj = read_csv()   # read context from files
    # pdb.set_trace()
    '''select AD'''
    # subj_AD = table_join(doc_sum, doc_arm, doc_rid_subj, dxch[2], arm[5])
    subj_AD = table_join(doc_sum, doc_arm, doc_rid_subj, dxch[2], arm[2]) # select required context
    print("There are {} unique subjects in AD".format(subj_AD.shape))
    write_result(name_ad, prepath, subj_AD)  # write results into a file

    '''select NC'''
    subj_NC = table_join(doc_sum, doc_arm, doc_rid_subj, dxch[0], arm[0]) # select required context
    print("There are {} unique subjects in NC".format(subj_NC.shape))
    write_result(name_nc, prepath, subj_NC)  # write results into a file
    print ("subj_NC is {}".format(subj_NC))

    ''' select MCI '''
    subj_MCI = table_join(doc_sum, doc_arm, doc_rid_subj, dxch[1], arm[1]) # select required context
    print("There are {} unique subjects in MCI".format(subj_MCI.shape))
    write_result(name_mci, prepath, subj_MCI)  # write results into a file
    print ("subj_MCI is {}".format(subj_MCI))


    # Copy folders
    # prepath_src = '/media/linlin/LinlinGao-BME360/ADNI'
    # prepath_dst = '/media/linlin/LinlinGao-BME360/ADNI_category'
    # folder_name_src = 'MPR-R__GradWarp__B1_Correction__N3'
    # img_categories(prepath_src, prepath_dst, folder_name_src, 'NC', subj_NC)
    # print ("NC finished...\n")
    # img_categories(prepath_src, prepath_dst, folder_name_src, 'MCI', subj_MCI)
    # print ("MCI finished...\n")
    # img_categories(prepath_src, prepath_dst, folder_name_src, 'AD', subj_AD)
    # print ("AD finished...\n")
