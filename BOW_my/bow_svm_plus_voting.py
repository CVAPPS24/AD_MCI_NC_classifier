import os, time, pdb, math
import numpy as np
from numpy import linalg as LA
import deepdish as dd
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn import svm


def normalize_feat(x):
    '''
        X: n*p, where n is # of samples, p is feat dim
        Here, only L2 is used to normalize feat,
        other normalized method should also be considered
    '''
    norm = LA.norm(x, ord = 2, axis = 1)
    return x / norm[:, np.newaxis]

def generate_kmeans_model(category_folder_name):
    # 1. read files about slices
    folder = 'train'
    filename_path = os.path.join(os.path.dirname(__file__), 'output', category_folder_name, folder+'_filenames_googLeNet.h5')
    feat_path = os.path.join(os.path.dirname(__file__), 'output', category_folder_name, folder+'_feat_googLeNet.h5')
    filenames = dd.io.load(filename_path)
    feats = dd.io.load(feat_path)
    # do L2 normalization
    feats = normalize_feat(feats)

    # 2. slices' bow features
    print('Begining Slice BOW...')
    # clustering features
    kmeans_model = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, n_jobs=-1)
    kmeans_model.fit(feats)

    return kmeans_model

def patient_timepoint_bow_feats(folder, category_folder_name, kmeans_model):
    '''
        1. read files    2. produce slice bow features
        3. produce patient bow feature
    '''
    # 1. read files about slices
    filename_path = os.path.join(os.path.dirname(__file__), 'output', category_folder_name, folder+'_filenames_googLeNet.h5')
    feat_path = os.path.join(os.path.dirname(__file__), 'output', category_folder_name, folder+'_feat_googLeNet.h5')
    filenames = dd.io.load(filename_path)
    feats = dd.io.load(feat_path)
    # do L2 normalization
    feats = normalize_feat(feats)

    # 2. slices' bow features
    # predict the label that X belongs to which index of centers
    slices_bow_feats = kmeans_model.predict(feats)

    # 3. produce patient bow feature based on their time sequences
    patients_bow = {}
    # patients_bow = {pid1: {timepoint1: feat1, timepoint2: feat2,...}
    #                 pid2: {timepoint1: feat1, timepoint2: feat2,...}}
    patients_label_gt = {}
    ## Get the true and predicted labels of subjects
    for i, filename in enumerate(filenames):
        # print('{} slice'.format(filename))
        if filename[2:12] == 'ADNI_bet-B':
            continue
        patient_name = filename[7:17]
        filename_strs = filename.split('_')
        timepoint_name = filename_strs[-4]
        bow_feat = slices_bow_feats[i]
        if patient_name in patients_bow.keys():   # patient_name exists or not
            timepoints_dict = patients_bow[patient_name]  # get the timepoints_dict
            if timepoint_name in timepoints_dict.keys():  # timepoint exists or not
                timepoints_dict[timepoint_name][bow_feat] += 1.0
            else:    # timepoint_name does not exit
                timepoint_bow = np.zeros(num_clusters)
                timepoint_bow[bow_feat] += 1.0
                timepoints_dict[timepoint_name] = timepoint_bow
        else:  # patient_name does not exist
            patients_label_gt[patient_name] = int(filename[0])
            timepoint_bow = np.zeros(num_clusters)
            timepoint_bow[bow_feat] += 1.0
            timepoint_bow_dict = {}
            timepoint_bow_dict[timepoint_name] = timepoint_bow
            patients_bow[patient_name] = timepoint_bow_dict

    patient_bow_dict = {}
    '''
        patient_bow_dict={pid1: list(timepoints_bow), pid2: list(timepoints_bow),...}
    '''
    for pid in patients_bow.keys():
        timepoint_bow_list = patients_bow[pid].values()
        patient_bow_dict[pid] = timepoint_bow_list
    # patient_bow_list = normalize_feat(patient_bow_list)   #normalizing do great help to results

    return patient_bow_dict, patients_label_gt

def train_svm_model(patients_bow_feats, patient_label_gt):
    '''
        1. get all the timepoint feats and the corresponding labels
        2. train svm_model
    '''
    # Step 1
    timepoints_bow_feats = []
    timepoints_labels = []
    for i, pid in enumerate(patients_bow_feats):
        timepoint_bow_feats = patients_bow_feats[pid]
        timepoint_bow_feats = np.array(timepoint_bow_feats)
        timepoint_labels = np.ones((len(timepoint_bow_feats), ))*patient_label_gt[pid]
        if i == 0:
            timepoints_bow_feats = timepoint_bow_feats
            timepoints_labels = timepoint_labels
        else:
            timepoints_bow_feats = np.append(timepoints_bow_feats, timepoint_bow_feats, axis = 0)
            timepoints_labels = np.append(timepoints_labels, timepoint_labels, axis = 0)
        # pdb.set_trace()
    print('timepoints_bow_feats.shape {}\ntimepoints_labels.shape {}'.format(timepoints_bow_feats.shape, timepoints_labels.shape))

    est = svm.SVC()
    tuned_parameters = [{'kernel': ['rbf', 'linear'], 'C': [0.001, 0.01, 1, 10]}]
    svm_model = GridSearchCV(est, tuned_parameters, cv=5)  #grid search
    svm_model.fit(timepoints_bow_feats, timepoints_labels)

    return svm_model

def evaluate_classifier(svm_model, patients_bow_feats, patients_label_gt):
    '''
        1. get all the timepoint feats and the corresponding labels
        2. train svm_model
    '''
    TP, TN, FP, FN, PP, PN = .0, .0, .0, .0, .0, .0
    PPV = 0.0  # Positive predictive value (precision) = sum(TP)/sum(PP)
    FOR = 0.0  # False Omission Rate (FOR) = sum(FN)/sum(FN)
    FDR = 0.0  # False Discover Rate (FDR) = sum(FP)/sum(PP)
    NPV = 0.0  # Negative Predictive Value (NPV) = sum(TN)/sum(PN)
    for pid in patients_bow_feats.keys():
        patient_label_gt = patients_label_gt[pid]
        patient_bow_feats = patients_bow_feats[pid]
        patient_labels_pred = svm_model.predict(patient_bow_feats)  # the labels of all the timepoint of a patient
        num_timepoints_0 = np.sum(patient_labels_pred == 0)  # the # of predicted positive timepoints
        if num_timepoints_0 > (len(patient_bow_feats) - num_timepoints_0):
            patient_label_pred = 0
        else:
            patient_label_pred = 1

        if patient_label_pred == 0:
            PP += 1.0
            if patient_label_gt == patient_label_pred:
                TP += 1.0
            else:
                FP += 1.0
        else:
            PN += 1.0
            if patient_label_gt == patient_label_pred:
                TN += 1.0
            else:
                FN += 1.0
    PPV = 1.0*TP/PP
    FOR = 1.0*FN/PN
    FDR = 1.0*FP/PP
    NPV = 1.0*TN/PN
    patient_acc = 1.0*(TP+TN)/len(patients_bow_feats)

    print('Patient_acc {}'.format(patient_acc))
    print('Confusion matrix:\n{}  {}  \n{}  {}'.format(PPV, FOR, FDR, NPV))
    # pdb.set_trace()
    num_patients_0 = np.sum(np.array(patients_label_gt.values()) == 0)
    num_patients_1 = np.sum(np.array(patients_label_gt.values()) == 1)
    print('GT: {} patients {} MCI_patients {} NC_patients'.format(len(patients_label_gt), num_patients_0, num_patients_1))

    return patient_acc

def main_funct(num_clusters, category_folder_name):
    '''
        1. generate kmeans_model  2. generate SVM model based on patients' bow feats of train data
        3. evaluate svm_model based on the bow feats of train, vald and test data
    '''

    print '-------------Train and evaluate SVM model-------------'
    # 1. generate kmeans_model
    kmeans_model = generate_kmeans_model(category_folder_name)
    # 2. extract patients' BOW feats of train data and train svm model
    folder = 'train'
    patients_bow_feats, patients_label_gt = patient_timepoint_bow_feats(folder, category_folder_name, kmeans_model)
    # 3. evaluate SVM model based on the bow feats of training data
    # pdb.set_trace()
    svm_model = train_svm_model(patients_bow_feats, patients_label_gt)
    train_score = evaluate_classifier(svm_model, patients_bow_feats, patients_label_gt)
    print '-------------Evaluate Test Data-------------'
    folder = 'test'
    patients_bow_feats, patient_label_gt = patient_timepoint_bow_feats(folder, category_folder_name, kmeans_model)
    test_score = evaluate_classifier(svm_model, patients_bow_feats, patient_label_gt)
    print '-------------Evaluate Vald Data-------------'
    folder = 'vald'
    patients_bow_feats, patient_label_gt = patient_timepoint_bow_feats(folder, category_folder_name, kmeans_model)
    vald_score = evaluate_classifier(svm_model, patients_bow_feats, patient_label_gt)

    return train_score, test_score, vald_score

if __name__ == '__main__':
    num_clusters = 32
    category_folder_name = 'NCvsMCI'
    suffix = '_normslice'
    folder_name = category_folder_name+suffix   # filename

    # score_max = []  # classification acc under diff 'num_clusters'
    # for num_clusters in xrange(3, , 1):
    print('================= {} Num_Cluster {} ================='.format(category_folder_name, num_clusters))
    train_score, test_score, vald_score = main_funct(num_clusters, folder_name)
    # score_max.append([train_score, test_score, vald_score])

    # print('train, test, vald')
    # for score in score_max:
    #     print(score)
    # pdb.set_trace()
    # print('Over')
