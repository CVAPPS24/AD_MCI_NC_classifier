import os, pdb, time
import deepdish as dd
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

def load_model(in_dir, model_name, weight_name):
    model_path = os.path.join(in_dir, model_name)
    json_file = open(model_path, 'r')
    model_json = json_file.read(json_file)
    json_file.close()
    trained_model = model_from_json(model_json)
    weight_path = os.path.join(in_dir, weight_name)
    trained_model.load_weights(weight_path)
    return trained_model

def generate_probs(category_path, folder, img_size, batch_size, model):
    vald_dir = vald_dir = os.path.join(category_path, folder)
    datagen = ImageDataGenerator(preprocessing_function = preprocessing)
    vald_generator = datagen.flow_from_directory(vald_dir, target_size=(img_size, img_size),
                    batch_size=batch_size, class_mode='categorical', shuffle = False)
    vald_slice_filenames = vald_generator.filenames
    vald_num = len(vald_slice_filenames)
    vald_slice_probs = model.predict_generator(vald_generator, steps=int(math.ceil(1.0*vald_num/batch_size)), verbose = 1)
    # pdb.set_trace()
    print('Writing filenames...')
    vald_filenames_path = os.path.join(category_path, 'output', method_folder, folder+'_prob_filenames_googLeNet.h5')
    dd.io.save(vald_filenames_path, vald_slice_filenames)
    print('Finished!!!')
    print('Writing features...')
    tb = time.time()
    vald_feat_path = os.path.join(category_path, 'output', method_folder, folder+'_feat_googLeNet.h5')
    dd.io.save(vald_feat_path, vald_slice_feats)
    te = time.time()
    print('Writing finished {}s'.format(int(te-tb)))
    return vald_slice_probs, vald_slice_filenames

def majority_voting_patient_classify(slice_probs, filenames, ):
    #slice_record
    TP, TN, FP, FN, PP, PN = .0, .0, .0, .0, .0, .0
    PPV = 0.0  # Positive predictive value (precision) = sum(TP)/sum(PP)
    FOR = 0.0  # False Omission Rate (FOR) = sum(FN)/sum(FN)
    FDR = 0.0  # False Discover Rate (FDR) = sum(FP)/sum(PP)
    NPV = 0.0  # Negative Predictive Value (NPV) = sum(TN)/sum(PN)
    #patient_record
    pTP, pTN, pFP, pFN, pPP, pPN = .0, .0, .0, .0, .0, .0
    pPPV = 0.0  # Positive predictive value (precision) = sum(TP)/sum(PP)
    pFOR = 0.0  # False Omission Rate (FOR) = sum(FN)/sum(FN)
    pFDR = 0.0  # False Discover Rate (FDR) = sum(FP)/sum(PP)
    pNPV = 0.0  # Negative Predictive Value (NPV) = sum(TN)/sum(PN)
    patient_pred_prob = {}
    patient_label_gt = {}
    # print('Patient classify...')
    slice_probs_np = np.array(slice_probs)

    ## Get the true and predicted labels of subjects
    for i, filename in enumerate(filenames):
        ind = filename.find('_', -8, -1)
        key_filename = filename[:ind]

        value_label = slice_probs_np[i][np.newaxis,:]
        # print('subj_label_predict_prob is {}'.format(subj_label_predict_prob.keys()))
        if key_filename in patient_pred_prob.keys(): # calculate the probability of the same subjects
            assert int(filename[0]) == patient_label_gt[key_filename]
            patient_pred_prob[key_filename] = np.concatenate((patient_pred_prob[key_filename], value_label), axis = 0)  # a patient has a prob array
        else:
            patient_label_gt[key_filename] = int(filename[0])
            patient_pred_prob[key_filename] = value_label

    ## validate the accuracy of slices, and calculate the accuracy of patients
    slice_acc = .0
    patient_acc = .0
    print ('patient num is {}'.format(len(patient_pred_prob.keys())))
    for pid in patient_pred_prob.keys():
        true_label = patient_label_gt[pid]
        probs = patient_pred_prob[pid]
        labels = np.argmax(probs, axis = 1)
        true_slice = np.sum(true_label == labels)
        slice_acc += true_slice
        patient_label = np.argmax(np.mean(probs, axis = 0))
        patient_acc += np.sum(true_label == patient_label)
        # count slice confusion matrix
        PP += np.sum(labels == 0)
        PN += np.sum(labels == 1)
        if true_label == 0:
            TP += true_slice
            FN += (probs.shape[0]-true_slice)
        else:
            TN += true_slice
            FP += (probs.shape[0]-true_slice)
        # count patient confusion matrix
        pPP += (patient_label == 0)
        pPN += (patient_label == 1)
        if true_label == 0:
            if true_label == patient_label:
                pTP += 1
            else:
                pFN += 1
        else:
            if true_label == patient_label:
                pTN += 1
            else:
                pFP += 1

    PPV = 1.0*TP/PP
    FOR = 1.0*FN/PN
    FDR = 1.0*FP/PP
    NPV = 1.0*TN/PN
    pPPV = 1.0*pTP/pPP
    pFOR = 1.0*pFN/pPN
    pFDR = 1.0*pFP/pPP
    pNPV = 1.0*pTN/pPN
    patient_num = len(patient_label_gt)
    slice_num = len(filenames)
    slice_acc /= len(filenames)
    patient_acc /= len(patient_label_gt)

    print("Slice_acc is {}   Confusion matrix:\n{}\t{}\n{}\t{}".format(slice_acc, PPV, FOR, FDR, NPV))
    print("Patient_acc is {}   Confusion matrix:\n{}\t{}\n{}\t{}".format(patient_acc, pPPV, pFOR, pFDR, pNPV))

def read_file(in_dir, category_name, filenames, probs_name):
    file_path= os.path.join(in_dir, category_name+filenames)
    filenames = dd.io.load(file_path)
    prob_path = os.path.join(in_dir, category_nameprobs_name)
    probs = dd.io.load(prob_path)
    return filenames, probs

if __name__ == '__main__':
    root_dir = '/data/data2/Linlin/GoogLeNet_input/'
    batch_size = 16
    img_size = 299
    category_folder = 'NCvsAD'
    method_folder = 'baseline'
    model_name = 'model_googLeNet.json'
    weight_name = 'weights_googLeNet.h5 '

    # load model
    category_path = os.path.join(root_dir, category_folder)
    in_dir = os.path.join(category_path, 'output', method_folder)
    trained_model = load_model(in_dir, model_name, weight_name)

    folder = 'vald'
    print ('==================={}==============='.format(folder))
    probs, filenames = generate_probs(category_path, folder, img_size, batch_size, trained_model)
    majority_voting_patient_classify(probs, filenames, )

    folder = 'test'
    print ('==================={}==============='.format(folder))
    probs, filenames = generate_probs(category_path, folder, img_size, batch_size, trained_model)
    majority_voting_patient_classify(probs, filenames, )
