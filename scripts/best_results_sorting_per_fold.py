# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 10:26:14 2023

@author: zaidi
"""

import torch
import os
import pandas as pd
import numpy as np
import glob
import fnmatch
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description="Best Results per fold and combination of results.")

parser.add_argument(
    '--file',
    type=str,
    default='./id_to_labels.csv',
    help='The source file with relevant IDs.'
)

arg = parser.parse_args()

def confusion_mat(data_df):
    data2 = data_df.copy()
    
    # data2['pred_value']>=6 
    
    preds = np.array(data2['val_pred'], dtype = np.double)
    # preds = preds.round()
    preds2 = preds.copy()
    preds2[preds>=6] = 2
    preds2[preds<2] = 0
    preds2[np.logical_and(preds>=2,preds<6)] = 1
    
    gts = np.array(data2['val_gt'], dtype = np.double)
    gts2 = gts.copy()
    gts2[gts>=6] = 2
    gts2[gts<2] = 0
    gts2[np.logical_and(gts>=2,gts<6)] = 1
    
    # preds2[preds>=6] = 2
    # preds2[preds<2] = 1
    # preds2[np.logical_and(preds>=2,preds<6)] = 0
    
    # gts = np.array(data2['val_gt'], dtype = np.double)
    # gts2 = gts.copy()
    # gts2[gts>=6] = 2
    # gts2[gts<2] = 1
    # gts2[np.logical_and(gts>=2,gts<6)] = 0
    
    
    trues = gts2 == preds2
    
    # print(trues.sum()/len(trues))
    
    dict1 = {'name':data2['val_id'],
             'pred': preds2,
            'gt':gts2,
            'pred_reg':preds,
            'gt_reg':gts}
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(dict1['gt'], dict1['pred'])
    return dict1, cm



def accuracy_cal(cnf_matrix):
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN + FACTOR)
    # Specificity or true negative rate
    TNR = TN/(TN+FP + FACTOR) 
    
    
    # Precision or positive predictive value
    PPV = TP/(TP+FP + FACTOR)
    # Negative predictive value
    NPV = TN/(TN+FN + FACTOR)

    # # Fall out or false positive rate
    # FPR = FP/(FP+TN)
    # # False negative rate
    # FNR = FN/(TP+FN)
    # # False discovery rate
    # FDR = FP/(TP+FP)
    # # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN + FACTOR)
    
    return ACC, TP, TN, FP, FN, TPR, TNR



FACTOR = 1e-10
file2 = arg.file
# file2 = 'id_to_labels.csv'
data2 =pd.read_csv(file2)
fold_names = ['*Fold_{}*'+'.csv'.format(i) for i in range(0,10)]
# fold_name = 'Fold_0.csv'

fold_names_all = glob.glob(os.path.join('*.csv'))

pattern = '*Fold_*'

# Select files that match the pattern
fold_names = fnmatch.filter(fold_names_all, pattern)

for count, fold_name in tqdm(enumerate(fold_names)):
    
    
    file = pd.read_csv(fold_name)
    FACTOR = len(set(file['val_id']))
    di = {'val_id_check':[]}
    # ids = [data2[data2['image_id'] == file['val_id'][i]]['image_id'].item() for i in tqdm(range(0,len(file)))]
    for i in tqdm(range(0,len(file))):
        # di['val_gt'].append((data2[data2['image_id'] == file['val_id'][i].split('.npy')[0]]['labels']).item())
        di['val_id_check'].append(file['val_id'][i])
                     
    
    file1 = pd.concat([file,pd.DataFrame(di)],1)
    
    chunks = len(file1)/FACTOR
    
    
    
    
    
    data = []
    for i in range(0,int(chunks)):
        start = i*FACTOR
        end = i*FACTOR + FACTOR
        
        data.append(file1.iloc[start:end])
    
    
    data_results = {'CM':[],
                    'ACC':[],
                    'ACC_mean':[],
                    'TP':[],
                    'TP_mean':[],
                    'TN':[],
                    'TN_mean':[],
                    'FP':[],
                    'FP_mean':[],
                    'FN':[],
                    'FN_mean':[],
                    'TPR':[],
                    'TNR':[]}
    
    for i in range(0,len(data)):
        assert len(set(data[i]['val_id'])) == FACTOR
    
    for i in range(0,len(data)):
        dict1, cm = confusion_mat(data[i])
        ACC, TP, TN, FP, FN, TPR, TNR = accuracy_cal(cm)
        data_results['CM'].append(cm)
        data_results['ACC'].append(ACC)
        data_results['ACC_mean'].append(ACC.mean())
        data_results['TP'].append(TP)
        data_results['TP_mean'].append(TP.mean())
        data_results['TN'].append(TN)
        data_results['TN_mean'].append(TN.mean())
        data_results['FP'].append(FP)
        data_results['FP_mean'].append(FP.mean())
        data_results['FN'].append(FN)
        data_results['FN_mean'].append(FN.mean())
        data_results['TPR'].append(TPR)
        data_results['TNR'].append(TNR)
        
        
    aa = np.array(data_results['TP_mean']) + np.array(data_results['TN_mean']) - np.array(data_results['FP_mean']) - np.array(data_results['FN_mean'])
    index = aa.argmax()
    value = aa.max()  
    
    if count == 0:
        bet_data = data[index]
    elif count>0:
        bet_data = pd.concat([bet_data,data[index]],axis=0, ignore_index=True)

dict1, cm = confusion_mat(bet_data)
ACC, TP, TN, FP, FN, TPR, TNR = accuracy_cal(cm)
from sklearn import metrics
# confusion_matrix = metrics.confusion_matrix(actual, predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Low', 'Medium', 'High'])
import matplotlib.pyplot as plt
plt.figure()

# Plot your confusion matrix
cm_display.plot()

# Save the figure to a file
plt.savefig('confusion_matrix.png')
bet_data.to_csv('best_results.csv')


a = bet_data['L1A_gt'] + bet_data['L1P_gt']
b = bet_data['L2A_gt'] + bet_data['L2P_gt']
c = bet_data['L3A_gt'] + bet_data['L3P_gt']
d = bet_data['L4A_gt'] + bet_data['L4P_gt']

a1 = bet_data['L1A_pred'] + bet_data['L1P_pred']
b1 = bet_data['L2A_pred'] + bet_data['L2P_pred']
c1 = bet_data['L3A_pred'] + bet_data['L3P_pred']
d1 = bet_data['L4A_pred'] + bet_data['L4P_pred']


dict1 = {'L1_gt':a,
         'L2_gt':b,
         'L3_gt':c,
         'L4_gt':d,
         'L1_pred':a1,
         'L2_pred':b1,
         'L3_pred':c1,
         'L4_pred':d1,
                  }

best_data = pd.concat([bet_data,pd.DataFrame(dict1)],axis=1)

bet_data.to_csv('best_results.csv')


# import pandas as pd
# import os
# file = r'D:\zaidi\Downloads'

# fn = os.path.join(file,'best_results.csv')
# fn_afsah = os.path.join(file,'Final_Manitoba_Results.csv')

# df=pd.read_csv(fn)
# names = df['val_id']
# l_names = list(names)
# n1 = set(list(names))

# df2=pd.read_csv(fn_afsah)
# names2 = df2['PatientID']
# l_names2 = list(names2)
# n2 = set(list(names2))


# p = [i.split('_')[-1] for i in l_names]
# p2 = [i.split('_')[-1] for i in l_names2]

# path1 = r'D:\AAC_Networks_Redesign\Regression_Manitoba\output\trained_models\Fold_0.csv'
# path2 = r'D:\AAC_Networks_Redesign\Regression_Manitoba\output\trained_models\Fold_1.csv'

# f0_df = pd.read_csv(path1)
# f1_df = pd.read_csv(path2)

# print(len(set(f0_df['val_id'])))
# print(len(set(f1_df['val_id'])))

# a = set(f0_df['val_id'])
# b = set(f1_df['val_id'])

# import torch.nn as nn
# input = torch.randn(10, 4)
# weight = torch.randn(2,4)
 
# torch.mm(input,weight.t())
# layer=nn.Linear(in_features=4,out_features=2,bias=False)
