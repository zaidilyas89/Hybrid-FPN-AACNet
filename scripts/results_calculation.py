# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 14:12:47 2023

@author: zaidi
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import yaml
import json
import numpy as np
from scipy.stats import kendalltau
    # LA Pearson Correlation (0-12)
    # LP Pearson Correlation (0-12)

    # L1A Pearson Correlation (0-3)
    # L2A Pearson Correlation (0-3)
    # L3A Pearson Correlation (0-3)
    # L4A Pearson Correlation (0-3)    

    # L1P Pearson Correlation (0-3)
    # L2P Pearson Correlation (0-3)
    # L3P Pearson Correlation (0-3)
    # L4P Pearson Correlation (0-3)

    # L1 Pearson Correlation (0-6)
    # L2 Pearson Correlation (0-6)
    # L3 Pearson Correlation (0-6)
    # L4 Pearson Correlation (0-6)
    
    # AAC Score Range Correlation (0-5)
    # AAC Score Range Correlation (6-10)
    # AAC Score Range Correlation (11-15)
    # AAC Score Range Correlation (16-20)
    # AAC Score Range Correlation (21-25)
    
    # AAC-8 Scoring Accuracy
    # AAC-24 Scoring Accuracy
FACTOR = 1e-10
folders = ['Step_1_DE_Manitoba','Step_2_DE_Manitoba']

FILE_NAME = 'id_to_labels.csv'
base_path = '../output'
data_main = {}

file2 = FILE_NAME

data2 =pd.read_csv(file2)

def confusion_mat(data_df):
    data2 = data_df.copy()
    
    # data2['pred_value']>=6 
    # preds = np.array(data_df.iloc[:,11:19].sum(1))
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

    
    # F1 = 2 * (PPV * NPV) / (PPV + NPV) Wrong
    F1 = 2 * (PPV * TPR)/(PPV + TPR + FACTOR)
    
    # # Fall out or false positive rate
    # FPR = FP/(FP+TN)
    # # False negative rate
    # FNR = FN/(TP+FN)
    # # False discovery rate
    # FDR = FP/(TP+FP)
    # # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN + FACTOR)
    
    return ACC, TP, TN, FP, FN, TPR, TNR, F1, PPV, NPV

def mean_absolute_error(actual, predicted):
    return np.mean(abs(actual - predicted))

for folder in folders:
    file_name = os.path.join(base_path,folder,'best_results.csv')
    df = pd.read_csv(file_name)
    
    data = {}
    ### Combined Wall Granular Scores Correlation
    
    gt_ant = list(df['L1A_gt'] + df['L2A_gt'] + df['L3A_gt'] + df['L4A_gt'])
    gt_post = list(df['L1P_gt'] + df['L2P_gt'] + df['L3P_gt'] + df['L4P_gt'])
    
    pred_ant = list(df['L1A_pred'] + df['L2A_pred'] + df['L3A_pred'] + df['L4A_pred'])
    pred_post = list(df['L1P_pred'] + df['L2P_pred'] + df['L3P_pred'] + df['L4P_pred'])
    
    data['correl_combined_all_4_ant_0_12'] = np.corrcoef(gt_ant,pred_ant)[0][1]
    data['correl_combined_all_4_post_0_12'] = np.corrcoef(gt_post,pred_post)[0][1]
    
    
    gt_ant_L1 = list(df['L1A_gt'])
    gt_post_L1 = list(df['L1P_gt'])
    
    pred_ant_L1 = list(df['L1A_pred'])
    pred_post_L1 = list(df['L1P_pred'])
    
    data['correl_combined_L1_ant_0_3'] = np.corrcoef(gt_ant_L1,pred_ant_L1)[0][1]
    data['correl_combined_L1_post_0_3'] = np.corrcoef(gt_post_L1,pred_post_L1)[0][1]
    
    pred_ant_L1_round = [round(x) for x in pred_ant_L1]
    pred_post_L1_round = [round(x) for x in pred_post_L1]
    
    data['cm_L1_ant_0_3'] = confusion_matrix(gt_ant_L1,pred_ant_L1_round)
    data['cm_L1_post_0_3'] = confusion_matrix(gt_post_L1,pred_post_L1_round)
    data['kendall_correlation_ant_L1_0_3'] =  kendalltau(gt_ant_L1,pred_ant_L1)[0]
    data['kendall_correlation_post_L1_0_3'] =  kendalltau(gt_post_L1,pred_post_L1)[0]
    data['mae_L1_ant_0_3'] =  mean_absolute_error(np.array(gt_ant_L1), np.array(pred_ant_L1))
    data['mae_L1_post_0_3'] =  mean_absolute_error(np.array(gt_post_L1), np.array(pred_post_L1))
    # n = 'cm_L1_ant_0_3'
    # data[n + '_ACC'], data[n + '_TP'], data[n + '_TN'], data[n + '_FP'], data[n + '_FN'], data[n + '_TPR'], data[n + '_TNR'], data[n + '_F1'], data[n + '_PPV'], data[n + '_NPV']   = accuracy_cal(data[n])
    # n = 'cm_L1_post_0_3'
    # data[n + '_ACC'], data[n + '_TP'], data[n + '_TN'], data[n + '_FP'], data[n + '_FN'], data[n + '_TPR'], data[n + '_TNR'], data[n + '_F1'], data[n + '_PPV'], data[n + '_NPV']   = accuracy_cal(data[n])
    
    gt_ant_L2 = list(df['L2A_gt'])
    gt_post_L2 = list(df['L2P_gt'])
    
    pred_ant_L2 = list(df['L2A_pred'])
    pred_post_L2 = list(df['L2P_pred'])
    
    data['correl_combined_L2_ant_0_3'] = np.corrcoef(gt_ant_L2,pred_ant_L2)[0][1]
    data['correl_combined_L2_post_0_3'] = np.corrcoef(gt_post_L2,pred_post_L2)[0][1]
    
    data['kendall_correlation_ant_L2_0_3'] =  kendalltau(gt_ant_L2,pred_ant_L2)[0]
    data['kendall_correlation_post_L2_0_3'] =  kendalltau(gt_post_L2,pred_post_L2)[0]
    
    data['mae_L2_ant_0_3'] =  mean_absolute_error(np.array(gt_ant_L2), np.array(pred_ant_L2))
    data['mae_L2_post_0_3'] =  mean_absolute_error(np.array(gt_post_L2), np.array(pred_post_L2))
    
    gt_ant_L3 = list(df['L3A_gt'])
    gt_post_L3 = list(df['L3P_gt'])
    
    pred_ant_L3 = list(df['L3A_pred'])
    pred_post_L3 = list(df['L3P_pred'])
    
    data['correl_combined_L3_ant_0_3'] = np.corrcoef(gt_ant_L3,pred_ant_L3)[0][1]
    data['correl_combined_L3_post_0_3'] = np.corrcoef(gt_post_L3,pred_post_L3)[0][1]
    data['kendall_correlation_ant_L3_0_3'] =  kendalltau(gt_ant_L3,pred_ant_L3)[0]
    data['kendall_correlation_post_L3_0_3'] =  kendalltau(gt_post_L3,pred_post_L3)[0]
    data['mae_L3_ant_0_3'] =  mean_absolute_error(np.array(gt_ant_L3), np.array(pred_ant_L3))
    data['mae_L3_post_0_3'] =  mean_absolute_error(np.array(gt_post_L3), np.array(pred_post_L3))
    
    gt_ant_L4 = list(df['L4A_gt'])
    gt_post_L4 = list(df['L4P_gt'])
    
    pred_ant_L4 = list(df['L4A_pred'])
    pred_post_L4 = list(df['L4P_pred'])
    
    data['correl_combined_L4_ant_0_3'] = np.corrcoef(gt_ant_L4,pred_ant_L4)[0][1]
    data['correl_combined_L4_post_0_3'] = np.corrcoef(gt_post_L4,pred_post_L4)[0][1]
    data['kendall_correlation_ant_L4_0_3'] =  kendalltau(gt_ant_L4,pred_ant_L4)[0]
    data['kendall_correlation_post_L4_0_3'] =  kendalltau(gt_post_L4,pred_post_L4)[0]
    data['mae_L4_ant_0_3'] =  mean_absolute_error(np.array(gt_ant_L4), np.array(pred_ant_L4))
    data['mae_L4_post_0_3'] =  mean_absolute_error(np.array(gt_post_L4), np.array(pred_post_L4))
    
    # L1 Anterior and Posterior Combined Correlation (0-6)
    # L2 Anterior and Posterior Combined Correlation (0-6)
    # L3 Anterior and Posterior Combined Correlation (0-6)
    # L4 Anterior and Posterior Combined Correlation (0-6)
    
    gt_ant_post_L1 = list(df['L1A_gt'] + df['L1P_gt'])
    pred_ant_post_L1 = list(df['L1A_pred'] + df['L1P_pred'])
    data['correl_combined_L1_ant_post_0_6'] = np.corrcoef(gt_ant_post_L1,pred_ant_post_L1)[0][1]
    data['kendall_correlation_L1_0_6'] =  kendalltau(gt_ant_post_L1,pred_ant_post_L1)[0]
    data['mae_L1_0_6'] =  mean_absolute_error(np.array(gt_ant_post_L1), np.array(pred_ant_post_L1))
    
    gt_ant_post_L2 = list(df['L2A_gt'] + df['L2P_gt'])
    pred_ant_post_L2 = list(df['L2A_pred'] + df['L2P_pred'])
    data['correl_combined_L2_ant_post_0_6'] = np.corrcoef(gt_ant_post_L2,pred_ant_post_L2)[0][1]
    data['kendall_correlation_L2_0_6'] =  kendalltau(gt_ant_post_L2,pred_ant_post_L2)[0]
    data['mae_L2_0_6'] =  mean_absolute_error(np.array(gt_ant_post_L2), np.array(pred_ant_post_L2))
    
    gt_ant_post_L3 = list(df['L3A_gt'] + df['L3P_gt'])
    pred_ant_post_L3 = list(df['L3A_pred'] + df['L3P_pred'])
    data['correl_combined_L3_ant_post_0_6'] = np.corrcoef(gt_ant_post_L3,pred_ant_post_L3)[0][1]
    data['kendall_correlation_L3_0_6'] =  kendalltau(gt_ant_post_L3,pred_ant_post_L3)[0]
    data['mae_L3_0_6'] =  mean_absolute_error(np.array(gt_ant_post_L3), np.array(pred_ant_post_L3))
    
    
    gt_ant_post_L4 = list(df['L4A_gt'] + df['L4P_gt'])
    pred_ant_post_L4 = list(df['L4A_pred'] + df['L4P_pred'])
    data['correl_combined_L4_ant_post_0_6'] = np.corrcoef(gt_ant_post_L4,pred_ant_post_L4)[0][1]
    data['kendall_correlation_L4_0_6'] =  kendalltau(gt_ant_post_L4,pred_ant_post_L4)[0]
    data['mae_L4_0_6'] =  mean_absolute_error(np.array(gt_ant_post_L4), np.array(pred_ant_post_L4))
    
    data1 = {}
    data1['correl_df0_5'] = df[df['val_gt']<=5]
    data1['correl_df6_10']= df[(df['val_gt']>5) & (df['val_gt']<=10)]
    data1['correl_df11_15'] = df[(df['val_gt']>10) &(df['val_gt']<=15)]
    data1['correl_df16_20'] = df[(df['val_gt']>15) &(df['val_gt']<=20)]
    data1['correl_df21_24'] = df[(df['val_gt']>20) &(df['val_gt']<=24)]    
    

    for name in list(data1):
        data[name] = np.corrcoef(data1[name]['val_gt'],data1[name]['val_pred'])[0][1]
    
    
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
    
    dict1, cm = confusion_mat(df)
    ACC, TP, TN, FP, FN, TPR, TNR, F1, PPV, NPV = accuracy_cal(cm)
    data['CM'] = cm
    data['ACC'] = ACC
    data['ACC_mean'] = ACC.mean()
    data['TP'] = TP
    data['TP_mean'] = TP.mean()
    data['TN'] = TN
    data['TN_mean'] = TN.mean()
    data['FP'] = FP
    data['FP_mean'] = FP.mean()
    data['FN'] = FN
    data['FN_mean'] = FN.mean()
    data['TPR'] = TPR
    data['TPR_mean'] = TPR.mean()
    data['TNR'] = TNR
    data['TNR_mean'] = TNR.mean()
    data['F1'] = F1
    data['F1_mean'] = F1.mean()
    data['PPV'] = PPV
    data['PPV_mean'] = PPV.mean()
    data['NPV'] = NPV
    data['NPV_mean'] = NPV.mean()
    dict_aac_8 = {}
    dict_aac_8['gt_ant_L1_bool'] = np.array(gt_ant_L1, dtype=bool)
    dict_aac_8['gt_ant_L2_bool'] = np.array(gt_ant_L2, dtype=bool)
    dict_aac_8['gt_ant_L3_bool'] = np.array(gt_ant_L3, dtype=bool)
    dict_aac_8['gt_ant_L4_bool'] = np.array(gt_ant_L4, dtype=bool)
 
    dict_aac_8['pred_ant_L1_bool'] = np.array(np.round(pred_ant_L1), dtype=bool)
    dict_aac_8['pred_ant_L2_bool'] = np.array(np.round(pred_ant_L2), dtype=bool)
    dict_aac_8['pred_ant_L3_bool'] = np.array(np.round(pred_ant_L3), dtype=bool)
    dict_aac_8['pred_ant_L4_bool'] = np.array(np.round(pred_ant_L4), dtype=bool)

    dict_aac_8['gt_post_L1_bool'] = np.array(gt_post_L1, dtype=bool)
    dict_aac_8['gt_post_L2_bool'] = np.array(gt_post_L2, dtype=bool)
    dict_aac_8['gt_post_L3_bool'] = np.array(gt_post_L3, dtype=bool)
    dict_aac_8['gt_post_L4_bool'] = np.array(gt_post_L4, dtype=bool)
 
    dict_aac_8['pred_post_L1_bool'] = np.array(np.round(pred_post_L1), dtype=bool)
    dict_aac_8['pred_post_L2_bool'] = np.array(np.round(pred_post_L2), dtype=bool)
    dict_aac_8['pred_post_L3_bool'] = np.array(np.round(pred_post_L3), dtype=bool)
    dict_aac_8['pred_post_L4_bool'] = np.array(np.round(pred_post_L4), dtype=bool)
    
    names = ['L1A', 'L1P', 'L2A', 'L2P', 'L3A', 'L3P', 'L4A', 'L4P']
    files = [('gt_ant_L1_bool', 'pred_ant_L1_bool'),
             ('gt_ant_L2_bool', 'pred_ant_L2_bool'),
             ('gt_ant_L3_bool', 'pred_ant_L3_bool'),
             ('gt_ant_L4_bool', 'pred_ant_L4_bool'),
             ('gt_post_L1_bool', 'pred_post_L1_bool'),
             ('gt_post_L2_bool', 'pred_post_L2_bool'),
             ('gt_post_L3_bool', 'pred_post_L3_bool'),
             ('gt_post_L4_bool', 'pred_post_L4_bool'),]
    for i in range(0,8):
        fname = 'AAC_8_' + names[i]
    
        data[fname + '_cm'] = confusion_matrix(dict_aac_8[files[i][0]], dict_aac_8[files[i][1]])
        data[fname + '_ACC'], data[fname + '_TP'], data[fname + '_TN'], data[fname + '_FP'], data[fname + '_FN'], data[fname + '_TPR'], data[fname + '_TNR'], data[fname + '_F1'], data[fname + '_PPV'], data[fname + '_NPV'] = accuracy_cal(data[fname + '_cm'])
        
        data[fname + '_ACC_mean'] = data[fname + '_ACC'].mean()
        data[fname + '_TP_mean'] = data[fname + '_TP'].mean()
        data[fname + '_TN_mean'] = data[fname + '_TN'].mean()
        data[fname + '_FP_mean'] = data[fname + '_FP'].mean()
        data[fname + '_FN_mean'] = data[fname + '_FN'].mean()
        data[fname + '_TPR_mean'] = data[fname + '_TPR'].mean()
        data[fname + '_TNR_mean'] = data[fname + '_TNR'].mean()
        data[fname + '_F1_mean'] = data[fname + '_F1'].mean()
        data[fname + '_PPV_mean'] = data[fname + '_PPV'].mean()        
        data[fname + '_NPV_mean'] = data[fname + '_NPV'].mean() 
        
    dict_aac_24 = {}
    dict_aac_24['gt_ant_L1'] = np.array(gt_ant_L1)
    dict_aac_24['gt_ant_L2'] = np.array(gt_ant_L2)
    dict_aac_24['gt_ant_L3'] = np.array(gt_ant_L3)
    dict_aac_24['gt_ant_L4'] = np.array(gt_ant_L4)
 
    dict_aac_24['pred_ant_L1'] = np.array(np.round(pred_ant_L1))
    dict_aac_24['pred_ant_L2'] = np.array(np.round(pred_ant_L2))
    dict_aac_24['pred_ant_L3'] = np.array(np.round(pred_ant_L3))
    dict_aac_24['pred_ant_L4'] = np.array(np.round(pred_ant_L4))

    dict_aac_24['gt_post_L1'] = np.array(gt_post_L1)
    dict_aac_24['gt_post_L2'] = np.array(gt_post_L2)
    dict_aac_24['gt_post_L3'] = np.array(gt_post_L3)
    dict_aac_24['gt_post_L4'] = np.array(gt_post_L4)
 
    dict_aac_24['pred_post_L1'] = np.array(np.round(pred_post_L1))
    dict_aac_24['pred_post_L2'] = np.array(np.round(pred_post_L2))
    dict_aac_24['pred_post_L3'] = np.array(np.round(pred_post_L3))
    dict_aac_24['pred_post_L4'] = np.array(np.round(pred_post_L4))
    
    
    
    names = ['L1A', 'L1P', 'L2A', 'L2P', 'L3A', 'L3P', 'L4A', 'L4P']
    files = [('gt_ant_L1', 'pred_ant_L1'),
             ('gt_ant_L2', 'pred_ant_L2'),
             ('gt_ant_L3', 'pred_ant_L3'),
             ('gt_ant_L4', 'pred_ant_L4'),
             ('gt_post_L1', 'pred_post_L1'),
             ('gt_post_L2', 'pred_post_L2'),
             ('gt_post_L3', 'pred_post_L3'),
             ('gt_post_L4', 'pred_post_L4'),]
    
    
    for i in range(0,8):
        fname = 'AAC_24_' + names[i]
    
        data[fname + '_cm'] = confusion_matrix(dict_aac_24[files[i][0]], dict_aac_24[files[i][1]])
        data[fname + '_ACC'], data[fname + '_TP'], data[fname + '_TN'], data[fname + '_FP'], data[fname + '_FN'], data[fname + '_TPR'], data[fname + '_TNR'], data[fname + '_F1'], data[fname + '_PPV'], data[fname + '_NPV']  = accuracy_cal(data[fname + '_cm'])
        
        data[fname + '_ACC_mean'] = data[fname + '_ACC'].mean()
        data[fname + '_TP_mean'] = data[fname + '_TP'].mean()
        data[fname + '_TN_mean'] = data[fname + '_TN'].mean()
        data[fname + '_FP_mean'] = data[fname + '_FP'].mean()
        data[fname + '_FN_mean'] = data[fname + '_FN'].mean()
        data[fname + '_TPR_mean'] = data[fname + '_TPR'].mean()
        data[fname + '_TNR_mean'] = data[fname + '_TNR'].mean()    
        data[fname + '_F1_mean'] = data[fname + '_F1'].mean()
        data[fname + '_PPV_mean'] = data[fname + '_PPV'].mean()        
        data[fname + '_NPV_mean'] = data[fname + '_NPV'].mean() 
        
    # file=open(folder + ".yaml","w")
    # yaml.dump(data,file)
    # file.close()
    # print("YAML file saved.")
    
    data['correl_0_24'] = np.corrcoef(df['val_gt'],df['val_pred'])[0][1]
    data['kendalltau_0_24'] = kendalltau(df['val_gt'],df['val_pred'])[0]
    
    def convert_np_arrays(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    json_data = json.dumps(data, default=convert_np_arrays, indent=2)
    
    # Specify the file path
    file_path = folder + '_results.txt'
    
    # Write the JSON-formatted string to a text file
    with open(file_path, 'w') as file:
        file.write(json_data)
    
    print(f"Data has been saved to {file_path}")
    
    data_main[folder] = data
    
        


# Sample data



# Calculate Kendall correlation coefficient
# tau, p_value = kendalltau(x, y)

# print("Kendall correlation coefficient:", tau)
# print("p-value:", p_value)
    
    
    
# df1 = pd.DataFrame.from_dict(data_main,orient='index')
# df2 = pd.DataFrame.from_dict(data_main)
# df1.to_csv('analysis_results.csv')
# df2.to_csv('analysis_results2.csv')

