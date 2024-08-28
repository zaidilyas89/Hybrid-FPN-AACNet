# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 14:12:47 2023

@author: zaidi
"""

import os
import numpy as np
import pandas as pd



# folders = ['AACLiteNet_Manitoba_and_SE_andUS', 'AACLiteNet_Manitoba_Baseline_SE', 'AACLiteNet_Manitoba_DE_US', 'AACLiteNet_Manitooba_US_SE_Model',
#            'AACLiteNet_Reg_and_Gran_Baseline','AAC_LiteNet_MROS_BASE_US','AACLiteNet_MROS_base_Model','EfficientNetB4_20_folds','AACLiteNet_Baseline']

folders = ['Step_1_DE_Manitoba','Step_2_DE_Manitoba']


base_path = './outputs'
data_main = {}
for folder in folders:
    file_name = os.path.join(base_path,folder,'best_results.csv')
    df = pd.read_csv(file_name)
    
    data = {}
    ### Combined Wall Granular Scores Correlation
    
    gt_ant = list(df['L1A_gt'] + df['L2A_gt'] + df['L3A_gt'] + df['L4A_gt'])
    gt_post = list(df['L1P_gt'] + df['L2P_gt'] + df['L3P_gt'] + df['L4P_gt'])
    
    pred_ant = list(df['L1A_pred'] + df['L2A_pred'] + df['L3A_pred'] + df['L4A_pred'])
    pred_post = list(df['L1P_pred'] + df['L2P_pred'] + df['L3P_pred'] + df['L4P_pred'])
    
    data['combined_all_4_ant_0_12'] = np.corrcoef(gt_ant,pred_ant)[0][1]
    data['combined_all_4_post_0_12'] = np.corrcoef(gt_post,pred_post)[0][1]
    
    # L1 Anterior Correlation (0-3) = 0.52
    # L2 Anterior Correlation (0-3) = 0.54
    # L3 Anterior Correlation (0-3) = 0.67
    # L4 Anterior Correlation (0-3) = 0.69
    
    # L1 Posterior Correlation (0-3) = 0.63
    # L2 Posterior Correlation (0-3) = 0.65
    # L3 Posterior Correlation (0-3) = 0.69
    # L4 Posterior Correlation (0-3) = 0.73
    
    gt_ant_L1 = list(df['L1A_gt'])
    gt_post_L1 = list(df['L1P_gt'])
    
    pred_ant_L1 = list(df['L1A_pred'])
    pred_post_L1 = list(df['L1P_pred'])
    
    data['combined_L1_ant_0_3'] = np.corrcoef(gt_ant_L1,pred_ant_L1)[0][1]
    data['combined_L1_post_0_3'] = np.corrcoef(gt_post_L1,pred_post_L1)[0][1]
    
    gt_ant_L2 = list(df['L2A_gt'])
    gt_post_L2 = list(df['L2P_gt'])
    
    pred_ant_L2 = list(df['L2A_pred'])
    pred_post_L2 = list(df['L2P_pred'])
    
    data['combined_L2_ant_0_3'] = np.corrcoef(gt_ant_L2,pred_ant_L2)[0][1]
    data['combined_L2_post_0_3'] = np.corrcoef(gt_post_L2,pred_post_L2)[0][1]
    
    gt_ant_L3 = list(df['L3A_gt'])
    gt_post_L3 = list(df['L3P_gt'])
    
    pred_ant_L3 = list(df['L3A_pred'])
    pred_post_L3 = list(df['L3P_pred'])
    
    data['combined_L3_ant_0_3'] = np.corrcoef(gt_ant_L3,pred_ant_L3)[0][1]
    data['combined_L3_post_0_3'] = np.corrcoef(gt_post_L3,pred_post_L3)[0][1]
    
    gt_ant_L4 = list(df['L4A_gt'])
    gt_post_L4 = list(df['L4P_gt'])
    
    pred_ant_L4 = list(df['L4A_pred'])
    pred_post_L4 = list(df['L4P_pred'])
    
    data['combined_L4_ant_0_3'] = np.corrcoef(gt_ant_L4,pred_ant_L4)[0][1]
    data['combined_L4_post_0_3'] = np.corrcoef(gt_post_L4,pred_post_L4)[0][1]
    
    # L1 Anterior and Posterior Combined Correlation (0-6)
    # L2 Anterior and Posterior Combined Correlation (0-6)
    # L3 Anterior and Posterior Combined Correlation (0-6)
    # L4 Anterior and Posterior Combined Correlation (0-6)
    
    gt_ant_post_L1 = list(df['L1A_gt'] + df['L1P_gt'])
    pred_ant_post_L1 = list(df['L1A_pred'] + df['L1P_gt'])
    data['combined_L1_ant_post_0_6'] = np.corrcoef(gt_ant_post_L1,pred_ant_post_L1)[0][1]
    
    gt_ant_post_L2 = list(df['L2A_gt'] + df['L2P_gt'])
    pred_ant_post_L2 = list(df['L2A_pred'] + df['L2P_gt'])
    data['combined_L2_ant_post_0_6'] = np.corrcoef(gt_ant_post_L2,pred_ant_post_L2)[0][1]
    
    gt_ant_post_L3 = list(df['L3A_gt'] + df['L3P_gt'])
    pred_ant_post_L3 = list(df['L3A_pred'] + df['L3P_gt'])
    data['combined_L3_ant_post_0_6'] = np.corrcoef(gt_ant_post_L3,pred_ant_post_L3)[0][1]
    
    gt_ant_post_L4 = list(df['L4A_gt'] + df['L4P_gt'])
    pred_ant_post_L4 = list(df['L4A_pred'] + df['L4P_gt'])
    data['combined_L4_ant_post_0_6'] = np.corrcoef(gt_ant_post_L4,pred_ant_post_L4)[0][1]
    
    data_main[folder] = data
    
df1 = pd.DataFrame.from_dict(data_main,orient='index')
df2 = pd.DataFrame.from_dict(data_main)
df1.to_csv('analysis_results.csv')
df2.to_csv('analysis_results2.csv')
