# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:33:03 2023

@author: zaidi
"""
import torch
import os
from utilities.datasets import us_326_model_182_combined_npy_cropped_reg_and_gran
from utilities.efficientNet_FPN.efficientnet_drsa_fusion_encoder_with_classification_head import model_DRSA # Last three hierarchies with DRSA and EFFM (Stable Results)
# from utilities.efficientNet_FPN.efficientnet_drsa_fusion_encoder_all_hierarchies_with_classification_head import model_DRSA # All hierarchies with DRSA and EFFM 
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import copy 
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import scipy
from utilities.utils import define_transforms, r2_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset, Sampler, BatchSampler, SequentialSampler
from glob import glob
from utilities.utils_reg_and_gran import train_model
import logging
opt = {}

seed = 3
torch.manual_seed(seed)

opt['data_path'] = './data/US_MODEL_Combined/SE/npy_cropped'
opt['file_path'] = './data/US_MODEL_Combined/id_to_labels_us_model_combined.csv'
opt['file_path_granular'] = './data/US_MODEL_Combined/ids_to_score-us-model-combined-fine.npy'
# opt['model_path'] = './output/EB3_HiLo_EFF_all_hierarchies'
opt['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
opt['k'] = 10
opt['shape'] = (320, 320)
opt['mean'] = 0.5
opt['std'] = 0.5
opt['batch_size'] = 16
opt['epochs'] = 50
opt['lr'] = 5e-4
opt['beta_1'] = 0.9
opt['beta_2'] = 0.999
opt['weight_decay'] = .0
opt['loss_option'] = 'MSE'
opt['model_name'] = 'Step_1_SE_US_Model_Combined'
opt['patience'] = 50
opt['dst_folder'] = './outputs'
# model_path = './adacon_efficient-netb3_mse_89.2222acc_2th-fold_lr-0.00010_sen-91.67_spec-92.424_corr-0.894_epoch-99.00_pv-0.00_rho-0.87_prho-20.00-bill-all-de.pt'
# opt['model_path'] = './utilities/caifos_efficient-netb3_mse_70.3125acc_1th-fold_lr-0.00050_sen-87.50_spec-89.375_corr-0.856_epoch-78.00_pv-0.00_rho-0.81_prho-0.00.pt'
################### Dataset Testing ###################
'''
dataset = caifos_dataset_dcm_using_csv(data_path=data_path, file_path=file_path, transforms=val_transforms)

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16,16))

for i in range(4):


    image,fname,label =dataset[i]
    
    image = np.squeeze(image)
    # print(image.min(axis=0))
    # print(image.max(axis=0).max(axis=0))
    # print(image.shape)
    ax[i].imshow(image[0],cmap='gray')
    ax[i].axis('off')
'''
dst_folder = os.path.join(opt['dst_folder'],opt['model_name'])

if not os.path.exists(dst_folder):
    # Create the folder if it doesn't exist
    os.makedirs(dst_folder)
    print(f"Folder created: {dst_folder}")
else:
    print(f"Folder already exists: {dst_folder}")


name = opt['model_name'] + ".log"  
logging.basicConfig(filename=os.path.join(dst_folder,name),
                    format='%(asctime)s %(message)s',
                    filemode='w')
# Creating an object
logger = logging.getLogger()
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)
################### k folds ################### 

if __name__ ==  '__main__':
    df = pd.read_csv(opt['file_path'])
    splitter = StratifiedKFold(n_splits=opt['k'], shuffle=True, random_state=0)
    
    splits = []
    for train_idx, val_idx in splitter.split(df['image_id'], df['labels']):
        splits.append((train_idx, val_idx))
    
    ################### Transforms ###################
    train_transforms, val_transforms = define_transforms(opt)
    ################### Datasets and Dataloaders ###################
    

    train_dataset = us_326_model_182_combined_npy_cropped_reg_and_gran(data_path=opt['data_path'], file_path=opt['file_path'], file_path_granular=opt['file_path_granular'], transforms=train_transforms)
    val_dataset = us_326_model_182_combined_npy_cropped_reg_and_gran(data_path=opt['data_path'], file_path=opt['file_path'], file_path_granular=opt['file_path_granular'], transforms=val_transforms)    
    # d = train_dataset[1]
    # e = val_dataset[0]
    
    # x = val_dataset[0]
        
    
    
    acc_list=[]
    sen_list=[]
    spec_list=[]
    acc_all_list=[]
        
    for curr_fold, (train_split, valid_split) in enumerate(splits):  
        # if curr_fold < 2:
        #     continue
        print(curr_fold)
        train_sampler = SubsetRandomSampler(train_split)
        val_sampler = SubsetRandomSampler(valid_split)
        train_loader = DataLoader(train_dataset, batch_size=opt['batch_size'], sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=opt['batch_size'], sampler=val_sampler)
        ################### Model ###################
        model = model_DRSA().to(opt['device'])
        # name_ = glob(opt['model_path']+"/**fold_"+str(curr_fold)+"**")
        # model.load_state_dict(torch.load(name_[-1]), strict =True)
        # model.load_state_dict(torch.load(opt['model_path']), strict =True)
        # model_name = './AACLiteNet_MSE_83.7827acc_2th-fold_lr-0.00005_sen-83.33_spec-91.212_corr-0.857_epoch-48.00_pv-0.00_rho-0.86_prho-16.00-bill-all-de.pt'
        # model.load_state_dict(torch.load(model_name), strict =True)
    
        # for name, param in model.named_parameters():
        #     param.requires_grad = False
        #     print('Name: ', name,  'Requires_Grad:', param.requires_grad)        
        # for name, param in model.linear_layers.named_parameters():
        #     param.requires_grad = True
        #     print('Name: ', name,  'Requires_Grad:', param.requires_grad)
        # for name, param in model.linear_layers2.named_parameters():
        #     param.requires_grad = True
        #     print('Name: ', name,  'Requires_Grad:', param.requires_grad)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'], betas=(opt['beta_1'], opt['beta_2']), weight_decay=opt['weight_decay'])
        
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        criterion_L1A = nn.CrossEntropyLoss(reduction='mean').to(opt['device'])
        criterion_L1P = nn.CrossEntropyLoss(reduction='mean').to(opt['device'])
        
        criterion_L2A = nn.CrossEntropyLoss(reduction='mean').to(opt['device'])
        criterion_L2P = nn.CrossEntropyLoss(reduction='mean').to(opt['device'])
        
        criterion_L3A = nn.CrossEntropyLoss(reduction='mean').to(opt['device'])
        criterion_L3P = nn.CrossEntropyLoss(reduction='mean').to(opt['device'])
        
        criterion_L4A = nn.CrossEntropyLoss(reduction='mean').to(opt['device'])
        criterion_L4P = nn.CrossEntropyLoss(reduction='mean').to(opt['device'])
            
        criterion_aac_score = torch.nn.MSELoss(reduction='mean').to(opt['device'])
        
        trained_model, data = train_model(model,
                                          criterion_aac_score, criterion_L1A, criterion_L1P, criterion_L2A, criterion_L2P, criterion_L3A, criterion_L3P, criterion_L4A, criterion_L4P,
                                          optimizer,
                                          scheduler,
                                          train_loader,
                                          val_loader,
                                          opt['device'],
                                          opt['loss_option'],
                                          logger,
                                          curr_fold,
                                          num_epochs=opt['epochs'],
                                          patience = opt['patience'],
                                          )
        
        # model.load_state_dict(torch.load(model_path), strict=True)
        
        #save_history("%s_%.4facc_%dth_fold_lr-%.5f_beta1-%.2f_beta2-%.3f.csv"%(model_name, curr_best, curr_fold, lr, betas[0], betas[1]), curr_history)
        torch.save(trained_model.state_dict(),os.path.join(dst_folder, "Fold_%s_%s_%s_%.4facc_%dth-fold_lr-%.5f_sen-%.2f_spec-%.3f_corr-%.3f_epoch-%.2f_pv-%.2f_rho-%.2f_prho-%.2f.pt"%(str(curr_fold), opt['model_name'],opt['loss_option'], data['best_acc'], 
                                                                                                                                                                    curr_fold, opt['lr'], data['bsen'][2], data['bspec'][2],
                                                                                                                                                                    data['bcorr'],data['epoch'], data['bp'],data['brho'], 
                                                                                                                                                                    opt['batch_size'])))
        
        outs = {'val_id': [i for j in data['details']['val_id']
                           for i in j], 'val_pred': [i*24 for i in data['details']['val_pred']],
                                            'val_gt':[i*24 for i in data['details']['val_gt']]}                    
        
        # outs = {'val_id': [i for j in details['val_id']
        #                    for i in j][-len(test_idx):], 'val_pred': [i*24 for i in details['val_pred']][-len(test_idx):]}
        
        df1 = pd.DataFrame(outs, columns=['val_id', 'val_pred','val_gt'])
        
        val_pred_grans = np.array(data['details']['val_pred_gran'])
        val_gt_grans = np.array(data['details']['val_gt_gran'])
        
        dicta = {'L1A_gt':val_gt_grans[:,0],
                 'L1P_gt':val_gt_grans[:,1],
                 'L2A_gt':val_gt_grans[:,2],
                 'L2P_gt':val_gt_grans[:,3],
                 'L3A_gt':val_gt_grans[:,4],
                 'L3P_gt':val_gt_grans[:,5],
                 'L4A_gt':val_gt_grans[:,6],
                 'L4P_gt':val_gt_grans[:,7],
                 'L1A_pred':val_pred_grans[:,0],
                 'L1P_pred':val_pred_grans[:,1],
                 'L2A_pred':val_pred_grans[:,2],
                 'L2P_pred':val_pred_grans[:,3],
                 'L3A_pred':val_pred_grans[:,4],
                 'L3P_pred':val_pred_grans[:,5],
                 'L4A_pred':val_pred_grans[:,6],
                 'L4P_pred':val_pred_grans[:,7],}
        
        df2 = pd.DataFrame(dicta)
        df = pd.concat([df1,df2],1)
        df.to_csv(os.path.join(dst_folder, 'Fold_'+str(curr_fold)+'_'+ opt['model_name']+'.csv'),index=False, header=True)
        
        # df.to_csv('Fold_'+str(curr_fold)+'.csv',index=False, header=True)
    
    

