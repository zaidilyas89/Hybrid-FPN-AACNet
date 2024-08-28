# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:33:03 2023

@author: zaidi
"""
import torch
import os
from utilities.datasets import manitoba_1916_npy_full_sized, manitoba_1916_npy_cropped,manitoba_1916_npy_cropped_reg_and_gran_predict
# from utilities.efficientNet_FPN.efficientnet_hilo_fusion_encoder_with_reg_head_last_3_hierarchies import model_Hilo
from utilities.efficientNet_FPN.efficientnet_drsa_fusion_encoder_with_classification_head import model_DRSA
# from utilities.model_reg_and_granular import model_reg_and_gran
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
from utilities.utils_reg_and_gran import predict_reg
import argparse

parser = argparse.ArgumentParser(description="Prediction for new images using Hybrid_AAC_FPN.")

# Add the arguments
parser.add_argument(
    '-s', '--source',
    type=str,
    default='./data/test',
    help='The source folder with data or images for prediction.'
)

parser.add_argument(
    '-d', '--destination',
    type=str,
    default= './outputs/predictions/Manitoba',
    help='The destination folder. Default is a folder named "output" in the current working directory.'
)

parser.add_argument(
    '-f', '--fold_number',
    type=int,
    default = None,
    help='Which fold number model to use for prediction. Default value is None which means all folds will be used.'
)

parser.add_argument(
    '-m', '--models_path',
    type=str,
    default = './outputs/Step_2_DE_Manitoba',
    help='Models path for prediction'
)


# Parse the arguments
args = parser.parse_args()


opt = {}

# seed = 3
# torch.manual_seed(seed)
# opt['data_path'] = './data//Manitoba/SE/npy_cropped'
opt['data_path'] = args.source
opt['predictions'] = args.destination
# opt['file_path'] ='./id_to_labels.csv'
opt['models'] = args.models_path 


# opt['model_path'] = './pretrained_models/AACLiteNet_MSE_81.1818acc_4th-fold_lr-0.00050_sen-80.56_spec-90.968_corr-0.858_epoch-99.00_pv-0.00_rho-0.78_prho-20.00-bill-all-de.pt'
opt['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
opt['k'] = 10
opt['shape'] = (320, 320)
opt['mean'] = 0.5
opt['std'] = 0.5
opt['batch_size'] = 40
# model_path = './adacon_efficient-netb3_mse_89.2222acc_2th-fold_lr-0.00010_sen-91.67_spec-92.424_corr-0.894_epoch-99.00_pv-0.00_rho-0.87_prho-20.00-bill-all-de.pt'
# opt['model_path'] = './utilities/caifos_efficient-netb3_mse_70.3125acc_1th-fold_lr-0.00050_sen-87.50_spec-89.375_corr-0.856_epoch-78.00_pv-0.00_rho-0.81_prho-0.00.pt'
################### Dataset Testing ###################


################### k folds ################### 

if __name__ ==  '__main__':
    
    

    if not os.path.exists(opt['predictions']):
        # Create the folder if it doesn't exist
        os.makedirs(opt['predictions'])
        print(f"Folder created: {opt['predictions']}")
    else:
        print(f"Folder already exists: {opt['predictions']}")
    
    
    model_files = glob(os.path.join(opt['models'],'*.pt'))
    # model_files[2].split('\\')[-1][26:34]
    # df = pd.read_csv(opt['file_path'])
    # splitter = StratifiedKFold(n_splits=opt['k'], shuffle=True, random_state=0)
    
    # splits = []
    # for train_idx, val_idx in splitter.split(df['image_id'], df['labels']):
    #     splits.append((train_idx, val_idx))
    
    ################### Transforms ###################
    _, val_transforms = define_transforms(opt)
    ################### Datasets and Dataloaders ###################
    
    # train_dataset = manitoba_1916_npy_cropped_reg_and_gran(data_path=opt['data_path'], file_path=opt['file_path'], transforms=train_transforms)
    val_dataset = manitoba_1916_npy_cropped_reg_and_gran_predict(data_path=opt['data_path'], transforms=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=opt['batch_size'], num_workers=1)

        
    
    
    acc_list=[]
    sen_list=[]
    spec_list=[]
    acc_all_list=[]
        
    for curr_model, model_file in enumerate(model_files):   
        
        ################### Model ###################
        # model = model_reg_and_gran().to(opt['device'])
        model = model_DRSA().to(opt['device'])
        model.linear_layers2 = nn.Identity()
        model.linear_layers = nn.Sequential(
                nn.BatchNorm1d(256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Linear(64, 9),
                
            ).to(opt['device'])  
        model.load_state_dict(torch.load(model_file), strict =True)

        df = predict_reg(model, val_loader, opt['device'])
        
        name = model_file.split('\\')[-1][0:25]
        
        df.to_csv(os.path.join(args.destination,name+'.csv'),index=False, header=True)
    
    

