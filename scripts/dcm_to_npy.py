# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:03:20 2023

@author: zaidi
"""

import os
from tqdm import tqdm
from glob import glob
import numpy as np
from mat4py import loadmat
import pydicom as dicom
import argparse


# Create the parser
parser = argparse.ArgumentParser(description="DICOM to npy format conversion")

# Add an argument for the folder path
parser.add_argument('--src_folder_path', 
                    type=str, nargs='?',  
                    default = '../data/Model/dcm',  
                    help='Enter Source Folder.')
parser.add_argument('--dst_folder_path', 
                    type=str, nargs='?',  
                    default = '../data/Model/npy',  
                    help='Enter Destination Folder.')
parser.add_argument('--perc_crop_from_top', 
                    type=float, nargs='?',  
                    default = 0.50,  
                    help='Enter crop percentage factor.')
parser.add_argument('--normalization_max_value', 
                    type=float, nargs='?',  
                    default = 4096,  
                    help='Enter Max Value for Normalization.')
parser.add_argument('--normalization_min_value', 
                    type=float, nargs='?',  
                    default = 0,  
                    help='Enter Min Value for Normalization.')

# Parse the arguments
args = parser.parse_args()

# Use the folder path argument
src_folder = args.src_folder_path
dst_folder = args.dst_folder_path
perc_crop_from_top = args.perc_crop_from_top

 
# Check if the folder exists
if not os.path.exists(dst_folder):
    # Create the folder if it doesn't exist
    os.makedirs(dst_folder)
    print(f"Folder created: {dst_folder}")
else:
    print(f"Folder already exists: {dst_folder}")

files = glob(os.path.join(src_folder,'*.dcm'))



def crop_image(img, test=False):
    '''
    Works on numpy images
    '''
    y,x = img.shape

    # cropx=int(x*0.60) # for training anf VFA x
    
    # if test:
    #     cropx=int(x*0.60)
        

    # startx = int(x*0.10)
    # startx = int(x*0.10)
    starty = int(y*perc_crop_from_top)  # for training and VFA int(y*0.50)
    return img[starty:y,:]

for file in tqdm(files):
    try:
        ds = dicom.dcmread(file)
        image_2d1 = np.array(ds.pixel_array)
        file_name = file.split('\\')[-1].split('.dcm')[0]
        
        # image_2d1 = np.array(loadmat(file_path)['BMD'])
        # shape = image_2d1.shape
    
        # image_2d1=np.fliplr(image_2d1)
        
        image_2d1 = (image_2d1-args.normalization_min_value)/(args.normalization_max_value-args.normalization_min_value)
        img=crop_image(image_2d1, None)
        np.save(os.path.join(dst_folder,file_name+'.npy'),image_2d1)
        np.save(os.path.join(dst_folder,file_name+'.npy'),img)
    except:
        print('error')
        print('fname:'+file_name)
        pass
    
