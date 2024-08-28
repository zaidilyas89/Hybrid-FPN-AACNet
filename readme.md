# Hybrid-FPN-AACNet (Accepted in MICCAI 2024)
A PyTorch implementation of Hybrid-FPN-AACNet for granular Abdominal Aortic Calcification (AAC) detection in lateral view DXA images.
![Network Architecture](result/architecture.png)


## Dataset
Custom dataset was used for the training of this model which has not been shared due to data ethics limitations. 
For training, testing, and prediction using Hybrid-FPN-AACNet, make sure the directory like this:
```                           
|-- data     
    |-- Manitoba
        |-- DE
            |-- mat
                SBH-01022013_144654.mat
                SBH-01022016_105551.mat
                ...  
            |-- npy_cropped
                SBH-01022013_144654.npy
                SBH-01022016_105551.npy
                ...                      
        |-- SE
            |-- mat
                SBH-01022013_144654.mat
                SBH-01022016_105551.mat
                ...                  
            |-- npy_cropped
                SBH-01022013_144654.npy
                SBH-01022016_105551.npy
                ...                                     
    |-- US_Model_Combined
        |-- SE
            |-- dcm
                PNC-1.dcm
                MODEL CK1_CK_VFA_BL_19FEB2020.dcm         
                ...                        
            |-- npy_cropped
                PNC-1.npy
                MODEL CK1_CK_VFA_BL_19FEB2020.npy
                ...                                         
```
For data conversion from .dcm format to .npy format, navigate to ```./utilities``` and use the following command:
```
python dcm_to_npy.py --src_folder_path ../data/US_Model_Combined/SE/dcm --dst_folder_path ../data/US_Model_Combined/SE/npy_cropped --perc_crop_from_top 0.50 --normalization_max_value 4096 --normalization_min_value 0 
```
The images from US and Model datasets are cropped 50% from top and are normalized with '0' and '4096' as minimum and maximum values.

For data conversion from .mat format to .npy format, navigate to ```./utilities``` and use the following commands:
```
python mat_to_npy.py --src_folder_path ../data/Manitoba/SE/mat --dst_folder_path ../data/Manitoba/SE/npy_cropped --perc_crop_from_top 0.50 --normalization_max_value 32767 --normalization_min_value -32767 --modality SE
python mat_to_npy.py --src_folder_path ../data/Manitoba/DE/mat --dst_folder_path ../data/Manitoba/DE/npy_cropped --perc_crop_from_top 0.50 --normalization_max_value 32767 --normalization_min_value -32767 --modality DE
```
The images from Manitoba datasets are cropped 50% from top and are normalized with '32767' and '-32767' as minimum and maximum values.

## Train Model
For Manitoba SE dataset training using 10 fold stratified cross validation approach, run the following commands sequentially:
```
python Step_1_Pretraining_Classification_Manitoba_SE.py 

python Step_2_Training_Regression_Manitoba_SE.py
```

For Manitoba DE dataset training using 10 fold stratified cross validation approach, run the following commands sequentially:
```
python Step_1_Pretraining_Classification_Manitoba_DE.py 

python Step_2_Training_Regression_Manitoba_DE.py
```

For US-Model combined SE dataset training using 10 fold stratified cross validation approach, run the following commands sequentially:
```
python Step_1_Pretraining_Classification_us_model_combined_SE.py 

python Step_2_Training_Regression_us_model_combined_SE.py
```

The trained models and the corresponding results for each fold would be saved in the following folders:
```
|-- outputs     
    |-- Step_1_DE_Manitoba
        |-- Fold_0_Step_1_DE_Manitoba.csv
        |-- Fold_1_Step_1_DE_Manitoba.csv
        ...
        |-- Fold_0_Step_1_DE_Manitoba_MSE_xxxacc_0th-fold_lr-xxx_sen-xxx_spec-xxx_corr-xxx_epoch-xxx_pv-xxx_rho-xxx_prho-xxx.pt
        |-- Fold_1_Step_1_DE_Manitoba_MSE_xxxacc_1th-fold_lr-xxx_sen-xxx_spec-xxx_corr-xxx_epoch-xxx_pv-xxx_rho-xxx_prho-xxx.pt
        ...
    |-- Step_2_DE_Manitoba
        |-- Fold_0_Step_2_DE_Manitoba.csv
        |-- Fold_1_Step_2_DE_Manitoba.csv
        ...
        |-- Fold_0_Step_2_DE_Manitoba_MSE_xxxacc_0th-fold_lr-xxx_sen-xxx_spec-xxx_corr-xxx_epoch-xxx_pv-xxx_rho-xxx_prho-xxx.pt
        |-- Fold_1_Step_2_DE_Manitoba_MSE_xxxacc_1th-fold_lr-xxx_sen-xxx_spec-xxx_corr-xxx_epoch-xxx_pv-xxx_rho-xxx_prho-xxx.pt
        ...
    |-- Step_1_SE_Manitoba
        |-- Fold_0_Step_1_SE_Manitoba.csv
        |-- Fold_1_Step_1_SE_Manitoba.csv
        ...
        |-- Fold_0_Step_1_SE_Manitoba_MSE_xxxacc_0th-fold_lr-xxx_sen-xxx_spec-xxx_corr-xxx_epoch-xxx_pv-xxx_rho-xxx_prho-xxx.pt
        |-- Fold_1_Step_1_SE_Manitoba_MSE_xxxacc_1th-fold_lr-xxx_sen-xxx_spec-xxx_corr-xxx_epoch-xxx_pv-xxx_rho-xxx_prho-xxx.pt
        ...
    |-- Step_2_SE_Manitoba
        |-- Fold_0_Step_1_SE_US_Model_Combined.csv
        |-- Fold_1_Step_1_SE_US_Model_Combined.csv
        ...
        |-- Fold_0_Step_1_SE_US_Model_Combined_MSE_xxxacc_0th-fold_lr-xxx_sen-xxx_spec-xxx_corr-xxx_epoch-xxx_pv-xxx_rho-xxx_prho-xxx.pt
        |-- Fold_1_Step_1_SE_US_Model_Combined_MSE_xxxacc_1th-fold_lr-xxx_sen-xxx_spec-xxx_corr-xxx_epoch-xxx_pv-xxx_rho-xxx_prho-xxx.pt
        ...
    |-- Step_1_SE_US_Model_Combined   
        |-- Fold_0_Step_2_SE_US_Model_Combined.csv
        |-- Fold_1_Step_2_SE_US_Model_Combined.csv
        ...
        |-- Fold_0_Step_1_SE_US_Model_Combined_MSE_xxxacc_0th-fold_lr-xxx_sen-xxx_spec-xxx_corr-xxx_epoch-xxx_pv-xxx_rho-xxx_prho-xxx.pt
        |-- Fold_1_Step_1_SE_US_Model_Combined_MSE_xxxacc_1th-fold_lr-xxx_sen-xxx_spec-xxx_corr-xxx_epoch-xxx_pv-xxx_rho-xxx_prho-xxx.pt
        ...
    |-- Step_2_SE_US_Model_Combined 
        |-- Fold_0_Step_1_SE_US_Model_Combined.csv
        |-- Fold_1_Step_1_SE_US_Model_Combined.csv
        ...
        |-- Fold_0_Step_2_SE_US_Model_Combined_MSE_xxxacc_0th-fold_lr-xxx_sen-xxx_spec-xxx_corr-xxx_epoch-xxx_pv-xxx_rho-xxx_prho-xxx.pt
        |-- Fold_1_Step_2_SE_US_Model_Combined_MSE_xxxacc_1th-fold_lr-xxx_sen-xxx_spec-xxx_corr-xxx_epoch-xxx_pv-xxx_rho-xxx_prho-xxx.pt
        ...
```
## Combined Result Calculation from k-folds
Copy the corresponding file ```id_to_labels.csv``` to the output folder. For example, in case of Manitoba DE, copy the mentioned files in ```./outputs/Step_2_DE_Manitoba``` folder
and then copy the file ```best_results_sorting_per_fold.py``` in same folder. It would look something like the following:
```
|-- outputs     
    |-- Step_2_DE_Manitoba
        |-- id_to_labels.csv
        |-- best_results_sorting_per_fold.py
        |-- Fold_0_Step_2_DE_Manitoba.csv
        |-- Fold_1_Step_2_DE_Manitoba.csv
        ...
        |-- Fold_0_Step_2_DE_Manitoba_MSE_xxxacc_0th-fold_lr-xxx_sen-xxx_spec-xxx_corr-xxx_epoch-xxx_pv-xxx_rho-xxx_prho-xxx.pt
        |-- Fold_1_Step_2_DE_Manitoba_MSE_xxxacc_1th-fold_lr-xxx_sen-xxx_spec-xxx_corr-xxx_epoch-xxx_pv-xxx_rho-xxx_prho-xxx.pt
        ...
```
Run the following command:
```
python best_results_sorting_per_fold.py --file id_to_labels.csv
```
Results would be saved in the form of a single ```.csv``` file i.e. ```best_results.csv``` and as a confusion matrix in the form of an image file as shown below:
```
|-- outputs     
    |-- Step_2_DE_Manitoba
        |-- id_to_labels.csv
        |-- best_results_sorting_per_fold.py
        |-- Fold_0_Step_2_DE_Manitoba.csv
        |-- Fold_1_Step_2_DE_Manitoba.csv
        |-- best_results.csv
        |-- confusion_matrix.png
        ...
        |-- Fold_0_Step_2_DE_Manitoba_MSE_xxxacc_0th-fold_lr-xxx_sen-xxx_spec-xxx_corr-xxx_epoch-xxx_pv-xxx_rho-xxx_prho-xxx.pt
        |-- Fold_1_Step_2_DE_Manitoba_MSE_xxxacc_1th-fold_lr-xxx_sen-xxx_spec-xxx_corr-xxx_epoch-xxx_pv-xxx_rho-xxx_prho-xxx.pt
        ...
```
For results calculation including 3-class accuracy based on AAC-24 scoring method, Pearson Correlation, Kendall Tau etc., copy the file ```./scripts/results_calculation.py``` to ```./outputs/results_calculation.py``` 
and run the following command by adding the relevant folders in the file:

```
python results_calculation.py --folder Step_2_DE_Manitoba Step_2_SE_Manitoba Step_2_SE_US_Model_Combined
```
The results would be saved in the form of ```.txt``` files as follows:
```
|-- outputs
    |-- results_calculation.py
    |-- Step_1_DE_Manitoba
    |-- Step_2_DE_Manitoba
    |-- Step_1_SE_Manitoba
    |-- Step_2_SE_Manitoba
    |-- Step_1_SE_US_Model_Combined   
    |-- Step_2_SE_US_Model_Combined
    |-- Step_2_DE_Manitoba_results.txt
    |-- Step_2_SE_Manitoba_results.txt
    |-- Step_2_SE_US_Model_Combined_results.txt
    
```
## Predictions on New Data
For prediction using Manitoba SE trained models, use the following command:
```
python predict_Manitoba_SE.py
```
For prediction using Manitoba DE trained models, use the following command:
```
python predict_Manitoba_DE.py
```
For prediction using US-Model combined SE trained models, use the following command:
```
python predict_US_Model_combined_SE.py
```
The predictions would be in the following directories:
```
|-- outputs     
    |-- predictions
        |-- Manitoba_DE
            |-- fold_0_model_predictions.csv
            |-- fold_1_model_predictions.csv
            ...
        |-- Manitoba_SE
            |-- fold_0_model_predictions.csv
            |-- fold_1_model_predictions.csv
            ...
        |-- US_Model_combined_SE
            |-- fold_0_model_predictions.csv
            |-- fold_1_model_predictions.csv
            ...
    |-- Step_1_DE_Manitoba
    |-- Step_2_DE_Manitoba
    |-- Step_1_SE_Manitoba
    |-- Step_2_SE_Manitoba
    |-- Step_1_SE_US_Model_Combined   
    |-- Step_2_SE_US_Model_Combined 
```

