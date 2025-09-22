# Model Evaluation

This repository contains code for evaluating model performance at both patch-wise and slide-wise levels for histopathology image analysis.

## Overview

The evaluation pipeline consists of four main components:

- **patch_evaluation.py**: Evaluates patch-wise performance of trained models
- **feature_extract.py**: Extracts features from train/validation/test patches using trained models
- **wsi_evaluation.py**: Performs slide-wise evaluation using CIV-based hard voting for subtype classification
- **mil_train_evaluation.py**: Trains MIL models on extracted features and evaluates on test WSIs


## Patch-wise Evaluation

### Data Format
The pickle file for patch evaluation should contain tuples of image paths and one-hot encoded labels:
```
[('/data1/sample11/1_1.png', tensor()),
('/data1/sample11/1_2.png', tensor()),
...
]
```

### Usage
```
python patch_evaluation.py
-m resnet50
-mdp ./model_save_path.pth
-tep ./test_pickle_path.pickle
-rsp ./results_save_path.txt
```


## Feature Extraction

### Directory Structure
Ensure patches are organized in the following directory structure:
```
train_base_dir/organ/subtype/case_dir/
test_patch_dir/organ/patient/case_dir/

# Example
train_base_dir/ISH/GIST/slide1/
test_patch_dir/KUMC/patient1/slide1/
```

### Usage
```
python feature_extract.py
-m resnet50
-msp ./model_save_path.pth
-tbd ./train_base_dir
-tpd ./test_patch_dir
-trp ./train_patch.pickle
-vp ./valid_patch.pickle
```


## Slide-wise Evaluation

### Data Format
CSV files for inference should follow this format:

```
path,label,stage
test_patch_dir/organ/patient/case_dir/resnet_50.pt,0,test
test_patch_dir/organ/patient/case_dir/resnet_50.pt,1,test
```

### Usage
```
python wsi_evaluation.py
-m resnet50
-mp ./resnet50_base.pth
-icp ./resnet50_base_internal_test_feature.csv
-ecp ./resnet50_base_external_test_feature.csv
-rsp ./result_save_path.txt
```


## Multiple Instance Learning (MIL)

### Data Format
CSV files should use the same format as slide-wise evaluation:
```
test_patch_dir/organ/patient/case_dir/resnet_50.pt,0,test
test_patch_dir/organ/patient/case_dir/resnet_50.pt,1,test
```

### Usage
```
python mil_train_evaluation.py
-m abmil
-tcp ./resnet50_base_feature.csv
-icp ./resnet50_base_internal_test_feature.csv
-ecp ./resnet50_base_external_test_feature.csv
-msp ./model_save_path.pth
-rsp ./results_save_path.txt
```