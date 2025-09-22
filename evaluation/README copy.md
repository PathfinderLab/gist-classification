# Evaluation 

patch-wise와 slide-wise 수준에서 모델의 성능을 평가하기 위한 코드입니다.

patch_evaluation.py
- training에서 학습시킨 모델의 patch-wise 성능을 확인합니다.

feature_extract.py
- training에서 학습시킨 모델로 train / valid / test의 patch들에서 feature를 추출합니다.

wsi_evaluation.py
- test WSI들에 대해서 feat_extract에서 뽑은 feature를 기반으로 CIV 기반 hard voting을 진행하여 subtype 분류 후 evaluation을 진행합니다.

mil_train_evaluation.py
- train WSI에서 추출한 feature를 기반으로 MIL 모델을 학습 후 test WSI들에 대해서 feat_extract에서 뽑은 feature를 기반으로 subtype 분류 후 evaluation을 진행합니다.



## patch wise evaluation

### pickle file example for patch evaluation
```
[('/data1/sample11/1_1.png',tensor([1,0,0,0])),
...
]
```

### patch evaluation example
```
python patch_evaluation.py \
    -m resnet50 \
    -mdp ./model_save_path.pth \
    -tep ./test_pickle_path.pickle \
    -rsp ./results_save_path.txt \
```


## feature extraction

### preparation for patch evaluation
train_base_dir과 test_patch_dir 설정을 위하여 다음과 같은 경로로 patch가 만들어져 있어야 합니다.
```
train_base_dir/organ/subtype/case_dir/
test_patch_dir/organ/patient/case_dir/

ex) 
train_base_dir/ISH/GIST/slide1/
test_patch_dir/KUMC/patient1/slid1

```

### feature extraction example
```
python feature_extract.py \
    -m resnet50 \
    -msp ./model_save_path.pth \
    -tbd ./train_base_dir \
    -tpd ./test_patch_dir \
    -trp ./train_patch.pickle \
    -vp ./valid_patch.pickle \
```


## slide wise evaluation

### csv file example for inference
```
path,label,stage
test_patch_dir/organ/patient/case_dir/resnet_50.pt,0,test
```

### sldie wise evaluation example
```
python wsi_evaluation.py \
    -m resnet50 \
    -mp ./resnet50_base.pth \
    -icp ./resnet50_base_internal_test_feature.csv \
    -ecp ./resnet50_base_external_test_feature.csv \
    -rsp ./result_save_path.txt
```


## MIL

### csv file example for inference
```
path,label,stage
test_patch_dir/organ/patient/case_dir/resnet_50.pt,0,test
```

### MIL train evaluation example
```
python inference.py \
    -m abmil \
    -tcp ./resnet50_base_feature.csv \
    -icp ./resnet50_base_internal_test_feature.csv \
    -ecp ./resnet50_base_external_test_feature.csv \
    -msp ./model_save_path.pth \
    -rsp ./results_save_path.txt 
```