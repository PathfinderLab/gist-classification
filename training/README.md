# Model Training Example

This directory provides training code for a patch-level classification model.  
- **Input**: 512x512 patches at 10X magnification from WSIs  
- **Task**: Classify patches into **GIST, Leiomyoma, Schwannoma, Normal**  
- **Method**: Trained with **CutMix patches** → labels are **one-hot encoded**  
- **Cross-validation**: 4-fold, requiring multiple training/validation pickle files  

---

### Data Preparation

- For each fold, you need separate pickle files.  
- **Training & Validation**  
  - `train_zip.pickle`, `valid_zip.pickle`  
  - `train2_zip.pickle`, `train3_zip.pickle`, `train4_zip.pickle`  
- **CutMix Pairs**  
  - `normal_schwannoma.pickle`, `normal_leiomyoma.pickle`, `leiomyoma_schwannoma.pickle`  
  - `normal_schwannoma_2.pickle`, `normal_schwannoma_3.pickle`, `normal_schwannoma_4.pickle` (and similarly for others)

### train example
```
python patch_train.py \
    -m resnet50 \
    -msp ./model_save_path.pth \
    -trp ./data_prepare/train_zip.pickle \
    -vp ./data_prepare/valid_zip.pickle \
    -nsp ./data_prepare/normal_schwannoma.pickle \
    -nlp ./data_prepare/normal_leiomyoma.pickle \
    -lsp ./data_prepare/leiomyoma_schwannoma.pickle \
    --use_cutmix \
    --use_imb_sampler \
```

### pickle file example for train
```
[('/data1/sample11/1_1.png',tensor([1,0,0,0])),
('/cutmix_path/leiomyoma_schwannoma/1.png',tensor([0,0,0.73,0.27])),
...
]
```