# GIST-classification
Subtype classification of gastric spindle cell tumors in whole slide images

## Abstract
**Aims**: Accurate cancer subtype classification is critical due to variations in tumor progression and prognosis. Traditionally, pathologists classified subtypes manually by examining pathological slides under the microscope. To address increasing workloads, this study aimed to classify gastric spindle cell tumor (GSCT) subtypes—gastrointestinal stromal tumors (GIST), leiomyomas, and schwannomas—using a convoluted neural network applied to whole-slide images (WSIs). To date, no automated pipeline has been proposed for classifying GSCTs into GIST, leiomyoma, and schwannoma from hematoxylin and eosin (H&E) WSIs. The limited number of leiomyoma and schwannoma cases necessitates training on class-imbalanced datasets, which can cause performance degradation for minority classes and model overconfidence. </br>
**Methods and results**:
We applied CutMix operations to generate datasets for minority classes, constructing a balanced dataset that improved F1 scores for leiomyoma and schwannoma from 0.9187 to 0.9651 and 0.9258 to 0.9808 in internal validation, and from 0.6004 to 0.8795 and 0.4792 to 0.6181 in external validation, respectively. Additionally, considering model overconfidence, we employed Confident Instance Voting (CIV) on instances with confidence exceeding 0.9, unlike conventional hard voting, achieving accuracy from 0.9680 to 0.9826 and from 0.9138 to 0.9262 in internal and external validation. We propose a first fully automated pipeline for classifying GSCTs into three classes from hematoxylin and eosin (H&E) WSIs using the aforementioned approaches. </br>
**Conclusions**:
This pipeline demonstrates potential as a clinical tool for categorizing gastric spindle cell tumor subtypes. It offers improved diagnostic efficiency and reliability, particularly in addressing class imbalance in histopathological analysis.

### The workflow of the algorithm we proposed
![img1](./img/figure_1.png)

(A) Data from two of the three institutions were used for training and internal validation, while data from the remaining institution were used for external validation. (B) Our CutMix operation was applied to Leiomyoma, Schwannoma, and Normal data. (C) During training, balanced instances were generated through CutMix, followed by color and distortion augmentations before model training. (D) In the proposed Confident Instance Voting method, hard voting was applied only when the predicted probability of the trained model exceeded 0.9

## Description

### Repository Structure
- `data_prepare/`: patch generation and offline cutmix augmentation directory
- `evaluation/`: patch-wise and slide-wise evaluation directory
- `img/`: img directory
- `models/`: model directory
- `sample/`: sample dataset directory
- `training/`: our model training directory
- `utils/`: utils for training and inference directory
- `run.py`: automated pipeline for GSCT WSIs subtype 

### inference example

Pretrained model weights, an anonymized example WSI, and a corresponding **coarse ROI-level XML annotation** used in this repository can be downloaded from [Google Drive](<https://drive.google.com/drive/folders/1mPNHhik41d8YuscOifX9MnoADBT04Kzs?usp=sharing>).  
The XML file provides a simplified region-of-interest annotation that is sufficient to demonstrate the patch-generation pipeline and is **not intended as precise pixel-wise ground truth**.

After downloading, save the model weight file (e.g., `resnet50_aug_cutmix2.pth`) in your working directory (or provide its path via `-model_path`), and place the example WSI file **together with its XML annotation** in `./sample/wsi/` (or update `-slide_path` accordingly) before running the command below.


```
python run.py \
    --slide_path ./sample/wsi/schwannoma_0001.svs \
    --model_path ./resnet50_aug_cutmix2.pth \
    --results_save_path ./results/test_schw.png
```

### results example
- `test.png/`: visualized CIV results 
- `test.txt`: GSCT subtype results
