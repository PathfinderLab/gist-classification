# GIST-classification
Subtype Classification of Gastric Spindle Cell Tumors in Whole Slide Images

## Abstract
**Aims**: Accurate cancer subtype classification is critical due to variations in tumor progression and prognosis. Traditionally, pathologists classified subtypes manually by examining pathological slides under the microscope. To address increasing workloads, this study aimed to classify gastric spindle cell tumor subtypes (GSCT)—gastrointestinal stromal tumors (GIST), leiomyomas, and schwannomas—using a convoluted neural network applied to whole-slide images (WSIs).  
**Methods and results**:  To date, no automated pipeline has been proposed for classifying GSCTs into GIST, Leiomyoma, and Schwannoma from hematoxylin and eosin (H&E) WSIs. The limited number of Leiomyoma and Schwannoma cases necessitates training on class-imbalanced datasets, which can cause performance degradation for minority classes and model overconfidence. We applied CutMix operations to generate datasets for minority classes, constructing a balanced dataset that improved F1 scores for Leiomyoma and Schwannoma from 0.9187 to 0.9651 and 0.9258 to 0.9808 in internal validation, and from 0.6004 to 0.8795 and 0.4782 to 0.6181 in external validation, respectively. Additionally, considering model overconfidence, we employed Confident Instance Voting (CIV) on instances with confidence exceeding 0.9, unlike conventional hard voting, achieving accuracy from 0.9680 to 0.9826 and from 0.9138 to 0.9262 in internal and external validation. We propose a first fully automated pipeline for classifying GSCTs into three classes from hematoxylin and eosin (H&E) WSIs using the aforementioned approaches.  
**Conclusions**: This pipeline demonstrates potential as a clinical tool for categorizing gastric spindle cell tumor subtypes. It offers improved diagnostic efficiency and reliability, particularly in addressing class imbalance in histopathological analysis.


### The workflow of the algorithm we proposed
![img1](./img/figure_1.png)

(A) Data from two of the three institutions were used for training and internal validation, while data from the remaining institution were used for external validation. (B) Our CutMix operation was applied to Leiomyoma, Schwannoma, and Normal data. (C) During training, balanced instances were generated through CutMix, followed by color and distortion augmentations before model training. (D) In the proposed Confident Instance Voting method, hard voting was applied only when the predicted probability of the trained model exceeded 0.9

## Description

### Repository Structure
- `data_prepare/`: patch generation and offline cutmix augmentation directory
- `evaluation/`: patch-wise and slide-wise evaluation directory
- `img/`: img directory
- `mil/`: mil model training directory
- `model/`: model directory
- `training/`: our model training directory
- `utils/`: utils for training and inference directory
- `run.py`: automated pipeline for GSCT WSIs subtype 

### inference example
```
python run.py \
    -slide_path ./example.svs \
    -model_path ./model_save_path.pth \
    -results_save_path ./results/test.png
```

### results example
- `test.png/`: visualized CIV results 
- `test.txt`: GSCT subtype results
