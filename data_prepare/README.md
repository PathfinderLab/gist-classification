# Data Prepare
This directory contains an implementation of Patch Generation, Offline Cutmix Augmentation

## Patch Generation
---
We provide code to generate patches from whole-slide images (WSIs). To efficiently create patches, we utilize multiprocessing so that patches from multiple slides can be generated in parallel. The script expects a CSV file containing the paths of the slides in a column named `slide_path`, and we assume that slide-level annotation files (one `.xml` file per slide) are located in the same directory as the corresponding WSIs.

In this repository, we additionally provide **one anonymized WSI from the external cohort together with a very coarse ROI-level XML annotation**. This XML contains only a simplified label/ROI that is sufficient to demonstrate how to run the patch-generation pipeline on the example WSI, and it is **not intended to serve as precise pixel-wise ground truth**. For any other datasets, users are expected to prepare their own slide-level XML annotations in the same directory structure and format.

### csv file example
```
slide_path
../sample/wsi/schwannoma_0001.svs
...
```

### How to use
```
python3 get_patch.py \
    -dp ./patch_sample.csv \
    -sp ../sample/test_generation 
```

## Offline Cutmix Augmentation
---
Additionally, we provide code to generate cutmix patches from the created patches. For this, a CSV file containing a column named normal_path with paths to normal patches, a column named leiomyoma_path with paths to leiomyoma patches, and a column named schwannoma_path with paths to schwannoma patches is required.

### csv file example
```
normal_path,leiomyoma_path,schwannoma_path
../sample/patch/leiomyoma/1_1.png,../sample/patch/leiomyoma/2_2.png,../sample/patch/schwannoma/2_2.png
../sample/patch/schwannoma/3_1.png,../sample/patch/leiomyoma/3_2.png,../sample/patch/schwannoma/1_2.png
...
```

### How to use
```
python3 get_cutmix_patch.py \
    -dp ./cutmix_sample.csv  \
    -cd ../sample/patch/cutmix_patch/ \
    -nns 5 \
    -nnl 5 \
    -nls 5 
```