# Data Prepare
This directory contains an implementation of Patch Generation, Offline Cutmix Augmentation

## Patch Generation
---
We provide code to generate patches from slides. To efficiently create patches, we utilize multiprocessing to generate patches from multiple slides simultaneously. For this, a CSV file containing the paths of the slides in a column named slide_path is required. We also assume that the annotation files, which are .xml files for each slide, are located in the same path as the slides.

### csv file example
```
slide_path
/data1/sample11.svs
/data1/sample12.svs
/data2/sample21.ndpi
/data2/sample22.ndpi
/data3/sample31.mrxs
/data3/sample32.mrxs
...
```

### How to use
```
python3 get_patch.py -dp ./slide/path/csv/file.csv -sp ./path/to/patch/saved
```

## Offline Cutmix Augmentation
---
Additionally, we provide code to generate cutmix patches from the created patches. For this, a CSV file containing a column named normal_path with paths to normal patches, a column named leiomyoma_path with paths to leiomyoma patches, and a column named schwannoma_path with paths to schwannoma patches is required.

### csv file example
```
normal_path,leiomyoma_path,schwannoma_path
/data1/sample11/1_1.png,/data2/sample21/1_2.png,/data3/sample31/1_2.png
/data1/sample12/1_1.png,/data2/sample21/1_2.png,/data3/sample31/1_2.png
...
```

### How to use
```
python3 get_cutmix_patch.py -dp ./patch/path/csv/file.csv  -cd ./path/to/cutmix/patch/saved
```