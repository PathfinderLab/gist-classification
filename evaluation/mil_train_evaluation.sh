python mil_train_evaluation.py \
    -m abmil \
    -tcp /workspace/gist/data/patch/resnet50_aug_cutmix_feature.csv \
    -icp /workspace/gist/data/patch/resnet50_aug_cutmix_internal_test_feature.csv \
    -ecp /workspace/gist/data/patch/resnet50_aug_cutmix_external_test_feature.csv \
    -msp ./model_save_path.pth \
    -rsp ./results_save_path.txt \