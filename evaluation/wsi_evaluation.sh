python wsi_evaluation.py \
    -m resnet50 \
    -mp /workspace/gist/script/stomach-subtype/test/resnet50_base.pth \
    -icp /workspace/gist/data/patch/resnet50_base_internal_test_feature.csv \
    -ecp /workspace/gist/data/patch/resnet50_base_external_test_feature.csv \
    -rsp ./result_save_path.txt