python3 patch_train.py \
    -m resnet50 \
    -msp ./resnet50_base3.pth \
    -trp /workspace/gist/script/stomach-subtype/data/train_zip.pickle \
    -vp /workspace/gist/script/stomach-subtype/data/valid_zip.pickle \
    --loss CrossEntropyLoss \