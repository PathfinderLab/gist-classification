python feature_extract.py \
    -m resnet50 \
    -msp /workspace/gist/script/stomach-subtype/test/resnet50_base.pth \
    -tbd /workspace/gist/data/patch/ \
    -tpd /workspace/gist/data/patch/internal_test/ \
    -trp /workspace/gist/script/stomach-subtype/data/train_zip.pickle \
    -vp /workspace/gist/script/stomach-subtype/data/valid_zip.pickle \