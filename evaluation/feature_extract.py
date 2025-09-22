import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.model import resnet50, swin_transformer_tiny, se_resnext101_32x4d
import pandas as pd
import numpy as np
import cv2, os, torch, argparse, pickle, warnings, glob
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings('ignore')
cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True

# ----------------- model utils -----------------
def model_setting(model_name):
    if model_name == 'resnet50':
        model = resnet50()
    elif model_name == 'swin_t':
        model = swin_transformer_tiny()
    elif model_name == 'se_resnext101':
        model = se_resnext101_32x4d()
    else:
        raise ValueError(f'Unknown model: {model_name}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model


# ----------------- data utils -----------------
class PatchDataset(Dataset):
    def __init__(self, img_paths):
        self.paths = img_paths
        self.tf = A.Compose([
            ToTensorV2()
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = cv2.imread(p).astype(np.float32)/255
        x = self.tf(image=img)['image']  # C,H,W float32
        return x, p

def run_dir_feature_extraction(model, img_list, batch_size, num_workers, use_fp16):
    """feature extraction for a list of images with multi-worker & batch"""
    device = next(model.parameters()).device
    ds = PatchDataset(img_list)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=(num_workers > 0),
        shuffle=False
    )

    feats = []
    model.eval()
    autocast = torch.cuda.amp.autocast if (use_fp16 and device.type == 'cuda') else torch.cpu.amp.autocast
    with torch.inference_mode():
        for xb, _paths in dl:
            xb = xb.to(device, non_blocking=True)
            with (torch.cuda.amp.autocast() if use_fp16 and device.type=='cuda' else torch.no_grad()):
                y = model(xb)          
                if y.dim() == 4:       
                    y = F.adaptive_avg_pool2d(y, 1).flatten(1)
            feats.append(y.cpu())
    return torch.cat(feats, dim=0)

# ----------------- main -----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',   '--model', required=True, help='Model: resnet50 | swin_t | se_resnext101')
    parser.add_argument('-msp', '--model_save_path', required=True, help='Path to model.pth')
    parser.add_argument('-tbd', '--train_base_dir', required=True, help='Output directory')
    parser.add_argument('-tpd', '--test_patch_dir', required=True, help='Output directory')
    parser.add_argument('-trp', '--train_path', required=True, help='Path to train pickle')
    parser.add_argument('-vp',  '--valid_path', required=True, help='Path to valid pickle')
    parser.add_argument('--batch_size', required=False, type=int, default=128)
    parser.add_argument('--num_workers', required=False, type=int, default=8)
    parser.add_argument('--fp16', action='store_true', help='Use AMP (float16) inference')
    args = parser.parse_args()

    description = os.path.basename(args.model_save_path).split('.')[0]
    subtype_dict = {'Gist':0, 'Leiomyoma':1, 'Schwannoma':2}

    # model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model_setting(args.model)
    model.load_state_dict(torch.load(args.model_save_path), strict=True)
    model = nn.Sequential(*list(model.children())[:-1])

    for i in range(1,5):
        if i != 1:
            model_save_path = args.model_save_path.replace('.pth', f'{i}.pth')
            train_zip_path = args.train_path.replace('_zip.pickle', f'{i}_zip.pickle')
            valid_zip_path = args.valid_path.replace('_zip.pickle', f'{i}_zip.pickle')
        else:
            model_save_path = args.model_save_path
            train_zip_path = args.train_path
            valid_zip_path = args.valid_path

        description = os.path.basename(model_save_path).split('.')[0]
        train_base_dir = args.train_base_dir
        organ_dir = ['ISH', 'SS']
        subtype_dict = {'Gist':0, 'Leiomyoma':1, 'Schwannoma':2}
        df = pd.DataFrame(columns=['path', 'label', 'stage'])

        # stage lists
        with open(train_zip_path,'rb') as fr: train_zip = pickle.load(fr)
        with open(valid_zip_path,'rb') as fr: valid_zip = pickle.load(fr)
        train_list = list(set([x[0].split('/')[-2] for x in train_zip]))
        valid_list = list(set([x[0].split('/')[-2] for x in valid_zip]))

        # model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model_setting(args.model)
        model.load_state_dict(torch.load(model_save_path), strict=True)
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()

        print(model_save_path, 'loaded')
        cnt = 0
        for organ in organ_dir:
            for subtype in subtype_dict.keys():
                #  train_base_dir/organ/subtype/case_dir/*.png
                organ_subtype_base = os.path.join(train_base_dir, organ, subtype)
                organ_subtype_list = os.listdir(organ_subtype_base)

                for case_dir in organ_subtype_list:
                    organ_subtype_path = os.path.join(organ_subtype_base, case_dir)
                    if not os.path.isdir(organ_subtype_path):
                        continue

                    img_list = [os.path.join(organ_subtype_path, x)
                                for x in os.listdir(organ_subtype_path)
                                if x.endswith('.png')]
                    if len(img_list) == 0:
                        continue

                    # multi-worker + batch inference
                    feats = run_dir_feature_extraction(
                        model, img_list,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        use_fp16=args.fp16
                    )  # (N, D) tensor (CPU)

                    feature_save_path = os.path.join(organ_subtype_path, f'{description}.pt')
                    torch.save(feats, feature_save_path)

                    if case_dir in train_list:
                        stage = 'train'
                    elif case_dir in valid_list:
                        stage = 'valid'
                    else:
                        continue

                    df.loc[len(df)] = [f'{organ_subtype_path}/{description}.pt', subtype_dict[subtype], stage]
                    cnt += 1
                    print(cnt, (len(train_list) + len(valid_list)), end='\r')
        df.to_csv(os.path.join(train_base_dir, f'{description}_feature.csv'), index=False)


    test_patch_list = glob.glob(f'{args.test_patch_dir}/*/*/*')

    cnt = 0
    df = pd.DataFrame(columns=['path', 'label', 'stage'])
    stage = 'test'
    for internal_base in test_patch_list:
        subtype = internal_base.split('/')[-2]  
        img_list = glob.glob(os.path.join(internal_base, '*.png'))
        if len(img_list) == 0:
            continue

        # multi-worker + batch inference
        feats = run_dir_feature_extraction(
            model, img_list,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_fp16=args.fp16
        )  # (N, D) tensor (CPU)

        feature_save_path = os.path.join(internal_base, f'{description}.pt')
        torch.save(feats, feature_save_path)

        df.loc[len(df)] = [feature_save_path, subtype_dict[subtype], stage]
        cnt += 1
        print(cnt, len(test_patch_list), end='\r')
    df.to_csv(os.path.join(args.test_patch_dir, f'{description}_{stage}_feature.csv'), index=False)