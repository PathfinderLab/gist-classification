from torch.utils.data import DataLoader
from models.model import resnet50, swin_transformer_tiny, se_resnext101_32x4d
from utils.util import SetWSI
from PIL import Image
import torch.nn.functional as F
import numpy as np
import openslide
import argparse
import torch
import cv2
import os


def model_setting(model_name, device):
    if model_name == 'resnet50':
        model = resnet50()
    elif model_name == 'swin_t':
        model = swin_transformer_tiny()
    elif model_name == 'se_resnext101':
        model = se_resnext101_32x4d()
    model.to(device)
    model.eval()
    return model


def get_results(model, device, slide_path, threshold, target_mpp, target_size, overlap, batch_size, num_workers, use_fp16, results_save_path):
    slide = openslide.open_slide(slide_path)
    slide_image = np.array(slide.get_thumbnail(((1000,1000))).convert("RGB"))
    wsi_set = SetWSI(slide_path, target_mpp, target_size, overlap, return_loc=True)
    ret0 = np.zeros((wsi_set.thumbnail_mask.shape[0]*target_size, wsi_set.thumbnail_mask.shape[1]*target_size))
    
    dl = DataLoader(
        wsi_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=(num_workers > 0),
        shuffle=False
    )

    # extract results
    feats = []
    tile_loc_list = []  
    with torch.inference_mode():
        for xb, tile_loc in dl:
            xb = xb.to(device, non_blocking=True)
            with (torch.cuda.amp.autocast() if use_fp16 and device.type=='cuda' else torch.no_grad()):
                y = model(xb)              
            feats.append(y.cpu())
            tile_loc_list.extend(tile_loc)
    patch_logits = torch.cat(feats, dim=0)  # (N, D)

    #############################
    # Confident Instance Voting #
    #############################
    with torch.no_grad():
        patch_probs = F.softmax(patch_logits, dim=1)
        max_probs, patch_preds = torch.max(patch_probs, dim=1)
        confident_mask = (max_probs >= threshold) & (patch_preds != 0)

        if confident_mask.sum() > 0:
            # Case 1: There are confident instances -> hard voting
            confident_preds = patch_preds[confident_mask]
            confident_probs = patch_probs[confident_mask]
            
            unique_classes, counts = torch.unique(confident_preds, return_counts=True)
            max_count = torch.max(counts)
            tied_classes = unique_classes[counts == max_count]
            
            if len(tied_classes) >= 2:
                # Case 1-1 There are equally voted classes -> soft voting
                class_prob_sums = {}
                for cls in tied_classes:
                    cls_mask = (confident_preds == cls)
                    cls_prob_sum = confident_probs[cls_mask][:, cls-1].sum().item()
                    class_prob_sums[cls.item()] = cls_prob_sum
                
                final_pred_cls = max(class_prob_sums, key=class_prob_sums.get)
            else:
                # Case 1-2 There is a clear winner -> hard voting
                final_pred = unique_classes[torch.argmax(counts)]
                final_pred_cls = final_pred.cpu().item() 
        else:
            # Case 2: No confident instances -> soft voting
            valid_patch_mask = (patch_preds != 0)
            
            if valid_patch_mask.sum() > 0:
                # Case 2-1: There are non-zero class patches -> soft voting
                valid_patch_preds = patch_preds[valid_patch_mask]
                valid_patch_probs = patch_probs[valid_patch_mask]
                
                unique_classes = torch.unique(valid_patch_preds)
                class_prob_sums = {}
                
                for cls in unique_classes:
                    cls_mask = (valid_patch_preds == cls)
                    cls_prob_sum = valid_patch_probs[cls_mask][:, cls-1].sum().item()
                    class_prob_sums[cls.item()] = cls_prob_sum
                
                final_pred_cls = max(class_prob_sums, key=class_prob_sums.get)
            else:
                # Case 2-2: All patches are class 0 -> error
                assert (patch_preds == 0).all(), "All patches should be predicted as class 0."
            
    # save visualization
    for (max_prob, patch_pred, tile_loc) in zip(max_probs, patch_preds, tile_loc_list):
        if (max_prob.item() >= threshold) and (patch_pred.item() != 0):
            ret0[eval(tile_loc)[1]*target_size:(eval(tile_loc)[1]+1)*target_size, eval(tile_loc)[0]*target_size:(eval(tile_loc)[0]+1)*target_size] = np.ones((target_size,target_size))*patch_pred.item()

    ret0 = cv2.resize(ret0, (slide_image.shape[1], slide_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    img_with_msk = image_with_mask(slide_image.copy(), ret0)
    os.makedirs('/'.join(results_save_path.split('/')[:-1]), exist_ok=True)
    Image.fromarray(img_with_msk).save(results_save_path)
    label_dict = {1:'GIST', 2:'Leiomyoma', 3:'Schwannoma'}
    with open(results_save_path.replace('.png','.txt'), 'w') as f:
        f.write(f'Slide Path: {slide_path}\n')
        f.write(f'Predicted Label: {label_dict[final_pred_cls]}\n')

    return final_pred_cls


def image_with_mask(image, mask):
    image_orig = image.copy()
    color_dict = {1:(255,0,0), 2:(0,0,255), 3:(44,44,44)}
    for i in range(3):
        mask_type    = np.where(mask==i+1,1,0).astype(np.uint8)
        mask_cont, _ = cv2.findContours(mask_type, cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_SIMPLE) 
        image        = cv2.fillPoly(image, mask_cont, color_dict[i+1])
    img_with_msk = cv2.addWeighted(src1=image_orig, alpha=0.6, src2=image, beta=0.4, gamma=0)
    return img_with_msk


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_path', required=True, type=str, help='Path to the Slide files')
    parser.add_argument('--model_path', required=True, type=str, help='Path to the Model file')
    parser.add_argument('--results_save_path', required=True, type=str, help='Path to the results save file')
    parser.add_argument('--threshold', required=False, type=float, default=0.9, help='Threshold for confident instance')
    parser.add_argument('-tm', '--target_mpp', required=False, default=1.0, help='Target mpp')
    parser.add_argument('-ts', '--target_size', required=False, default=512, help='Target patch size')
    parser.add_argument('-o', '--overlap', required=False, default=0, help='Overlap')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--fp16', action='store_true', help='Use AMP (float16) inference')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model_setting(model_name='resnet50', device=device)  
    model.load_state_dict(torch.load(args.model_path ))
    model.eval()

    label = get_results(model, device, args.slide_path, args.threshold, args.target_mpp, args.target_size, args.overlap, args.batch_size, args.num_workers, args.fp16, args.results_save_path)