import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.model import resnet50, swin_transformer_tiny, se_resnext101_32x4d
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import argparse

def model_setting():
    model = resnet50()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model

def get_results(model, df):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i in tqdm(range(len(df)), desc="Processing bags"):
            # load bag features
            bag_features = torch.load(df.iloc[i]['path'])
            bag_features = bag_features.to(device)
            
            # predict patch logits
            patch_logits = model(bag_features)
            patch_probs = F.softmax(patch_logits, dim=1)
            
            # ----------------- Confident Instance Voting -----------------
            max_probs, patch_preds = torch.max(patch_probs, dim=1)
            confident_mask = (max_probs >= 0.9) & (patch_preds != 0)

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
            
            # save final prediction
            all_preds.append(final_pred_cls - 1)  # 0-based indexing
            all_labels.append(int(df.iloc[i]['label']))

    print("Inference completed.")
    print(f"all labels: {all_labels}")    
    print(f"all preds: {all_preds}")    

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    mprecision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    mrecall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    mf1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return all_preds, all_labels, accuracy, precision, recall, f1, mprecision, mrecall, mf1

def get_results_hard_voting(model, df):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i in tqdm(range(len(df)), desc="Processing bags"):
            # load bag features
            bag_features = torch.load(df.iloc[i]['path'])
            bag_features = bag_features.to(device)
            
            # predict patch logits
            patch_logits = model(bag_features)
            patch_probs = F.softmax(patch_logits, dim=1)
            _, patch_preds = torch.max(patch_probs, dim=1)
            
            # except Normal class 
            valid_patch_mask = (patch_preds != 0)
            
            if valid_patch_mask.sum() > 0:  
                valid_patch_preds = patch_preds[valid_patch_mask]
                valid_patch_probs = patch_probs[valid_patch_mask]
                
                # ---------------- Hard Voting -----------------
                unique_classes, counts = torch.unique(valid_patch_preds, return_counts=True)
                max_count = torch.max(counts)
                tied_classes = unique_classes[counts == max_count]
                
                if len(tied_classes) > 1:
                    # Case 1-1 There are equally voted classes -> soft voting
                    class_prob_sums = {}
                    for cls in tied_classes:
                        cls_mask = (valid_patch_preds == cls)
                        cls_prob_sum = valid_patch_probs[cls_mask][:, cls-1].sum().item()
                        class_prob_sums[cls.item()] = cls_prob_sum
                    
                    final_pred_cls = max(class_prob_sums, key=class_prob_sums.get)
                    all_preds.append(final_pred_cls - 1)  # 0-based indexing
                else:
                    # Case 1-2 There is a clear winner -> hard voting
                    final_pred = unique_classes[torch.argmax(counts)]
                    all_preds.append(final_pred.cpu().item() - 1)  # 0-based indexing
            else:
                # Case 2-2: All patches are class 0 -> error
                assert (patch_preds == 0).all(), "All patches should be predicted as class 0."
            
            # Ground truth label
            all_labels.append(int(df.iloc[i]['label']))

    print("Hard Voting Inference completed.")
    print(f"all labels: {all_labels}")    
    print(f"all preds: {all_preds}")    

    # calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    mprecision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    mrecall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    mf1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return all_preds, all_labels, accuracy, precision, recall, f1, mprecision, mrecall, mf1


def calculate_class_f1_scores(pred, label):
    """calculate F1-score for each class"""
    f1_scores = f1_score(label, pred, average=None, labels=[0, 1, 2], zero_division=0)
    return f1_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, type=str, help='Path to the model path')
    parser.add_argument('-mp', '--model_path', required=True, type=str, help='Path to the model path')
    parser.add_argument('-icp', '--int_csv_path', required=True, type=str, help='Path to the internal CSV file containing data paths and labels')
    parser.add_argument('-ecp', '--ext_csv_path', required=True, type=str, help='Path to the external CSV file containing data paths and labels')
    parser.add_argument('-rsp', '--results_save_path', required=True, type=str, help='Path to the CSV file containing data paths and labels')
    parser.add_argument('--use_hard_voting', action='store_true')

    args = parser.parse_args()
    int_csv_path = args.int_csv_path
    ext_csv_path = args.ext_csv_path
    model_path = args.model_path  
    results_save_path = args.results_save_path
    method = model_path.split('/')[-1].split('resnet50_')[-1].split('.pth')[0]

    # Load dataframes and models for all folds
    int_dfs = []
    ext_dfs = []
    models = []
    
    for i in range(1, 5):
        suffix = '' if i == 1 else str(i)
        
        # Load dataframes
        int_dfs.append(pd.read_csv(int_csv_path.replace(method, method + suffix)))
        ext_dfs.append(pd.read_csv(ext_csv_path.replace(method, method + suffix)))
        
        # Load models
        model = model_setting()
        model.load_state_dict(torch.load(model_path.replace(method, method + suffix)))
        models.append(list(model.children())[-1])

    # Get results for all folds
    internal_results = []
    external_results = []
    
    for i in range(4):
        if args.use_hard_voting:
            int_result = get_results_hard_voting(models[i], int_dfs[i])
            ext_result = get_results_hard_voting(models[i], ext_dfs[i])
        else:
            int_result = get_results(models[i], int_dfs[i])
            ext_result = get_results(models[i], ext_dfs[i])
        
        internal_results.append(int_result)
        external_results.append(ext_result)

    # Extract metrics arrays
    internal_accuracies = np.array([result[2] for result in internal_results])
    internal_precisions = np.array([result[3] for result in internal_results])
    internal_recalls = np.array([result[4] for result in internal_results])
    internal_f1s = np.array([result[5] for result in internal_results])

    external_accuracies = np.array([result[2] for result in external_results])
    external_precisions = np.array([result[3] for result in external_results])
    external_recalls = np.array([result[4] for result in external_results])
    external_f1s = np.array([result[5] for result in external_results])

    # Calculate class-wise F1 scores
    internal_class_f1s = []
    external_class_f1s = []
    
    for i in range(4):
        internal_class_f1s.append(calculate_class_f1_scores(internal_results[i][0], internal_results[i][1]))
        external_class_f1s.append(calculate_class_f1_scores(external_results[i][0], external_results[i][1]))

    # Convert to numpy arrays for statistics
    internal_all_class_f1s = np.array(internal_class_f1s)
    external_all_class_f1s = np.array(external_class_f1s)

    internal_class_f1_means = np.mean(internal_all_class_f1s, axis=0)
    internal_class_f1_stds = np.std(internal_all_class_f1s, axis=0)
    external_class_f1_means = np.mean(external_all_class_f1s, axis=0)
    external_class_f1_stds = np.std(external_all_class_f1s, axis=0)

    # Write results to file
    with open(results_save_path, 'w') as f:
        # Internal test results
        f.write('[Internal Test]\n')
        for i in range(4):
            fold_num = '' if i == 0 else str(i + 1)
            acc, prec, rec, f1 = internal_results[i][2:6]
            class_f1 = internal_class_f1s[i]
            
            f.write(f'Accuracy{fold_num}: {acc:.4f}, Precision{fold_num}: {prec:.4f}, Recall{fold_num}: {rec:.4f}, F1-score{fold_num}: {f1:.4f}\n')
            f.write(f'Class F1-scores{fold_num}: Class0= {class_f1[0]:.4f}, Class1= {class_f1[1]:.4f}, Class2= {class_f1[2]:.4f}\n\n')
        
        # Average and std for internal
        f.write(f'Avg Accuracy: {np.mean(internal_accuracies):.4f} ± {np.std(internal_accuracies):.4f}, '
                f'Avg Precision: {np.mean(internal_precisions):.4f} ± {np.std(internal_precisions):.4f}, '
                f'Avg Recall: {np.mean(internal_recalls):.4f} ± {np.std(internal_recalls):.4f}, '
                f'Avg F1-score: {np.mean(internal_f1s):.4f} ± {np.std(internal_f1s):.4f}\n')
        f.write(f'Avg Class F1-scores: Class0= {internal_class_f1_means[0]:.4f} ± {internal_class_f1_stds[0]:.4f}, '
                f'Class1= {internal_class_f1_means[1]:.4f} ± {internal_class_f1_stds[1]:.4f}, '
                f'Class2= {internal_class_f1_means[2]:.4f} ± {internal_class_f1_stds[2]:.4f}\n\n')
        
        # External test results
        f.write('[External Test]\n')
        for i in range(4):
            fold_num = '' if i == 0 else str(i + 1)
            acc, prec, rec, f1 = external_results[i][2:6]
            class_f1 = external_class_f1s[i]
            
            f.write(f'Accuracy{fold_num}: {acc:.4f}, Precision{fold_num}: {prec:.4f}, Recall{fold_num}: {rec:.4f}, F1-score{fold_num}: {f1:.4f}\n')
            f.write(f'Class F1-scores{fold_num}: Class0= {class_f1[0]:.4f}, Class1= {class_f1[1]:.4f}, Class2= {class_f1[2]:.4f}\n\n')
        
        # Average and std for external
        f.write(f'Avg Accuracy: {np.mean(external_accuracies):.4f} ± {np.std(external_accuracies):.4f}, '
                f'Avg Precision: {np.mean(external_precisions):.4f} ± {np.std(external_precisions):.4f}, '
                f'Avg Recall: {np.mean(external_recalls):.4f} ± {np.std(external_recalls):.4f}, '
                f'Avg F1-score: {np.mean(external_f1s):.4f} ± {np.std(external_f1s):.4f}\n')
        f.write(f'Avg Class F1-scores: Class0= {external_class_f1_means[0]:.4f} ± {external_class_f1_stds[0]:.4f}, '
                f'Class1= {external_class_f1_means[1]:.4f} ± {external_class_f1_stds[1]:.4f}, '
                f'Class2= {external_class_f1_means[2]:.4f} ± {external_class_f1_stds[2]:.4f}\n')
