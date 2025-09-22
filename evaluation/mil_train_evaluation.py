import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.abmil import ABMIL
from models.transmil import TransMIL
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, accuracy_score, precision_score, recall_score, f1_score
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import argparse
    
# --------------------------------
# Dataset
# --------------------------------
class MIL_set(Dataset):
    def __init__(self, df, num_classes=1):
        self.df = df.reset_index(drop=True)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        row = self.df.iloc[idx]
        if self.num_classes==1:
            bag_data = torch.load(row['path'])
            label = row['label']
            bag_label = np.zeros(self.num_classes)
            bag_label[0] = label
        else:
            bag_data = torch.load(row['path'])
            label = int(row['label'])
            bag_label = np.zeros(self.num_classes, dtype=np.float32)
            bag_label[label] = 1.0
            bag_label = torch.tensor(bag_label)
        return bag_data, bag_label
    
def train_step(train_loader, optimizer, criterion, model):
    model.train()
    train_loss = 0.
    for batch_idx, (data, label) in enumerate(train_loader):
        data , label = data.cuda(), label.cuda()
        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        output = model.forward(data)
        loss = criterion(output['logit'],label)
        train_loss += loss
        # backward pass
        loss.backward()
        # step
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)

    return train_loss

@torch.no_grad()
def valid_step(test_loader,criterion,model,num_classes):
    model.eval()
    test_loss = 0.
    test_pred = []
    test_label = []
    for batch_idx, (data, label) in enumerate(test_loader):
        data, label = data.cuda(), label.cuda()
        output = model.forward(data)
        loss = criterion(output['logit'],label)
        test_loss+= loss
        if num_classes == 1:
            test_pred.append(output['logit'].sigmoid().round().cpu().detach().numpy().squeeze())
            test_label.append(label.cpu().detach().numpy().squeeze())
        else:
            pred_class = torch.argmax(output['logit'], dim=1)  
            test_pred.append(pred_class.cpu().detach().numpy().squeeze())
            true_class = torch.argmax(label, dim=1)
            test_label.append(true_class.cpu().detach().numpy().squeeze())
    test_loss/= len(test_loader)
    return test_loss, accuracy_score(test_pred,test_label)

@torch.no_grad()
def test_step(test_loader,criterion,model,num_classes):
    model.eval()
    test_loss = 0.
    test_pred = []
    test_label = []
    for batch_idx, (data, label) in enumerate(test_loader):
        data, label = data.cuda(), label.cuda()
        output = model.forward(data)
        loss = criterion(output['logit'],label)
        test_loss+= loss
        if num_classes == 1:
            test_pred.append(output['logit'].sigmoid().round().cpu().detach().numpy().squeeze())
            test_label.append(label.cpu().detach().numpy().squeeze())
        else:
            pred_class = torch.argmax(output['logit'], dim=1)  
            test_pred.append(pred_class.cpu().detach().numpy().squeeze())
            true_class = torch.argmax(label, dim=1)
            test_label.append(true_class.cpu().detach().numpy().squeeze())
    test_loss/= len(test_loader)
    return test_pred, test_label, accuracy_score(test_pred,test_label), precision_score(test_pred,test_label, average='weighted', zero_division=0), recall_score(test_pred,test_label, average='weighted', zero_division=0), f1_score(test_pred,test_label, average='weighted', zero_division=0), precision_score(test_pred,test_label, average='macro', zero_division=0), recall_score(test_pred,test_label, average='macro', zero_division=0), f1_score(test_pred,test_label, average='macro', zero_division=0)

def train_model(model_name, df, num_classes, epochs, best_loss, save_path):
    print('Init Model')
    if model_name == 'abmil':
        model = ABMIL(use_gate=True, dim_in=2048, dim_hid=256, num_classes=num_classes)
    elif model_name == 'transmil':
        model = TransMIL(n_classes=num_classes, dropout=False, act='relu')
    else:
        raise ValueError('Invalid model name. Choose from "abmil", "dsmil", or "transmil".')
    
    if torch.cuda.is_available():
        model.cuda()
    
    print('Init Objective Function')
    if num_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    print('Init Optimizer')
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad==True], lr=1e-4, betas=(0.9, 0.999), weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    train_df = df.query('stage=="train"')
    valid_df = df.query('stage=="valid"')
    print(f'Train: {len(train_df)}, Valid: {len(valid_df)}')
    train_set,valid_set, = MIL_set(train_df, num_classes), MIL_set(valid_df, num_classes)
    train_loader = DataLoader(train_set,shuffle=True)
    valid_loader = DataLoader(valid_set)
    print(len(train_set),len(valid_set))

    pbar = tqdm(range(epochs), total=epochs)
    for epoch in pbar:
        pbar.set_description(f'Epoch: {epoch}')
        train_loss = train_step(train_loader, optimizer, criterion, model)
        valid_loss, val_acc = valid_step(valid_loader, criterion, model, num_classes)
        pbar.set_postfix(train_loss=train_loss.item(), valid_loss=valid_loss.item(), val_acc=f'{val_acc:.4f}')

        scheduler.step(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), save_path)

def get_results(model_name, num_classes, model_path, test_df):
    print('Init Objective Function')
    if num_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    test_set = MIL_set(test_df, num_classes)
    test_loader = DataLoader(test_set)

    if model_name == 'abmil':
        model = ABMIL(use_gate=True, dim_in=2048, dim_hid=256, num_classes=num_classes)
    elif model_name == 'transmil':
        model = TransMIL(n_classes=num_classes, dropout=False, act='relu')
    else:
        raise ValueError('Invalid model name. Choose from "abmil", "dsmil", or "transmil".')
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    int_pred, int_label, int_accuracy, int_precision, int_recall, int_f1, int_mprecision , int_mrecall, int_mf1  = test_step(test_loader, criterion, model, num_classes)
    return int_pred, int_label, int_accuracy, int_precision, int_recall, int_f1, int_mprecision , int_mrecall, int_mf1

def calculate_class_f1_scores(pred, label):
    """calculate f1 score for each class"""
    f1_scores = f1_score(label, pred, average=None, labels=[0, 1, 2], zero_division=0)
    return f1_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, choices=['abmil', 'transmil'], type=str, help='Model name: abmil, transmil')
    parser.add_argument('-tcp', '--train_csv_path', required=True, type=str, help='Path to the CSV file containing data paths and labels')
    parser.add_argument('-icp', '--int_csv_path', required=True, type=str, help='Path to the CSV file containing data paths and labels')
    parser.add_argument('-ecp', '--ext_csv_path', required=True, type=str, help='Path to the CSV file containing data paths and labels')
    parser.add_argument('-msp', '--model_save_path', required=True, type=str, help='Path to the CSV file containing data paths and labels')
    parser.add_argument('-rsp', '--results_save_path', required=True, type=str, help='Path to the CSV file containing data paths and labels')
    parser.add_argument('--epochs', required=False, type=int, default=50, help='Number of training epochs') 
    parser.add_argument('--best_loss', required=False, type=float, default=1.0, help='Best loss for early stopping')
    parser.add_argument('--seed', required=False, type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_classes', required=False, type=int, default=3, help='Number of classes for classification')

    args = parser.parse_args()
   
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        print('\nGPU is ON!')

    # train for 4 folds
    for i in range(1,5):
        method = args.model_save_path.split('/')[-1].split('resnet50')[-1].split('.pth')[0]
        re_method = method + f'{i}' if i != 1  else method 
        model_path = args.model_save_path.replace(method, re_method)
        df = pd.read_csv(args.train_csv_path.replace(method, re_method))
        train_model(args.model, df, args.num_classes, args.epochs, args.best_loss, model_path)

    internal_accuracies = []
    internal_precisions = []
    internal_recalls = []
    internal_f1s = []
    internal_class_f1s = []

    external_accuracies = []
    external_precisions = []
    external_recalls = []
    external_f1s = []
    external_class_f1s = []

    # evaluate for 4 folds
    for i in range(1,5):
        method = args.model_save_path.split('/')[-1].split('resnet50')[-1].split('.pth')[0]
        re_method = method + f'{i}' if i != 1  else method 
        model_path = args.model_save_path.replace(method, re_method)
        int_df = pd.read_csv(args.int_csv_path.replace(method, re_method))
        ext_df = pd.read_csv(args.ext_csv_path.replace(method, re_method))
        int_pred, int_label, int_accuracy, int_precision, int_recall, int_f1, int_mprecision , int_mrecall, int_mf1, ext_pred, ext_label, ext_accuracy, ext_precision, ext_recall, ext_f1, ext_mprecision , ext_mrecall, ext_mf1 = get_results(args.model, args.num_classes, model_path, int_df, ext_df)
        internal_class_f1 = calculate_class_f1_scores(int_pred, int_label)
        external_class_f1 = calculate_class_f1_scores(ext_pred, ext_label)

        internal_accuracies.append(int_accuracy)
        internal_precisions.append(int_precision)
        internal_recalls.append(int_recall)
        internal_f1s.append(int_f1)
        external_accuracies.append(ext_accuracy)
        external_precisions.append(ext_precision)
        external_recalls.append(ext_recall)
        external_f1s.append(ext_f1)
        internal_class_f1s.append(internal_class_f1)
        external_class_f1s.append(external_class_f1)

    internal_class_f1_means = np.mean(internal_class_f1s, axis=0)
    internal_class_f1_stds = np.std(internal_class_f1s, axis=0)
    external_class_f1_means = np.mean(external_class_f1s, axis=0)
    external_class_f1_stds = np.std(external_class_f1s, axis=0)

    with open(args.results_save_path, 'w') as f:
        ## internal test
        f.write(f'[Internal Test]\n')
        for i in range(1,5):
            num = f' {i}' if i != 1  else '' 
            f.write(f'Accuracy{num}: {internal_accuracies[i-1]:.4f}, Precision{num}: {internal_precisions[i-1]:.4f}, Recall{num}: {internal_recalls[i-1]:.4f}, F1-score{num}: {internal_f1s[i-1]:.4f}\n')
            f.write(f'Class F1-scores{num}: Class0={internal_class_f1s[i-1][0]:.4f}, Class1={internal_class_f1s[i-1][1]:.4f}, Class2={internal_class_f1s[i-1][2]:.4f}\n\n')
        
        # calculate mean and std
        f.write(f'Avg Accuracy: {np.mean(internal_accuracies):.4f} ± {np.std(internal_accuracies):.4f}, Avg Precision: {np.mean(internal_precisions):.4f} ± {np.std(internal_precisions):.4f}, Avg Recall: {np.mean(internal_recalls):.4f} ± {np.std(internal_recalls):.4f}, Avg F1-score: {np.mean(internal_f1s):.4f} ± {np.std(internal_f1s):.4f}\n')
        f.write(f'Avg Class F1-scores: Class0={internal_class_f1_means[0]:.4f}±{internal_class_f1_stds[0]:.4f}, Class1={internal_class_f1_means[1]:.4f}±{internal_class_f1_stds[1]:.4f}, Class2={internal_class_f1_means[2]:.4f}±{internal_class_f1_stds[2]:.4f}\n\n')
        
        ## external test
        f.write(f'[External Test]\n')
        for i in range(1,5):
            num = f'{i}' if i != 1  else '' 
            f.write(f'Accuracy{num}: {external_accuracies[i-1]:.4f}, Precision{num}: {external_precisions[i-1]:.4f}, Recall{num}: {external_recalls[i-1]:.4f}, F1-score{num}: {external_recalls[i-1]:.4f}\n')
            f.write(f'Class F1-scores{num}: Class0={external_class_f1s[i-1][0]:.4f}, Class1={external_class_f1s[i-1][1]:.4f}, Class2={external_class_f1s[i-1][2]:.4f}\n\n')
    
        # calculate mean and std
        f.write(f'Avg Accuracy: {np.mean(external_accuracies):.4f} ± {np.std(external_accuracies):.4f}, Avg Precision: {np.mean(external_precisions):.4f} ± {np.std(external_precisions):.4f}, Avg Recall: {np.mean(external_recalls):.4f} ± {np.std(external_recalls):.4f}, Avg F1-score: {np.mean(external_f1s):.4f} ± {np.std(external_f1s):.4f}\n')
        f.write(f'Avg Class F1-scores: Class0={external_class_f1_means[0]:.4f}±{external_class_f1_stds[0]:.4f}, Class1={external_class_f1_means[1]:.4f}±{external_class_f1_stds[1]:.4f}, Class2={external_class_f1_means[2]:.4f}±{external_class_f1_stds[2]:.4f}\n')