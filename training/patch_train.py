import warnings
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
warnings.filterwarnings('ignore')

from models.model                     import resnet50, swin_transformer_tiny, se_resnext101_32x4d
from utils                            import Accuracy, Fscore, EarlyStopping, TrainEpoch, ValidEpoch, train_dataloader_setting
import torch.optim                    as optim
import torch.nn                       as nn
import segmentation_models_pytorch    as smp
import argparse
import torch 



def model_setting(model_name):
    if model_name == 'resnet50':
        model = resnet50()
    elif model_name == 'swin_t':
        model = swin_transformer_tiny()
    elif model_name == 'se_resnext101':
        model = se_resnext101_32x4d()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model


def train_dataset(model_name, train_path, valid_path, nor_sch_path, nor_lei_path, lei_sch_path, cutmix, imb_sampler, batch_size, augmentation, classes, optimizer, lr, epoch, patient, loss, model_save_path):
    # model setting
    model = model_setting(model_name)

    # dataloader setting
    train_loader, valid_loader = train_dataloader_setting(train_path, valid_path, nor_sch_path, nor_lei_path, lei_sch_path, cutmix, imb_sampler, batch_size, augmentation)

    # weight and log setting
    print('/'.join(model_save_path.split('/')[:-1]))
    os.makedirs('/'.join(model_save_path.split('/')[:-1]), exist_ok = True)

    # loss, metrics, optimizer and schduler setting
    loss = getattr(nn, loss)() if loss == 'CrossEntropyLoss' else getattr(smp.losses, loss)(mode='multilabel')
    optimizer = getattr(optim, optimizer)(params=model.parameters(), lr=lr)
    metrics = [Accuracy(classes), Fscore(classes,0), Fscore(classes,1), Fscore(classes,2), Fscore(classes,3)]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    early_stopping = EarlyStopping(patience=patient, verbose=True, path=model_save_path)

    train_epoch = TrainEpoch(
        model, 
        classes = classes,
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        scheduler=scheduler,
        verbose=True,
    )
    valid_epoch = ValidEpoch(
        model, 
        classes = classes,
        loss=loss, 
        metrics=metrics, 
        verbose=True,
    )

    for i in range(0, epoch):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        early_stopping(valid_logs['loss'], model)
        if early_stopping.early_stop:
            print('Early stopping')
            break
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='Model')
    parser.add_argument('-msp', '--model_save_path', required=True, help='Path to model')
    parser.add_argument('-trp', '--train_path', required=True, help='Path to train pickle')
    parser.add_argument('-vp', '--valid_path', required=True, help='Path to valid pickle')
    parser.add_argument('-nsp', '--normal_schwannoma_path', required=False, default='', help='Path to normal_schwannoma pickle')
    parser.add_argument('-nlp', '--normal_leiomyoma_path', required=False, default='', help='Path to normal_leiomyoma pickle')
    parser.add_argument('-lsp', '--leiomyoma_schwannoma_path', required=False, default='', help='Path to leiomyoma_schwannoma pickle')
    parser.add_argument('--use_cutmix', action='store_true')
    parser.add_argument('--use_imb_sampler', action='store_true')
    parser.add_argument('--use_augmentation', action='store_true')
    parser.add_argument('--classes', required=False, default=4, help='Classes')
    parser.add_argument('--batch_size', required=False, default=4, help='Batch Size')
    parser.add_argument('--epoch', required=False, default=100, help='Epoch')
    parser.add_argument('--patient', required=False, default=4, help='Patient')
    parser.add_argument('--loss', required=False, default='CrossEntropyLoss', help='Loss')
    parser.add_argument('--learning_rate', required=False, default=1e-4, help='Learning Rate')
    parser.add_argument('--optimizer', required=False, default='Adam', help='Optimizer')

    args = parser.parse_args()

    for i in range(1,5):
        if i != 1:
            model_save_path = args.model_save_path.replace('.pth', f'{i}.pth')
            train_path = args.train_path.replace('_zip.pickle', f'{i}_zip.pickle')
            valid_path = args.valid_path.replace('_zip.pickle', f'{i}_zip.pickle')
            nor_sch_path = args.normal_schwannoma_path.replace('.pickle', f'{i}.pickle')
            nor_lei_path = args.normal_leiomyoma_path.replace('.pickle', f'{i}.pickle')
            lei_sch_path = args.leiomyoma_schwannoma_path.replace('.pickle', f'{i}.pickle')
        else:
            model_save_path = args.model_save_path
            train_path = args.train_path
            valid_path = args.valid_path
            nor_sch_path = args.normal_schwannoma_path
            nor_lei_path = args.normal_leiomyoma_path
            lei_sch_path = args.leiomyoma_schwannoma_path
        train_dataset(args.model, train_path, valid_path, nor_sch_path, nor_lei_path, lei_sch_path, args.use_cutmix, args.use_imb_sampler, args.batch_size, args.use_augmentation, args.classes, args.optimizer, args.learning_rate, args.epoch, args.patient, args.loss, model_save_path)