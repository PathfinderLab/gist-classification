from models.model                                            import resnet50, swin_transformer_tiny, se_resnext101_32x4d
from utils.functional                                        import Accuracy, Fscore, EarlyStopping, TrainEpoch, ValidEpoch
from utils.datagen                                           import dataloader_setting
import torch.optim                                           as optim
import torch.nn                                              as nn
import argparse
import torch 
import os


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


def train_dataset():
    # model setting
    model = model_setting(MODEL)

    # dataloader setting
    train_loader, valid_loader, test_loader = dataloader_setting(TRAIN_PATH, VALID_PATH, TEST_PATH, NOR_SCH_PATH, NOR_LEI_PATH, LEI_SCH_PATH, CUTMIX, IMB_SAMPLER, BATCH_SIZE)

    # weight and log setting
    print('/'.join(MODEL_SAVE_PATH.split('/')[:-1]))
    os.makedirs('/'.join(MODEL_SAVE_PATH.split('/')[:-1]), exist_ok = True)

    # loss, metrics, optimizer and schduler setting
    loss = getattr(nn, LOSS)()
    optimizer = getattr(optim, OPTIMIZER)(params=model.parameters(), lr=LR)
    metrics = [Accuracy(CLASSES), Fscore(CLASSES,0), Fscore(CLASSES,1), Fscore(CLASSES,2), Fscore(CLASSES,3)]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    patience = PATIENCE
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=MODEL_SAVE_PATH)

    train_epoch = TrainEpoch(
        model, 
        classes = CLASSES,
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        scheduler=scheduler,
        verbose=True,
    )
    valid_epoch = ValidEpoch(
        model, 
        classes = CLASSES,
        loss=loss, 
        metrics=metrics, 
        verbose=True,
    )

    for i in range(0, EPOCH):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        early_stopping(valid_logs['loss'], model)
        if early_stopping.early_stop:
            print('Early stopping')
            break
    
    model = torch.load(MODEL_SAVE_PATH)

    test_epoch = ValidEpoch(
        model=model,
        classes = CLASSES,
        loss=loss,
        metrics=metrics,
        verbose=True,
    )
    test_epoch.run(test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='Model')
    parser.add_argument('-msp', '--model_save_path', required=True, help='Path to model')
    parser.add_argument('-trp', '--train_path', required=True, help='Path to train pickle')
    parser.add_argument('-vp', '--valid_path', required=True, help='Path to valid pickle')
    parser.add_argument('-tep', '--test_path', required=True, help='Path to test pickle')
    parser.add_argument('-nsp', '--normal_schwannoma_path', required=True, help='Path to normal_schwannoma pickle')
    parser.add_argument('-nlp', '--normal_leiomyoma_path', required=True, help='Path to normal_leiomyoma pickle')
    parser.add_argument('-lsp', '--leiomyoma_schwannoma_path', required=True, help='Path to leiomyoma_schwannoma pickle')
    parser.add_argument('--use_cutmix', action='store_true')
    parser.add_argument('--use_imb_sampler', action='store_true')
    parser.add_argument('--classes', required=False, default=4, help='Classes')
    parser.add_argument('--batch_size', required=False, default=16, help='Batch Size')
    parser.add_argument('--epoch', required=False, default=100, help='Epoch')
    parser.add_argument('--patient', required=False, default=4, help='Patient')
    parser.add_argument('--loss', required=False, default='CrossEntropyLoss', help='Loss')
    parser.add_argument('--learning_Rate', required=False, default=1e-4, help='Learning Rate')
    parser.add_argument('--optimizer', required=False, default='Adam', help='Optimizer')

    args = vars(parser.parse_args())
    MODEL = args['model']
    MODEL_SAVE_PATH = args['model_save_path']
    TRAIN_PATH = args['train_path']
    VALID_PATH = args['valid_path']
    TEST_PATH = args['test_path']
    NOR_SCH_PATH = args['normal_schwannoma_path']
    NOR_LEI_PATH = args['normal_leiomyoma_path']
    LEI_SCH_PATH = args['leiomyoma_schwannoma_path']
    CUTMIX = args['use_cutmix']
    IMB_SAMPLER = args['use_imb_sampler']
    CLASSES = args['classes']
    BATCH_SIZE = args['classes']
    EPOCH = args['epoch']
    PATIENCE = args['patient']
    LOSS = args['loss']
    LR = args['learning_Rate']
    OPTIMIZER = args['optimizer']

    train_dataset()