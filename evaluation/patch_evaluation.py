import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.model                                            import resnet50, swin_transformer_tiny, se_resnext101_32x4d
from utils.functional                                        import Accuracy, Fscore, EvaluationEpoch
from utils.datagen                                           import test_dataloader_setting
import torch.nn                                              as nn
import segmentation_models_pytorch                           as smp
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


def test_dataset(model_name, model_save_path, test_path, classes, batch_size, loss):
    # dataloader setting
    print('########')
    print(test_path)
    print('########')
    test_loader = test_dataloader_setting(test_path, batch_size)
    os.makedirs('/'.join(model_save_path.split('/')[:-1]), exist_ok = True)

    # loss, metrics, optimizer and schduler setting
    loss = getattr(nn, loss)() if loss == 'CrossEntropyLoss' else getattr(smp.losses, loss)(mode='multilabel')
    metrics = [Accuracy(classes), Fscore(classes,0), Fscore(classes,1), Fscore(classes,2), Fscore(classes,3)]
    model = model_setting(model_name)
    model.load_state_dict(torch.load(model_save_path), strict=True)
    model.eval()

    test_epoch = EvaluationEpoch(
        model=model,
        classes = classes,
        loss=loss,
        metrics=metrics,
        verbose=True,
    )
    results = test_epoch.run_memory_efficient(test_loader, chunk_size=5000)
    return results

def save_simple_results(results_list, result_save_path):
    keys = results_list[0].keys()
    
    # calculate mean and standrd deviation for each metric
    averages = {}
    std_devs = {}
    
    for k in keys:
        values = [r[k] for r in results_list]
        averages[k] = sum(values) / len(values)
        # calculate standard deviation
        if len(values) > 1:
            mean = averages[k]
            variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
            std_devs[k] = variance ** 0.5
        else:
            std_devs[k] = 0.0
    
    with open(result_save_path, 'w') as f:
        # results for each fold
        for i, result in enumerate(results_list, 1):
            f.write(f"=== Fold {i} Results ===\n")
            for k, v in result.items():
                f.write(f'{k}: {v:.4f}\n')
            f.write('\n')
        
        # average and std deviation
        f.write("=== 4-Fold Average Results ===\n")
        for k in keys:
            f.write(f'{k}: {averages[k]:.4f} ± {std_devs[k]:.4f}\n')

    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, choices=['resnet50', 'swin_t', 'se_resnext101'], type=str, help='Model name: resnet50, swin_t, se_resnext101')
    parser.add_argument('-msp', '--model_save_path', required=True, help='Path to model')
    parser.add_argument('-tep', '--test_pickle_path', required=True, help='Path to test pickle')
    parser.add_argument('-rsp', '--result_save_path', required=True, help='Path to resutls')
    parser.add_argument('--classes', required=False, default=4, help='Classes')
    parser.add_argument('--batch_size', required=False, default=16, help='Batch Size')
    parser.add_argument('--loss', required=False, default='CrossEntropyLoss', help='Loss')
    args = parser.parse_args()

    results_list = []
    for i in range(1,5):
        if i != 1:
            model_save_path = args.model_save_path.replace('.pth', f'{i}.pth')
        else:
            model_save_path = args.model_save_path
        results = test_dataset(args.model, model_save_path, args.test_pickle_path, args.classes, args.batch_size, args.loss)
        results_list.append(results)

    save_simple_results(results_list, args.result_save_path)
