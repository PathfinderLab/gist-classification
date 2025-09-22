from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
import argparse
import random
import pickle
import math
import cv2
import os

def get_clip_box(image_a, image_b):
    # image.shape = (height, width, channel)
    image_size_x = image_a.shape[1]
    image_size_y = image_a.shape[0]

    # get center of box
    x = random.randrange(1,image_size_x+1)
    y = random.randrange(1,image_size_y+1)

    # get width, height of box
    width = int(512*math.sqrt(1-random.random()))
    height = int(512*math.sqrt(1-random.random()))

    # clip box in image and get minmax bbox
    xa = max(0, x-width//2)
    ya = max(0, y-height//2)
    xb = min(image_size_x, x+width//2)
    yb = min(image_size_y, y+width//2)
    return xa, ya, xb, yb

def mix_2_images(image_a, image_b, xa, ya, xb, yb):
    image_size_x = image_a.shape[1]
    image_size_y = image_a.shape[0] 
    one = image_a[ya:yb,0:xa,:]
    two = image_b[ya:yb,xa:xb,:]
    three = image_a[ya:yb,xb:image_size_x,:]
    middle = np.concatenate([one,two,three],axis=1)
    top = image_a[0:ya,:,:]
    bottom = image_a[yb:image_size_y,:,:]
    mixed_img = np.concatenate([top, middle, bottom])
    return mixed_img

def mix_2_label(image_a, image_b, label_a, label_b, xa, ya, xb, yb):
    image_size_x = image_a.shape[1]
    image_size_y = image_a.shape[0] 
    mixed_area = (xb-xa)*(yb-ya)
    total_area = image_size_x*image_size_y
    a = mixed_area/total_area

    mixed_label = (1-a)*label_a + a*label_b #
    return mixed_label[0]

def make_label(path):
    if path[-5] == '1': 
        return F.one_hot(torch.Tensor([0]).to(torch.int64), 4)
    elif path.split('/')[-3] == 'Gist':
        return F.one_hot(torch.Tensor([1]).to(torch.int64), 4)
    elif path.split('/')[-3] == 'Leiomyoma':
        return F.one_hot(torch.Tensor([2]).to(torch.int64), 4)
    elif path.split('/')[-3] == 'Schwannoma':
        return F.one_hot(torch.Tensor([3]).to(torch.int64), 4)
    else:
        raise Exception('check your path')
        
def cutmix(a_image, b_image, a_label, b_label):
    xa, ya, xb, yb = get_clip_box(a_image, b_image)
    cutmix_image = mix_2_images(a_image, b_image, xa, ya, xb, yb)
    cutmix_label = mix_2_label(a_image, b_image, a_label, b_label, xa, ya, xb, yb)    
    return cutmix_image, cutmix_label    
    
def mixup(a_image, b_image, a_label, b_label, alpha):
    mixup_image = cv2.addWeighted(a_image, alpha, b_image, (1-alpha), 0)
    mixup_label = alpha*a_label + (1-alpha)*b_label 
    return mixup_image, mixup_label[0]

def apply_augpatch(i, a_list, b_list, cutmix_dir, mixed_name, cutmix_tensor_zip):
    a_random = random.randrange(len(a_list))
    b_random = random.randrange(len(b_list))
    a_path   = a_list[a_random]
    b_path   = b_list[b_random]
    a_image  = cv2.cvtColor(cv2.imread(a_path), cv2.COLOR_BGR2RGB)
    b_image  = cv2.cvtColor(cv2.imread(b_path), cv2.COLOR_BGR2RGB)
    a, b     = mixed_name.split('_')
    a_label  = make_label(a)
    b_label  = make_label(b)

    cutmix_image, cutmix_label = cutmix(a_image, b_image, a_label, b_label)
    cutmix_image =  cv2.cvtColor(cutmix_image, cv2.COLOR_BGR2RGB)
    
    os.makedirs(f'{cutmix_dir}/{mixed_name}', exist_ok=True)
    cutmix_path = f'{cutmix_dir}/{mixed_name}/{i}.png'
    cv2.imwrite(cutmix_path, cutmix_image)
    cutmix_tensor_zip.append((cutmix_path, cutmix_label)) 

    return cutmix_tensor_zip


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data_path', required=True, help='Path to patch csv file')
    parser.add_argument('-cd', '--cutmix_dir', required=True, help='Path to generated cutmix patches')
    parser.add_argument('-nns', '--num_nor_sch', required=False, default=20000, help='Number of normal-schwannoma')
    parser.add_argument('-nnl', '--num_nor_lei', required=False, default=20000, help='Number of normal-leiomyoma')
    parser.add_argument('-nls', '--num_lei_sch', required=False, default=20000, help='Number of leiomyoma-schwannoma')

    args = parser.parse_args()
    data_csv = pd.read_csv(args.data_path)
    normal_list = list(data_csv['normal_path'])
    leiomyoma_list = list(data_csv['leiomyoma_path'])
    schwannoma_list = list(data_csv['schwannoma_path'])

    # normal - schwannoma
    cutmix_tensor_zip1 = []
    for i in tqdm(range(args.num_nor_sch)):
        cutmix_tensor_zip1 = apply_augpatch(i, normal_list, schwannoma_list, args.cutmix_dir, 'normal_schwannoma', cutmix_tensor_zip1)
    with open(f'{args.cutmix_dir}/normal_schwannoma.pickle', 'wb') as fw:
        pickle.dump(cutmix_tensor_zip1, fw)

    # normal - leiomyoma
    cutmix_tensor_zip2 = []
    for i in tqdm(range(args.num_nor_lei)):
        cutmix_tensor_zip2 = apply_augpatch(i, normal_list, leiomyoma_list, args.cutmix_dir, 'normal_leiomyoma', cutmix_tensor_zip2)
    with open(f'{args.cutmix_dir}/normal_leiomyoma.pickle', 'wb') as fw:
        pickle.dump(cutmix_tensor_zip2, fw)

    # leiomyoma - schwannoma
    cutmix_tensor_zip3 = []
    for i in tqdm(range(args.num_lei_sch)):
        cutmix_tensor_zip3 = apply_augpatch(i, leiomyoma_list, schwannoma_list, args.cutmix_dir, 'leiomyoma_schwannoma', cutmix_tensor_zip3)
    with open(f'{args.cutmix_dir}/leiomyoma_schwannoma.pickle', 'wb') as fw:
        pickle.dump(cutmix_tensor_zip3, fw)