from models.model           import resnet50
from torch.utils.data       import DataLoader
from utils.util             import SetWSI
from tqdm                   import tqdm
import torch.nn             as nn
import pandas               as pd
import numpy                as np
import argparse
import openslide
import torch
import cv2
import os


class InferenceWSI(object):
    def __init__(self, model_path, target_mpp, target_size, overlap, slide_organ, save_base_dir, batch_size=32):
        self.model          = resnet50()
        self.device         = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.target_mpp     = target_mpp
        self.target_size    = target_size
        self.overlap        = overlap
        self.add_pix        = int(1/(1-self.overlap))
        self.batch_size     = batch_size
        self.softmax        = nn.Softmax() 
        self.slide_organ    = slide_organ
        self.save_base_dir  = save_base_dir
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

    def image_with_mask(self, image, mask):
        image_orig = image.copy()
        color_dict = {1:(255,0,0), 2:(0,0,255), 3:(44,44,44)}
        for i in range(3):
            mask_type    = np.where(mask==i+1,1,0).astype(np.uint8)
            mask_cont, _ = cv2.findContours(mask_type, cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_SIMPLE) 
            image        = cv2.fillPoly(image, mask_cont, color_dict[i+1])
        img_with_msk = cv2.addWeighted(src1=image_orig, alpha=0.6, src2=image, beta=0.4, gamma=0)
        return img_with_msk

    def get_predict(self, slide_path, total_df):
        slide = openslide.open_slide(slide_path)
        slide_image = np.array(slide.get_thumbnail(((1000,1000))).convert("RGB"))
        slide_name  = '.'.join(slide_path.split('/')[-1].split('.')[:-1])
        wsi_set     = SetWSI(slide_path, self.target_mpp, self.target_size, overlap=self.overlap, return_loc=True)
        wsi_loader  = DataLoader(wsi_set, batch_size=self.batch_size)

        ret0 = np.zeros((wsi_set.thumbnail_mask.shape[0]*self.target_size, wsi_set.thumbnail_mask.shape[1]*self.target_size))
        loc_x_list = []; loc_y_list = []; preds_list = []; probs_list = []
        
        for i, (batch_tiles, batch_tile_loc) in tqdm(enumerate(wsi_loader)):

            batch_probs   = self.softmax(self.model(batch_tiles.to(self.device)))
            batch_class   = torch.argmax(batch_probs, dim=1).cpu().detach().numpy().squeeze()
            batch_probs   = batch_probs.cpu().detach().numpy()

            del batch_tiles        
            torch.cuda.empty_cache()
            for i, tile_loc in enumerate(batch_tile_loc):
                try:
                    if batch_class[i] != 0:
                        loc_x_list.append(eval(tile_loc)[0])
                        loc_y_list.append(eval(tile_loc)[1])
                        preds_list.append(batch_class[i])
                        probs_list.append(max(batch_probs[i]))
                    if max(batch_probs[i]) > 0.9:
                        ret0[int(eval(tile_loc)[1]*self.target_size):int(eval(tile_loc)[1]+1)*self.target_size, int(eval(tile_loc)[0]*self.target_size):int(eval(tile_loc)[0]+1)*self.target_size] = np.ones((self.target_size,self.target_size))*batch_class[i]
                    continue   
                except:
                    continue
        
        count_dict = {'GIST':0, 'Leiomyoma':0, 'Schwannoma':0}
        for (pred, prob) in zip(preds_list, probs_list):
            if prob > 0.9:
                if pred == 1:
                    count_dict['GIST'] += 1
                elif pred == 2:
                    count_dict['Leiomyoma'] += 1
                elif pred == 3:
                    count_dict['Schwannoma'] += 1

        predict_class = max(count_dict, key=count_dict.get)
        # if the largest number is multipe ...
        if list(count_dict.values()).count(count_dict[predict_class]) >= 2:
            predict_class = 'Unclassfied'

        ret0 = cv2.resize(ret0, (slide_image.shape[1], slide_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        map_img_msk  = self.image_with_mask(slide_image.copy(), ret0)

        organ_name_list = [self.slide_organ]*len(loc_x_list)
        slide_name_list = [slide_name]*len(loc_x_list)
        df = pd.DataFrame(list(zip(organ_name_list, slide_name_list, loc_x_list, loc_y_list, preds_list, probs_list)), columns=['organ_name', 'slide_name', 'x_loc', 'y_loc', 'prediction', 'probability'])
        total_df = pd.concat([total_df, df])
        os.makedirs(f'{self.save_base_dir}/{self.slide_organ}', exist_ok=True)
        total_df.to_csv(f'{self.save_base_dir}/{self.slide_organ}/results.csv', index=False)
        os.makedirs(f'{self.save_base_dir}/{self.slide_organ}/{predict_class}', exist_ok=True)
        cv2.imwrite(f'{self.save_base_dir}/{self.slide_organ}/{predict_class}/{slide_name}.png', map_img_msk)

        return total_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data_path', required=True, help='Path to slide path csv file')
    parser.add_argument('-mp', '--model_path', required=True, help='Path to model')
    parser.add_argument('-sbd', '--save_base_dir', required=True, help='Path to base results')
    parser.add_argument('-on', '--organ_name', required=True, help='Organ name')
    parser.add_argument('-tm', '--target_mpp', required=False, default=1.0, help='Target mpp')
    parser.add_argument('-ts', '--target_size', required=False, default=512, help='Target patch size')
    parser.add_argument('-o', '--overlap', required=False, default=0, help='Overlap')
    parser.add_argument('-ib', '--infer_batch', required=False, default=4, help='Infer batch size')

    args = vars(parser.parse_args())
    TARGET_SLIDE = list(pd.read_csv(args['data_path'])['slide_path'])
    MODEL_PATH = args['model_path']
    SAVE_BASE_DIR = args['save_base_dir']
    ORGAN_NAME = args['organ_name']
    TARGET_MPP = args['target_mpp']
    TARGET_SIZE = args['target_size']
    OVERLAP = args['overlap']
    INFER_BATCH = args['infer_batch']

    error_list = []
    total_df = pd.DataFrame(columns=['organ_name', 'slide_name', 'x_loc', 'y_loc', 'prediction', 'probability'])

    for slide_path in TARGET_SLIDE:
        clf = InferenceWSI(MODEL_PATH, TARGET_MPP, TARGET_SIZE, OVERLAP, ORGAN_NAME, SAVE_BASE_DIR, INFER_BATCH)
        total_df = clf.get_predict(slide_path, total_df)