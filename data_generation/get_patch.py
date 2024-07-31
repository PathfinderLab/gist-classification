from util import PatchProcessor
from lxml import etree
import pandas as pd
import numpy as np
import multiprocessing
import argparse
import glob
import cv2
import os

class ExtractPatch(PatchProcessor):

    # init variable setting method
    def __init__(self, patch_size, target_mpp, down_level, anno_ratio, tissue_ratio):
        super().__init__(
            patch_size  = patch_size,
            target_mpp  = target_mpp,
            down_level  = down_level           
        )
        self.anno_ratio = anno_ratio
        self.tissue_ratio  = tissue_ratio

    def variance_of_laplacian(self, image):
        return cv2.Laplacian(image, cv2.CV_64F).var()

    # extract patch from get patch 
    def execute_patch(self, patch_img, patch_count, name):
        resize_image = cv2.resize(patch_img, (self.patch_size,self.patch_size), cv2.INTER_AREA)
        self.save_image(self.save_path + f'/{self.slide_name}', f'{patch_count}_{name}.png', resize_image)


    def except_black(self):
        check_patch = np.array(self.slide.read_region((0,0), 0, size=(self.read_size, self.read_size)))[...,:3]
        try:
            min_x = int(np.where(check_patch>[0,0,0])[0][0]/self.zero2min)+1
            min_y = int(np.where(check_patch>[0,0,0])[1][0]/self.zero2min)+1
        except:
            min_x = 1
            min_y = 1
        return min_x, min_y

    def get_anno_dict1(self):
        slide_dir = self.slide_path.replace('.mrxs','')
        dat_files = glob.glob(os.path.join(slide_dir,'*.dat'))
        dat_max = max([int(x.split('Data')[-1].split('.dat')[0]) if 'Data' in x else 0 for x in dat_files])
        anno_file = glob.glob(os.path.join(slide_dir,f'*{dat_max}.dat'))[0]
        with open(anno_file,'r') as f:
            lines = f.readlines()
            ret = lines[-1]

        tree = etree.fromstring(ret)
        ret = dict(); tmp_list = []
        for i, annos in enumerate(tree.iter('SimpleBookmark')):
            caption = annos.get('Caption')
            tmp_list.append(caption)
            if caption not in ret.keys():
                ret[caption] = []
            
        for i, polys in enumerate(tree.iter('polygon_points')):
            pts = []
            for pt in polys.iter('polygon_point'):
                x = float(pt.get('polypointX'))
                y = float(pt.get('polypointY'))
                x = np.clip(round(x/self.zero2min),0,self.level_min_w)
                y = np.clip(round(y/self.zero2min),0,self.level_min_h)
                pts.append((x,y))
            ret[tmp_list[i]].append(pts)

        return ret


    def get_anno_dict2(self):
        pts_dict = dict()
        root = etree.parse(self.anno_path).getroot()
        
        for trees in root:
            for tree in trees:
                for region in tree.findall('Region'):
                    anno_type = region.get('NegativeROA')
                    if (anno_type == '0') or (anno_type == '1'):
                        pts_list = []
                        for vertices in region.findall('Vertices'):
                            pts = list()
                            for vertex in vertices:
                                x = float(vertex.get('X'))
                                y = float(vertex.get('Y'))
                                x = np.clip(round(x/self.zero2min),0,self.level_min_w)
                                y = np.clip(round(y/self.zero2min),0,self.level_min_h)
                                pts.append((x,y))
                        pts_list.append(pts)
                        if anno_type == '0':
                            pts_dict['Tumor'] = pts_list
                        elif anno_type == '1':
                            pts_dict['Normal'] = pts_list

        return pts_dict


    def get_anno_mask(self):
        mask = np.zeros((self.level_min_h,self.level_min_w))
        anno_dict = self.get_anno_dict1() if self.slide_path[-4:] == 'mrxs' else self.get_anno_dict2() 
        anno_num = {'Tumor':2,'Normal':1}
        for anno in anno_dict.keys():
            if anno == 'Tumor' or anno == 'Normal':
                for region in anno_dict[anno]:
                    point = [np.array(region, dtype=np.int32)]
                    mask = cv2.fillPoly(mask, point, anno_num[anno])

        return mask


    # extract patches corresponding with annoation mask method
    def extract(self):
        tissue_mask = self.get_tissue_mask()
        tumor_mask = self.get_anno_mask()
        step = 1
        patch_count = 0
        blur_count = 0

        min_x, min_y = self.except_black() 
        slide_w=self.level_min_w; slide_h=self.level_min_h
        y_seq, x_seq = self.get_seq_range(slide_w, slide_h, self.read_size, self.zero2min)

        for y in y_seq:
            for x in x_seq:
                start_x = int(min_x + int(self.read_size/self.zero2min)*x)
                end_x = int(min_x + int(self.read_size/self.zero2min)*(x+step))
                start_y = int(min_y + int(self.read_size/self.zero2min)*y)
                end_y = int(min_y+ int(self.read_size/self.zero2min)*(y+step))

                tissue_mask_patch = tissue_mask[start_y:end_y, start_x:end_x]
                tumor_patch = tumor_mask[start_y:end_y, start_x:end_x]
                if (self.get_ratio_mask(tissue_mask_patch) >= self.tissue_ratio) and (self.get_ratio_mask(tumor_patch) >= self.anno_ratio):
                
                    img_patch = np.array(self.slide.read_region(
                        location = (int(start_x*self.zero2min), int(start_y*self.zero2min)),
                        level = 0,
                        size = (self.read_size, self.read_size)
                    )).astype(np.uint8)[...,:3]

                    if self.variance_of_laplacian(img_patch) >= 100:
                        patch_count += 1 
                        name = str(int(np.max(tumor_patch)))

                        self.execute_patch(img_patch, patch_count, name)
                    else:
                        blur_count += 1
                        self.save_image(self.save_path + f'/blur/{self.slide_name}', f'{blur_count}.png', img_patch)





def extract(main, slide_path, save_path):
    anno_path = '.'.join(slide_path.split('.')[:-1]) + 'xml'
    main._slide_setting(slide_path, anno_path, save_path)
    main.extract()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data_path', required=True, help='Path to slide path csv file')
    parser.add_argument('-sp', '--save_path', required=True, help='Path to generated patches')
    parser.add_argument('-ps', '--patch_size', required=False, default=512, help='Patch size')
    parser.add_argument('-tm', '--target_mpp', required=False, default=1.0, help='Target mpp')
    parser.add_argument('-dl', '--down_level', required=False, default=-1, help='Down sized level')
    parser.add_argument('-ar', '--anno_ratio', required=False, default=0.5, help='Annotation ratio')
    parser.add_argument('-tr', '--tissue_ratio', required=False, default=0.3, help='Tissue ratio')
    parser.add_argument('-b', '--batch', required=False, default=20, help='Slide batch size')

    args = vars(parser.parse_args())

    SLIDE_LIST = list(pd.read_csv(args['data_path'])['slide_path'])    
    SAVE_PATH = args['save_path']
    PATCH_SIZE = args['patch_size']
    TARGET_MPP = args['target_mpp']
    DOWN_LEVEL = args['down_level']
    ANNO_RATIO = args['anno_ratio']
    TISSUE_RATIO = args['tissue_ratio']
    BATCH = args['batch']

    # main setting
    main = ExtractPatch(PATCH_SIZE, TARGET_MPP, DOWN_LEVEL, ANNO_RATIO, TISSUE_RATIO)

    # multi processing
    for i in range(0,len(SLIDE_LIST),BATCH):
        slide_batch = SLIDE_LIST[i:i+BATCH]
        for slide_path in slide_batch:
            p = multiprocessing.Process(target=extract, args=(main, slide_path, SAVE_PATH, ))
            p.start()
        p.join()
