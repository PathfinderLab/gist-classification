from skimage                                 import morphology, filters
from tqdm                                    import trange
import numpy                                 as np
import openslide
import cv2
import os


class PatchProcessor(object):
    def __init__(self, patch_size, target_mpp, down_level):
        self.patch_size = patch_size
        self.target_mpp = target_mpp
        self.down_level = down_level

    def _slide_setting(self, slide_path, anno_path, save_path):
        self.slide_path = slide_path
        self.anno_path  = anno_path
        self.save_path  = save_path

        self.slide = openslide.OpenSlide(self.slide_path)
        self.slide_name = '.'.join(self.slide_path.split('/')[-1].split('.')[:-1])

        self.level0_w, self.level0_h = self.slide.level_dimensions[0]
        self.adjust_term = self.target_mpp / float(self.slide.properties.get('openslide.mpp-x'))
        self.read_size = int(self.patch_size*self.adjust_term)

        self.level_min_w, self.level_min_h = self.slide.level_dimensions[self.down_level]
        self.level_min_img = np.array(self.slide.read_region((0,0), self.down_level if self.down_level > 0 else self.slide.level_count-1, size=(self.level_min_w, self.level_min_h)))[...,:3]
        self.zero2min = self.level0_h // self.level_min_h

    # get tissue mask method
    def get_tissue_mask(self):
        level_min_w, level_min_h = self.slide.level_dimensions[-1]
        level_min_img = np.array(self.slide.read_region((0,0), self.slide.level_count-1, size=(level_min_w, level_min_h)))
        hsv = cv2.cvtColor(level_min_img, cv2.COLOR_RGB2HSV)

        mask = hsv[:, :, 1] > filters.threshold_otsu(hsv[:, :, 1])
        ret = morphology.remove_small_holes(mask, area_threshold=(level_min_h*level_min_w)//8)
        ret = np.array(ret).astype(np.uint8)
        
        kernel_size = 5
        tissue_mask = cv2.morphologyEx(ret*255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size)))  
        tissue_mask = cv2.resize(tissue_mask, (self.level_min_w, self.level_min_h), cv2.INTER_NEAREST)
        return tissue_mask

    # get sequence ratio method
    def get_seq_range(self, slide_width, slide_height, read_size, zero2two):
        y_seq = trange(int(((slide_height)) // int(read_size/zero2two)) + 1)
        x_seq = range(int(((slide_width)) // int(read_size/zero2two)) + 1)
        return y_seq, x_seq

    # get ratio mask method
    def get_ratio_mask(self, patch):
        h_, w_ = patch.shape[0], patch.shape[1]
        n_total = h_*w_
        n_cell = np.count_nonzero(patch)
        if (n_cell != 0):
            return n_cell*1.0/n_total*1.0
        else:
            return 0

    # save patch method
    def save_image(self, dir_path, file_name, img):
        os.makedirs(dir_path, exist_ok = True)
        cv2.imwrite(os.path.join(dir_path, file_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))