from albumentations.pytorch                  import ToTensorV2
from openslide.deepzoom                      import DeepZoomGenerator
from torch.utils.data                        import Dataset
from skimage                                 import filters
from tqdm                                    import tqdm
import albumentations                        as A
import pandas                                as pd
import numpy                                 as np
import openslide
import cv2
import os

class SetWSI(Dataset):
    def __init__(self, slide_path, target_mpp, target_size, overlap=0, return_loc=False, shuffle=False, mode='inference', save_true=False, save_dir=''):
        self.slide_path      = slide_path
        self.slide_name      = '.'.join(os.path.split(self.slide_path)[-1].split('.')[:-1])
        self.target_mpp      = target_mpp
        self.target_size     = target_size
        self.slide           = openslide.open_slide(self.slide_path)
        self.overlap         = 1-overlap
        self.dzi_size        = int(self.target_size*self.overlap)
        self.dzi_overlap     = int((self.target_size-self.dzi_size)//2)
        self.dzi             = DeepZoomGenerator(self.slide,tile_size=self.dzi_size,overlap=self.dzi_overlap)
        self.rgb_min         = 255*.25
        self.rgb_max         = 255*.9
        self.transform       = A.Compose([A.ToFloat(),ToTensorV2()])
        self.shuffle         = shuffle
        self.return_loc      = return_loc
        self.mode = mode
        self.save_true = save_true
        self.save_dir = save_dir
        if self.save_true:
            os.makedirs(save_dir, exist_ok=True)  
        self._init_property()
        self._init_thumbnail()
        self._init_grid()
    
    def _init_property(self):
        try:
            mpp = float(f'{float(self.slide.properties.get("openslide.mpp-x")):.2f}')
        except:
            mpp = .25
        self.target_downsample = int(self.target_mpp/mpp)
        self.target_dim = tuple(x//self.target_downsample for x in self.slide.level_dimensions[0])
        return self.target_downsample
    
    @staticmethod
    def get_tissue_mask(rgb_image):
        hsv = cv2.cvtColor(rgb_image,cv2.COLOR_RGB2HSV)
        ret = hsv[:, :, 1] > filters.threshold_otsu(hsv[:, :, 1])
        ret = np.array(ret).astype(np.uint8)
        return ret

    @staticmethod
    def get_rgb_val(image, location):
        y_s = location[0]; x_s = location[1]
        return np.mean(image[y_s:y_s+1, x_s:x_s+1])

    def _init_thumbnail(self):
        self.thumbnail = np.array(self.slide.get_thumbnail(
            (self.target_dim[0]//(self.dzi_size), self.target_dim[1]/(self.dzi_size))
        ).convert("RGB"))
        self.thumbnail_mask = self.get_tissue_mask(self.thumbnail)


    def _init_grid(self):
        binary = self.thumbnail_mask>0
        try:
            h,w = self.thumbnail_mask.shape
            for i in range(len(self.dzi.level_tiles)):
                _, tile_h = self.dzi.level_tiles[i]
                if (h/tile_h) > .85 and (h/tile_h) < 1.15:
                    self.dzi_lv = i
        except:
            self.dzi_lv = -1

        self.df = pd.DataFrame(pd.DataFrame(binary).stack())
        self.df['is_tissue'] = self.df[0]
        self.df.drop(0, axis=1, inplace=True)
        self.df['slide_path'] = self.slide_path
        self.df.query('is_tissue==True', inplace=True)
        self.df.drop(columns=['is_tissue'], inplace=True)
        self.tile_loc_list = [x[::-1] for x in list(self.df.index)]
        self.df.reset_index(inplace=True, drop=True)
        if self.shuffle==True:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.tile_loc_list)


    def __getitem__(self, idx):
        tile_loc = self.tile_loc_list[idx]
        tile = np.array(self.dzi.get_tile(self.dzi_lv, tile_loc).convert("RGB")).astype(np.uint8)
        tile = self.transform(image=tile)['image']
        
        if self.save_true:
            cv2.imwrite(f'{self.save_dir}/{self.slide_name}_{tile_loc[0]}_{tile_loc[1]}.png', np.array(tile.permute(1,2,0))*255)

        if self.return_loc is True:
            return (tile, str(tile_loc))
        
        else:
            return tile