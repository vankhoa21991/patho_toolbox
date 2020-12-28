import pathlib
import os
class img_proc_cfg():
    def __init__(self):
        self.sz = 512  # the size of tiles
        self.reduce = 4  # reduce the original images by 4 times

        self.THRESHOLD = 0.4
        self.WINDOW = 1024
        self.MIN_OVERLAP = 300



class path_cfg():
    def __init__(self):
        self.MASKS = pathlib.Path.home() / 'datasets' / 'Hubmap'/'hubmap-kidney-segmentation' / 'train.csv'

        self.CODE_DIR = pathlib.Path.home() / 'code' / 'kaggle'
        self.RESULTS_DIR = os.path.join(self.CODE_DIR, 'results')

        self.DATA_HOME_DIR = pathlib.Path.home() / 'datasets' / 'Hubmap'
        self.DATA_DIR = os.path.join(self.DATA_HOME_DIR, 'hubmap-kidney-segmentation')
        self.DATA_TRAIN_DIR = os.path.join(self.DATA_DIR, 'train')
        self.OUT_TRAIN = os.path.join(self.DATA_DIR, 'train_512.zip')
        self.OUT_MASKS = os.path.join(self.DATA_DIR, 'masks_512.zip')

        self.DATA_512_DIR = os.path.join(self.DATA_HOME_DIR, 'hubmap-512x512-full-size-tiles')
        self.DATA_512_IMG_DIR = os.path.join(self.DATA_512_DIR, 'train')
        self.DATA_512_MASK_DIR = os.path.join(self.DATA_512_DIR, 'masks')

        self.DF_IMG_DATA_512 = os.path.join(self.DATA_512_DIR, 'df_img.csv')

        self.DATA_512_AUG_DIR = os.path.join(self.DATA_HOME_DIR, 'hubmap_512x512_augmented')
        self.DATA_512_AUG_IMG_DIR = os.path.join(self.DATA_512_AUG_DIR, 'images_aug2')
        self.DATA_512_AUG_MASK_DIR = os.path.join(self.DATA_512_AUG_DIR, 'masks_aug2')

        self.DATA_TFRECORD = os.path.join(self.DATA_HOME_DIR, 'tfrecord')

class train_cfg():
    def __init__(self):
        self.seed = 21

    @classmethod
    def get_seed(cls):
        return cls()