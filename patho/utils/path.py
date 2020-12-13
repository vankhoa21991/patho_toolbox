import pathlib
import os
class img_proc_cfg():
    def __init__(self):
        self.sz = 512  # the size of tiles
        self.reduce = 4  # reduce the original images by 4 times


class path_cfg():
    def __init__(self):
        self.MASKS = pathlib.Path.home() / 'datasets' / 'Hubmap'/'hubmap-kidney-segmentation' / 'train.csv'
        self.DATA_HOME_DIR = pathlib.Path.home() / 'datasets' / 'Hubmap'
        self.DATA_DIR = os.path.join(self.DATA_HOME_DIR, 'hubmap-kidney-segmentation')
        self.DATA_TRAIN_DIR = os.path.join(self.DATA_DIR, 'train')
        self.CODE_DIR = pathlib.Path.home() / 'code' / 'kaggle'
        self.RESULTS_DIR = os.path.join(self.CODE_DIR, 'results')
        self.OUT_TRAIN = os.path.join(self.DATA_DIR, 'train_512.zip')
        self.OUT_MASKS = os.path.join(self.DATA_DIR, 'masks_512.zip')

        self.DATA_512_DIR = os.path.join(self.DATA_HOME_DIR, 'hubmap-512x512-full-size-tiles')
        self.DATA_512_IMG_DIR = os.path.join(self.DATA_512_DIR, 'train')
        self.DATA_512_MASK_DIR = os.path.join(self.DATA_512_DIR, 'masks')