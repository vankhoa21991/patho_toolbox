import glob
import matplotlib.pyplot as plt
from patho.utils.utils import read_data
from patho.cfg.config import path_cfg,img_proc_cfg
from patho.utils.plot import plot_img_and_mask

aug_dict = {
    0: "CLAHE",
    1: 'ElasticTransform',
    2: 'GridDistortion',
    3: 'OpticalDistortion',
    4: 'HorizontalFlip',
    5: 'RandomBrightnessCOntrast',
    6: 'RandomGamma',
    7: 'HueSaturationValue',
    8: 'RGBShift',
    9: 'MedianBLur',
    10: 'GaussianBlur',
    11: 'GaussNoise',
    12: 'ChanneslShuffle',
    13: 'CoarseDropout',
}

if __name__ == '__main__':
    tileProcCfg = img_proc_cfg()
    path_cfg = path_cfg()

    aug_img_paths = glob.glob(path_cfg.DATA_512_AUG_IMG_DIR + "/*.png")
    aug_msk_paths = glob.glob(path_cfg.DATA_512_AUG_MASK_DIR + "/*.png")

    print("Number of Augmented Images", len(aug_img_paths))
    print("Number of Augmented Masks", len(aug_msk_paths))

    aug_img_paths = aug_img_paths[-100:]
    aug_msk_paths = aug_msk_paths[-100:]
    aug_imgs, aug_msks = read_data(aug_img_paths, aug_msk_paths)
    plot_img_and_mask(aug_imgs, aug_msks, title="",
                      max_rows=10, max_cols=4, figsize=(20, 32))

    for i in range(len(aug_dict)):
        sel_img_paths = [img_path for img_path in aug_img_paths if f'_{i}.png' in img_path]
        sel_msk_paths = [msk_path for msk_path in aug_msk_paths if f'_{i}.png' in msk_path]
        aug_imgs, aug_msks = read_data(sel_img_paths, sel_msk_paths)
        plot_img_and_mask(aug_imgs, aug_msks, title=aug_dict[i],
                          max_rows=2, max_cols=2, figsize=(20, 9))
