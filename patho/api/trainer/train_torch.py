# https://www.kaggle.com/joshi98kishan/foldtraining-pytorch-tpu-8-cores

import os
import gc
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

from albumentations import *
from segmentation_models_pytorch.unet import Unet
from segmentation_models_pytorch.encoders import get_preprocessing_fn

import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader

from patho.utils.utils import set_all_seeds, check_lr
from patho.cfg.config import path_cfg,img_proc_cfg
from patho.model.model_torch import DiceLoss,HuBMAPModel, get_dice_coeff
from patho.datagen.datagen import HuBMAPDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER_NAME = 'se_resnext50_32x4d'

preprocessing_fn = Lambda(image = get_preprocessing_fn(encoder_name = ENCODER_NAME,
                                                       pretrained = 'imagenet'))

# https://www.kaggle.com/iafoss/hubmap-pytorch-fast-ai-starter
transforms = Compose([
                HorizontalFlip(),
                VerticalFlip(),
                RandomRotate90(),
                ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
                                 border_mode=cv2.BORDER_REFLECT),
                OneOf([
                    OpticalDistortion(p=0.3),
                    GridDistortion(p=.1),
                    IAAPiecewiseAffine(p=0.3),
                ], p=0.3),
                OneOf([
                    HueSaturationValue(10,15,10),
                    CLAHE(clip_limit=2),
                    RandomBrightnessContrast(),
                ], p=0.3),
            ], p=1.0)

def demo_lr():
    demo_model = HuBMAPModel()

    # LR which we set here is the highest LR value.
    demo_optim = optim.SGD(demo_model.parameters(),
                           lr=0.3,
                           momentum=0.9)
    DEMO_EPOCHS = 60

    # Change the T_0 argument to get more number of cycles.
    demo_sched = CosineAnnealingWarmRestarts(demo_optim,
                                             T_0=DEMO_EPOCHS // 3,
                                             T_mult=1,
                                             eta_min=0,
                                             last_epoch=-1)

    check_lr(DEMO_EPOCHS, demo_optim, demo_sched)

def trainer(DATA_PATH, epoch=30, output_dir=''):
    loss_fn = DiceLoss()
    for fold, (t_idx, v_idx) in enumerate(group_kfold.split(fnames,
                                                            groups=groups)):
        print(f'Fold: {fold + 1}')
        print('-' * 40)

        t_fnames = fnames[t_idx]
        v_fnames = fnames[v_idx]

        train_ds = HuBMAPDataset(data_path=DATA_PATH,
                                 fnames=t_fnames,
                                 preprocess_input=preprocessing_fn,
                                 transforms=transforms)

        val_ds = HuBMAPDataset(data_path=DATA_PATH,
                               fnames=v_fnames,
                               preprocess_input=preprocessing_fn,
                               transforms=None)

        print(f'Length of training set: {len(train_ds)}')
        print(f'Length of validation set: {len(val_ds)}')

        model = HuBMAPModel()
        model.float()

        lr = 0.3

        print('\n')

        train_dl = DataLoader(dataset=train_ds,
                              batch_size=4,
                              num_workers=8)

        val_dl = DataLoader(dataset=val_ds,
                            batch_size=4,
                            num_workers=8)

        fold_model = model
        fold_model.to(device)

        optimizer = optim.SGD(model.parameters(),
                              lr=lr,
                              momentum=0.9)

        # scheduler = CosineAnnealingWarmRestarts(optimizer,
        #                                         T_0=epoch// 3,
        #                                         T_mult=1,
        #                                         eta_min=0,
        #                                         last_epoch=-1)

        for e_no, epoch in enumerate(range(epoch)):
            # Here comes our data loader for 8 cores.
            # It takes famous 'DataLoader()' object and list of
            # devices where data has to be sent.
            # Calling 'per_device_loader()' on it will
            # return the data loader for the particular device.
            train_one_epoch(e_no,
                            train_dl,
                            fold_model,
                            loss_fn,
                            optimizer,
                            device)

            # del train_dl
            gc.collect()

        print('\nValidating now...')

        loss, dice_coeff = eval_fn(val_dl,
                                   fold_model,
                                   loss_fn,
                                   device)
        # del val_dl
        gc.collect()

        # Saving the model, so that we can import it in the inference kernel.
        torch.save(fold_model.state_dict(), output_dir + "/8core_fold_model_{fold}.pth")


        del train_ds, val_ds, model



# Updated
def train_one_epoch(epoch_no, data_loader, model, loss_fn, optimizer, device, scheduler=None):
    '''
    Run one epoch on the 'model'.
    Args:
        epoch_no: Serial number of the given epoch
        data_loader: Data iterator like DataLoader
        model: Model which needs to be train for one epoch
        optmizer : Pytorch's optimizer
        device: Device(A particular core) on which to run this epoch
        Scheduler : Pytorch's lr scheduler


    Returns: Nothing
    '''
    model.train()
    losses = []
    dice_coeffs = []

    for i, batch in tqdm(enumerate(data_loader)):
        img_btch, mask_btch = batch

        # *** Putting the images and masks to the TPU device. ***
        img_btch = img_btch.to(device)
        mask_btch = mask_btch.to(device)

        optimizer.zero_grad()

        pred_mask_btch = model(img_btch.float())

        loss = loss_fn(pred_mask_btch, mask_btch.float())

        loss.backward()

        '''
        xm.optimizer_step():
        Consolidates the gradients between cores and issues the XLA device step computation.
        The `step()` function now not only propagates gradients, but uses the TPU context 
        to synchronize gradient updates across each processes' copy of the network. 
        This ensures that each processes' network copy stays "in sync" (they are all identical).
        This means that each process's network has the same weights after this is called.
        [Source:PyTorch XLA doc]
        '''
        # Note: barrier=True not needed when using ParallelLoader
        # xm.optimizer_step(optimizer)
        if scheduler is not None:
            scheduler.step()

        # 'mesh_reduce()' reduce the loss calculated on 8 cores.
        # The way it needs to be reduced is defined in 'reduce()' function
        # loss_reduced = xm.mesh_reduce('train_loss_reduce',
        #                               loss,
        #                               reduce)
        losses.append(loss.item())

        dice_coeff = get_dice_coeff(torch.squeeze(pred_mask_btch),
                                    mask_btch.float())
        dice_coeffs.append(dice_coeff.cpu().detach().numpy())

        del img_btch, pred_mask_btch, mask_btch
        gc.collect()

    print(f'{epoch_no + 1} - Loss : {np.mean(losses): .4f}, Dice Coeff : {np.mean(dice_coeffs): .4f}')


# New stuff
def eval_fn(data_loader, model, loss_fn, device):
    '''
    Calculates metrics on the validation data.

    Returns: returns calculated metrics
    '''
    model.eval()

    dice_coeffs = []
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            img_btch, mask_btch = batch

            img_btch = img_btch.to(device)
            mask_btch = mask_btch.to(device)
            pred_mask_btch = model(img_btch.float())

            loss = loss_fn(pred_mask_btch,
                           mask_btch.float())
            losses.append(loss.item())

            dice_coeff = get_dice_coeff(torch.squeeze(pred_mask_btch),
                                        mask_btch.float())
            dice_coeffs.append(dice_coeff.cpu().detach().numpy())

    total_dice_coeff = (dice_coeffs)
    total_loss = (losses)

    print(f'Val Loss : {np.mean(losses): .4f}, Val Dice : {np.mean(dice_coeffs): .4f}')

    return total_loss, total_dice_coeff


if __name__ == '__main__':
    set_all_seeds(21)
    tileProcCfg = img_proc_cfg()
    path_cfg    = path_cfg()

    print(f'No. of training images: {len(os.listdir(path_cfg.DATA_512_IMG_DIR))}')
    print(f'No. of masks: {len(os.listdir(path_cfg.DATA_512_MASK_DIR))}')

    # Images and its corresponding masks are saved with the same filename.
    fnames = np.array(os.listdir(path_cfg.DATA_512_IMG_DIR))

    groups = [fname[:9] for fname in fnames]

    group_kfold = GroupKFold(n_splits=4)

    # demo_lr(
    trainer(path_cfg.DATA_512_DIR, output_dir=path_cfg.RESULTS_DIR)