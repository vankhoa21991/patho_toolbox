import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.unet import Unet

class HuBMAPModel(nn.Module):
    def __init__(self, ENCODER_NAME='se_resnext50_32x4d'):
        super(HuBMAPModel, self).__init__()
        self.model = Unet(encoder_name=ENCODER_NAME,
                          encoder_weights='imagenet',
                          classes=1,
                          activation=None)

    def forward(self, images):
        img_masks = self.model(images)
        return img_masks


# https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


def get_dice_coeff(pred, targs):
    '''
    Calculates the dice coeff of a single or batch of predicted mask and true masks.

    Args:
        pred : Batch of Predicted masks (b, w, h) or single predicted mask (w, h)
        targs : Batch of true masks (b, w, h) or single true mask (w, h)

    Returns: Dice coeff over a batch or over a single pair.
    '''

    pred = (pred > 0).float()
    return 2.0 * (pred * targs).sum() / ((pred + targs).sum() + 1.0)


def reduce(values):
    '''
    Returns the average of the values.
    Args:
        values : list of any value which is calulated on each core
    '''
    return sum(values) / len(values)