import torch
import os
import cv2
from torch.utils.data import Dataset, DataLoader

class HuBMAPDataset(Dataset):
    def __init__(self,
                 data_path,
                 fnames,
                 preprocess_input=None,
                 transforms=None):
        self.data_path = data_path
        self.fnames = fnames #[:100]
        self.preprocess_input = preprocess_input
        self.transforms = transforms

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, 'train', self.fnames[idx])
        mask_path = os.path.join(self.data_path, 'masks', self.fnames[idx])

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # img = cv2.resize(img, (256, 256))
        # mask = cv2.resize(mask, (256, 256))

        if self.transforms:
            # Applying augmentations if any.
            sample = self.transforms(image=img,
                                     mask=mask)

            img, mask = sample['image'], sample['mask']

        if self.preprocess_input:
            # Normalizing the image with the given mean and
            # std corresponding to each channel.
            img = self.preprocess_input(image=img)['image']

        # PyTorch assumes images in channels-first format.
        # Hence, bringing the channel at the first place.
        img = img.transpose((2, 0, 1))

        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        return img, mask
