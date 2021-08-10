
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class DriveDataset(Dataset): #build data pipeline
    def __init__(self, images_path, masks_path): #constructor

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR) #Read as RGB image, possibly change to grayscale
        image = image/255.0 ## (512, 512, 3)
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512) #change to channel first approach for torch
        image = image.astype(np.float32)
        image = torch.from_numpy(image) # torch tensor

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask/255.0   ## (512, 512)
        mask = np.expand_dims(mask, axis=0) ## (1, 512, 512) #channel first
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask) #torch tensor

        return image, mask

    def __len__(self):
        return self.n_samples
