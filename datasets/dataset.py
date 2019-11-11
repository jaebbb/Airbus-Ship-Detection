from data import *
import torch
from torch.utils.data import Dataset
import torchvision
import pandas as pd
import os
import numpy as np
from utils.read_segmentation_csv import get_segmentation
from configs import config
import cv2

class ShipDataset(Dataset):
    def __init__(self, image_transform = None, label_transform = None):
        """
        Args:
            image_transform
            label_transform
        __getitem__:
            Args:
                index (int)
            Returns:
                sample (Dict): Python dictionary contains images, labels
                    Image (H, W, C)
                    Label (H, W)
        """
        self.image_segmentation_dict = get_segmentation(config.train_ship_segmentations, config.image_size)
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.image_segmentation_dict)


    def __getitem__(self,index):
        
        image_list = os.listdir(config.image_folder)
        img = cv2.imread(os.path.join(config.image_folder,image_list[index]))[:,:,::-1]
        label = self.image_segmentation_dict[image_list[index]]
        if self.image_transform is not None:
            img = self.image_transform(face_vframes)

        if self.label_transform is not None:
            label = self.label_transform(aframes)

        sample = {'Image' : img,'Label': label}

        return sample