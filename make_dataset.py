# パッケージのimport
import pandas as pd
import numpy as np
import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms

class Make_Dataset(data.Dataset):
    def __init__(self, img_list, masks_list, input_transform, label_transform):
        self.img_list=img_list
        self.masks_list = masks_list
        self.input_transform = input_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img_file_path = self.img_list[index]
        img = Image.open(img_file_path)
        img = img.convert('L')

        mask_file_path = self.masks_list[index]
        mask = Image.open(mask_file_path)
        mask = mask.convert('L')

        img = self.input_transform(img)
        mask = self.label_transform(mask)

        return img , mask