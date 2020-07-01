# パッケージのimport
import random
import math
import time
import pandas as pd
import numpy as np
import os
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import os.path as osp
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import cv2
from glob import glob


#訓練画像とテスト画像のpathの準備
path_train='./mri_segmentation/train'
path_test='./mri_segmentation/val'

train_path_images = os.path.abspath(path_train+'/image/')
train_path_masks = os.path.abspath(path_train+'/label/')
val_path_images = os.path.abspath(path_test+'/image/')
val_path_masks = os.path.abspath(path_test+'/label/')

train_path_images_list = glob(os.path.join(train_path_images, '*.png'))
train_path_masks_list = glob(os.path.join(train_path_masks, '*.png'))
val_path_images_list = glob(os.path.join(val_path_images, '*.png'))
val_path_masks_list = glob(os.path.join(val_path_masks, '*.png'))
print(len(val_path_images_list))
train_images_path = []
train_masks_path = []
val_images_path = []
val_masks_path = []

for path in sorted(train_path_images_list):
    train_images_path.append(path)
for path in sorted(train_path_masks_list):
    train_masks_path.append(path)
for path in sorted(val_path_images_list):
    val_images_path.append(path)
for path in sorted(val_path_masks_list):
    val_masks_path.append(path)


from torchvision.transforms import Compose, Lambda, ToTensor

def input_transform():
    return Compose([
        ToTensor(),
    ])


def label_transform():
    return Compose([
        ToTensor(),
        Lambda(lambda x: x.mul_(255.)),
    ])


from make_dataset import Make_Dataset
train_dataset = Make_Dataset(train_images_path, train_masks_path, input_transform = input_transform(), label_transform = label_transform())
val_dataset = Make_Dataset(val_images_path, val_masks_path, input_transform = input_transform(), label_transform = label_transform())

batch_size = 16
print(len(val_masks_path))
train_dataloader = data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

from model_ import Unet
net = Unet(in_ch=1, f_out=32, n_classes = 1)

if torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'

print(device)
net.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

logs_train=[]
def train(epoch):
    epoch_loss = 0.
    epoch_iou = 0.
    net.train()
    for batch in train_dataloader:
        img, mask = batch[0].to(device), batch[1].to(device)
  
        optimizer.zero_grad()
        output = net(img)
        loss = criterion(output, mask)
        epoch_loss += loss.item() * len(img)
        loss.backward()
        optimizer.step()

        output = torch.sigmoid(output.detach())
        output, mask = output.cpu().numpy(), mask.cpu().numpy()
        output = np.round(output)
        output, mask = output[:, 0, ...], mask[:, 0, ...]
        inter = (output * mask).sum(axis=(1,2))
        union = (output + mask - output * mask).sum(axis=(1,2))
        epoch_iou += (inter / union).sum()

    if epoch % 10 == 0:
        print("### Epoch {}: Avg. Loss: {:.6f}: Avg. mIOU: {:.6f}".format(
            epoch, epoch_loss / len(train_dataset), epoch_iou / len(train_dataset)))
        log_epoch_train = {'epoch': epoch, 'train_loss': epoch_loss / len(train_dataset), 'train_iou': epoch_iou / len(train_dataset)}
        logs_train.append(log_epoch_train)
        df = pd.DataFrame(logs_train)
        df.to_csv('./mri_segmentation/log_train.csv')
        #ログを保存

logs_val=[]
def validation():
    avg_loss = 0.
    avg_iou = 0.
    net.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            img, mask = batch[0].to(device), batch[1].to(device)

            prediction = net(img)
            loss = criterion(prediction, mask)
            avg_loss += loss * len(img)
            avg_loss.cpu()

            output = torch.sigmoid(prediction)
            output, mask = output.cpu().numpy(), mask.cpu().numpy()
            output = np.round(output)
            output, mask = output[:, 0, ...], mask[:, 0, ...]
            inter = (output * mask).sum(axis=(1,2))
            union = (output + mask - output * mask).sum(axis=(1,2))
            avg_iou += (inter / union).sum()
            
    print('# Avg. Loss: {:.6f} : Avg. mIOU: {:.6f}'.format(
        avg_loss / len(val_dataset), avg_iou / len(val_dataset)))
    log_epoch_val = {'epoch': epoch, 'val_loss': avg_loss / len(val_dataset), 'val_iou': avg_iou / len(val_dataset)}
    logs_val.append(log_epoch_val)
    df = pd.DataFrame(logs_val)
    df.to_csv('./mri_segmentation/log_val.csv')


def checkpoint(epoch):
    net_out_path = "./mri_segmentation/res_net_epoch_{}.pth".format(epoch)
    torch.save(net.state_dict(), net_out_path)
    print("# model saved to {}".format(net_out_path))


num_epoch = 1000

for epoch in range(1, num_epoch+1):
    train(epoch)
    if epoch % 10 == 0:
        validation()
    if epoch % 50 == 0:
        checkpoint(epoch)