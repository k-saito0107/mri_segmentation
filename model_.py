# パッケージのimport
import numpy as np
import os
import torch
import torch.utils.data as data
import torch.nn as nn



class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(double_conv, self).__init__()
        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.conv2=nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x



class Unet(nn.Module):
    def __init__(self, in_ch, f_out, n_classes):
        super(Unet,self).__init__()
        #エンコーダー
        self.double_conv1 = double_conv(in_ch, f_out, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv2 = double_conv(f_out, f_out*2, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv3 = double_conv(f_out*2, f_out*4, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv4 = double_conv(f_out*4, f_out*8, kernel_size=3, stride=1, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv5 = double_conv(f_out*8, f_out*16, kernel_size=3, stride=1, padding=1)
        
        #デコーダー
        self.t_conv4 = nn.ConvTranspose2d(f_out*16, f_out*8, kernel_size=2, stride=2)
        self.ex_double_conv4 = double_conv(f_out*16, f_out*8, 3, 1, 1)
        self.t_conv3 = nn.ConvTranspose2d(f_out*8, f_out*4, kernel_size=2, stride=2)
        self.ex_double_conv3 = double_conv(f_out*8, f_out*4, 3, 1, 1)
        self.t_conv2 = nn.ConvTranspose2d(f_out*4, f_out*2, kernel_size=2, stride=2)
        self.ex_double_conv2 = double_conv(f_out*4, f_out*2, 3, 1, 1)
        self.t_conv1 = nn.ConvTranspose2d(f_out*2, f_out, kernel_size=2, stride=2)
        self.ex_double_conv1 = double_conv(f_out*2, f_out, 3, 1, 1)
        
        self.out_conv=nn.Conv2d(f_out, n_classes, 1, 1)
        
    def forward(self, x):
        d_conv1=self.double_conv1(x)
        d_conv2=self.double_conv2(self.maxpool1(d_conv1))
        d_conv3=self.double_conv3(self.maxpool2(d_conv2))
        d_conv4=self.double_conv4(self.maxpool3(d_conv3))
        d_conv5=self.double_conv5(self.maxpool4(d_conv4))
        
        t_conv4 = self.t_conv4(d_conv5)
        cat4 = torch.cat([d_conv4 ,t_conv4], 1)
        ex_conv4 = self.ex_double_conv4(cat4)
        
        t_conv3 = self.t_conv3(ex_conv4)
        cat3 = torch.cat([d_conv3 ,t_conv3], 1)
        ex_conv3 = self.ex_double_conv3(cat3)
        
        t_conv2 = self.t_conv2(ex_conv3)
        cat2 = torch.cat([d_conv2 ,t_conv2], 1)
        ex_conv2 = self.ex_double_conv2(cat2)
        
        t_conv1 = self.t_conv1(ex_conv2)
        cat1 = torch.cat([d_conv1 ,t_conv1], 1)
        ex_conv1 = self.ex_double_conv1(cat1)
        output=self.out_conv(ex_conv1)
        
        return output.reshape(x.shape[0], 1, output.shape[2], output.shape[3])
