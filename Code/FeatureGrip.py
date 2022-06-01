import time

import d2l.torch
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchsummary import summary
import torchvision.transforms as transforms
import cv2

data = pd.read_csv('../objects.csv')


def path2pic(path):
    img = cv2.imread(f'data/{path}.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.ToTensor()(img)  # tensor数据格式是torch(C,H,W)
    return img_tensor


x = 0.015


def count_un_black(img: torch.Tensor) -> float:
    return ((img >= x).float().sum() / (3 * 64 * 64)).item()


def count_brightness(img: torch.Tensor) -> float:
    return torch.sum(img).item() / 3


a = time.time()

data['img'] = data['objID'].apply(path2pic)
data['light'] = data['img'].apply(count_un_black)

print(time.time() - a)

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 26,
         }

plt.figure(figsize=(10, 5))
A= data.light[data['Class'] == 'STAR'].plot(kind='kde', label='STAR', fontsize=20)
B= data.light[data['Class'] == 'GALAXY'].plot(kind='kde', label='GALAXY', fontsize=20)
plt.xlabel('Active-Point Ratio', fontdict=font1)
plt.ylabel('Density', fontdict=font1)
plt.legend(prop=font1)
plt.show()
