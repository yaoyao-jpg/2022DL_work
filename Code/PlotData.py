import time

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torchvision import transforms

print('Starting to load...')

data = pd.read_csv('../objects.csv')


def path2pic(path):
    img = cv2.imread(f'../data/{path}.jpg')
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
data['light_block'] = data['img'].apply(count_un_black)
data['bri'] = data['img'].apply(count_brightness)
print(time.time() - a)

# y= data['class']
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y)

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }



sns.set_style('darkgrid')
plt.figure(figsize=(15,8))
sns.countplot(x = data['Class'], palette = 'ch:s=.25,rot=-.25')
plt.legend(prop=font1)
plt.xlabel('Class', fontdict=font1)
plt.ylabel('Num', fontdict=font1)
plt.show()

sns.set_style('darkgrid')
plt.figure(figsize=(15,8))
plt.title('光强-激活点散点图')
sns.scatterplot(x= 'bri', y = 'light_block', hue = 'Class',
                data = data, palette="flare")
plt.legend(prop=font1)
plt.xlabel('Brightness', fontdict=font1)
plt.ylabel('Active-Point', fontdict=font1)
plt.show()

sns.set_style('darkgrid')
plt.figure(figsize=(15,8))
plt.title('激活点-光强散点图')
sns.scatterplot(x= 'light_block', y = 'bri', hue = 'Class',
                data = data, palette="flare")
plt.ylabel('Active-Point', fontdict=font1)
plt.xlabel('Brightness', fontdict=font1)
plt.legend(prop=font1)
plt.show()









