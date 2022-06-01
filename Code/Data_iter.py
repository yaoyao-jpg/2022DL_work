import time
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from matplotlib import pyplot as plt
from skimage import io
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from torchvision import transforms

warnings.filterwarnings("ignore")


class SDSSDataSet(Dataset):
    def __init__(self, csv_file, root, transform=None):
        """
        csv_file（string）：带注释的csv文件的路径。
        root_dir（string）：包含所有图像的目录。
        transform（callable， optional）：一个样本上的可用的可选变换
        """
        self.data_csv = pd.read_csv(csv_file)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        path = self.root + str(self.data_csv.iloc[idx, 0]) + ".jpg"
        img = io.imread(path)
        label = int(self.data_csv.iloc[idx, 1] == "STAR")
        if self.transform:
            img = self.transform(img)

        return np.swapaxes(img,0,2) / 255, label


# def Get_Data_iter(BatchSize=256):
#     transform = torchvision.transforms.Compose([
#         torchvision.transforms.RandomHorizontalFlip(),
#         torchvision.transforms.ToTensor()])
#     data_set = SDSSDataSet(csv_file="../objects.csv",
#                            root='../data/', transform=None)
#     full_size = len(data_set)
#     train_size = int(full_size * 0.85)
#     valid_size = int(full_size - train_size)
#     train_set, valid_set = torch.utils.data.random_split(data_set, [train_size, valid_size])
#
#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=BatchSize, shuffle=True, num_workers=0)
#     valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=BatchSize, shuffle=False, num_workers=0)
#     return train_loader, valid_loader


def show_data(height=2, width=4, transform=None):
    data_iter, _ = Get_Data_iter_2()
    idx = 1
    X, y = None, None
    for _, __ in data_iter:
        X, y = _, __
        break
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            plt.subplot(height, width, idx)
            plt.axis('off')
            idx += 1
            if transform is not None:
                plt.imshow(transform(X[idx]))
            else:
                plt.imshow(X[idx])
            plt.title('STAR' if y[idx] == 1 else 'GALAXY')
    plt.tight_layout()
    plt.show()


def Get_Data_iter_2(BatchSize=256):
    print('Starting to get data...')
    a=time.time()

    data = pd.read_csv('../objects.csv')

    def path2pic(path)->np.ndarray:
        img = cv2.imread(f'../data/{path}.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img=np.swapaxes(img,0,2)
        if np.shape(img)!=(3,64,64):
            print(type(img))
            print(np.shape(img))
            print(img)
            print('Error!')
            exit(0)
        return img/256


    data['Class'] = pd.factorize(data.Class)[0]

    X = data['objID'].apply(path2pic).values
    Y = data.Class.values

    X_=np.zeros((36888,3,64,64))


    for index,i in enumerate(X):
        X_[index]=i

    X=X_

    train_x, test_x, train_y, test_y = train_test_split(X, Y,train_size=0.85)

    train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)
    test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
    test_y = torch.from_numpy(test_y).type(torch.LongTensor)

    batch=BatchSize

    train_iter = TensorDataset(train_x, train_y)
    train_iter = DataLoader(train_iter, batch_size=batch, shuffle=True)
    test_iter = TensorDataset(test_x, test_y)
    test_iter = DataLoader(test_iter, batch_size=batch)


    print('------------------------------------------')
    print('所用时间：'+str(time.time()-a)+'s')
    print('------------------------------------------')

    return train_iter,test_iter
