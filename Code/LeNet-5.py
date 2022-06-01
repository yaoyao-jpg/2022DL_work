import d2l.torch
import torch.nn as nn
import Data_iter
import Train_Function
from torchvision import models
import torch
batch_size=256

print('Start!')
train_iter,test_iter= Data_iter.Get_Data_iter_2(BatchSize=batch_size)


class LeNet5(nn.Module):
    def __init__(self, num_classes=2):
        super(LeNet5, self).__init__()
        '''第一层卷积，卷积核大小为5*5，步距为1，输入通道为3，输出通道为6'''
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1)

        '''第一层池化层，卷积核为2*2，步距为2，相当于特征图缩小了一般'''
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        '''第二层卷积，卷积核大小为5*5，步距为1，输入通道为6，输出通道为16'''
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)

        '''第二层池化层，卷积核为2*2，步距为2，相当于特征图缩小了一般'''
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        '''第一层全连接层，维度由16*13*13=>256'''
        self.linear1 = nn.Linear(16 * 13 * 13, 256)

        '''第二层全连接层，维度由120=>84'''
        self.linear2 = nn.Linear(256, 64)

        '''第三层全连接层，维度由64=>2'''
        self.linear3 = nn.Linear(64, num_classes)

    def forward(self, x):
        """将数据送入第一个卷积层"""
        out = torch.sigmoid(self.conv1(x))

        '''将数据送入第一个池化层'''
        out = self.pool1(out)

        '''将数据送入第二个卷积层'''
        out = torch.sigmoid(self.conv2(out))

        '''将数据送入第二个池化层'''
        out = self.pool2(out)

        '''将池化层后的数据进行Flatten，使数据变成能够被FC层接受的Vector'''
        out = out.reshape(-1, 16 * 13 * 13)

        '''将数据送入第一个全连接层'''
        out = torch.sigmoid(self.linear1(out))

        '''将数据送入第二个全连接层'''
        out = torch.sigmoid(self.linear2(out))

        '''将数据送入第三个全连接层得到输出'''
        out = self.linear3(out)

        return out

net=LeNet5()

#Max=0.974

if __name__ == '__main__':
    print('Start!')
    Train_Function.train(net,train_iter,test_iter,20,0.001,d2l.torch.try_gpu())



























