import torch
import torch.nn as nn
from d2l import torch as d2l
from torch.nn import functional as F
from torchvision import models
import Data_iter
import Train_Function

batch_size=256

train_iter,test_iter= Data_iter.Get_Data_iter_2(BatchSize=batch_size)

net=models.googlenet(num_classes=2)

#Max=0.981
if __name__ == '__main__':
    Train_Function.train(net,train_iter,test_iter,30,0.001,d2l.try_gpu())























