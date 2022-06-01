from d2l.torch import d2l
import torch.nn as nn
import Data_iter
import Train_Function
from torchvision import models

batch_size=256

train_iter,test_iter= Data_iter.Get_Data_iter_2(BatchSize=batch_size)

net=models.vgg11_bn(num_classes=2)

models.inception_v3()
#Max=0.981
if __name__ == '__main__':
    Train_Function.train(net,train_iter,test_iter,30,0.001,d2l.try_gpu())



























