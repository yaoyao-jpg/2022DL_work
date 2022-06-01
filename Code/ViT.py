import torch.nn as nn
from d2l import torch as d2l
from torchvision import models

import Data_iter
import Train_Function

batch_size = 128

train_iter, test_iter = Data_iter.Get_Data_iter_2(BatchSize=batch_size)

net=models.vit_l_32(pretrained=False,image_size=64,num_classes=2)

#num_classes=2
# Max=0.981
if __name__ == '__main__':
    Train_Function.train(net, train_iter, test_iter, 50, 0.0001, d2l.try_gpu())












