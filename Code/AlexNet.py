import d2l.torch
from torchvision import models

import Data_iter
import Train_Function

batch_size = 256

train_iter, test_iter = Data_iter.Get_Data_iter_2(BatchSize=batch_size)

net=models.alexnet(num_classes=2)

#Max=0.979
if __name__ == '__main__':
    Train_Function.train(net, train_iter, test_iter, 20, 0.001, d2l.torch.try_gpu())










