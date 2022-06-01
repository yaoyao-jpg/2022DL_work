import torch
from ptflops import get_model_complexity_info
from torchvision import models

net=models.convnext_base(pretrained=False,num_classes=2)


with torch.cuda.device(0):
  flops, params = get_model_complexity_info(net, (3,64,64), print_per_layer_stat=False)
  print('Flops:  ' + flops)
  print('Params: ' + params)




























