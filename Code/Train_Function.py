import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from d2l import torch as d2l
from torchsummary import summary
from Data_iter import show_data

# 训练函数
def train(net, train_iter, test_iter, num_epochs, lr, device: torch.device ,show=None):
    if show:
        show_data()
        plt.show()
        exit(0)


    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    cost_time_list=[]
    speed_list=[]

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', torch.cuda.get_device_name(device))
    net.to(device)
    print(device.type)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    timer, num_batches = d2l.Timer(), len(train_iter)

    train_l, train_acc, test_acc, metric = None, None, None, None

    for epoch in range(1,num_epochs+1):
        if 10<epoch<=20:
            lr=lr*0.8
        if epoch>20:
            lr=lr*0.5
        # if epoch>=20:
        #     optimizer = torch.optim.SGD(net.parameters(), lr=lr,momentum=0.9)
        t=time.time()
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(torch.float).to(device), y.to(torch.long).to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            if epoch>5 and l.item()>2:
                print('%%%%%%%%%%%%%%%%%')
                print('参数出现问题！')
                print(l)
                print('%%%%%%%%%%%%%%%%%')
                l/=100
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            print(f'epoch:{epoch},'
                  f'train_loss:{round(train_l, 3)},train_acc:{round(train_acc, 3)}')
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        print(f'loss {train_l:.6f}, train acc {train_acc:.6f}, '
              f'test acc {test_acc:.6f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
              f'on {str(device)}')
        train_loss_list.append(train_l)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        cost_time_list.append(round(time.time()-t,2))
        speed_list.append(round(metric[2] * num_epochs / timer.sum(),2))
        print(f'Cost Time ={round(time.time()-t,3)}s')
        print('-----------------------------------------------------------------------')

    # print('-------------------------------Net Summary--------------------------------------------------')
    #print(summary(net, (3,64,64), device=d2l.try_gpu()))
    print('--------------------------------Cost Time and Speed-----------------------------------------')
    print(f'Mean Cost Time per Epoch is {sum(cost_time_list)/num_epochs:.2f}')
    print(f'Mean:{sum(speed_list)/num_epochs:.2f}examples/sec  ')
    print('----------------------------------------------------------------------------------')
    print(f'Min Loss:{min(train_loss_list)}')
    print(f'Max ACC:{max(test_acc_list)}')
    #绘图
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20,
             }

    epoch=list(range(1,num_epochs+1))
    plt.plot(epoch, train_loss_list,"o-", label='train_loss')
    plt.xlabel('epoch',fontdict=font1)
    plt.ylabel('val',fontdict=font1)
    plt.grid(b=True,
             color='black',
             linestyle='--',
             linewidth=0.2,
             alpha=0.4,
             axis='y',
             which="major")
    plt.legend(prop=font1)
    plt.show()

    plt.plot(epoch, train_acc_list, "o-",label='train_acc')
    plt.plot(epoch, test_acc_list, "o-",label='test_acc')
    plt.xlabel('epoch',fontdict=font1)
    plt.ylabel('val',fontdict=font1)
    plt.grid(b=True,
             color='black',
             linestyle='--',
             linewidth=0.2,
             alpha=0.4,
             axis='y',
             which="major")
    plt.legend(prop=font1)
    plt.show()










