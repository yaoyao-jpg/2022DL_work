import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('../data/res.csv')
print(data.head())
data.sort_values(by='Train_speed(s/epoch)',inplace=True,ascending=True)
plt.bar(data['Class'],data['Train_speed(s/epoch)'],0.8,
        color=['lime','cyan','pink','lime','cyan','pink','lime',
               'cyan','pink','lime','cyan','pink','lime'])
plt.xticks(data['Class'],rotation='45')

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 10,
         }

x=0
for i in data['Train_speed(s/epoch)']:
    plt.text(x,i+0.5,str(i),ha='center',fontdict=font1)
    x+=1
plt.tight_layout()
plt.show()













