# import matplotlib  # 注意这个也要import一次
# import matplotlib.pyplot as plt
# import autogluon.core as ag
# import pandas
# from autogluon.vision import ImageDataset, ImagePredictor
# import pandas as pd
# import os


# df = ImageDataset.from_csv('objects.csv')
# df['image'] = df['image'].apply(lambda x: os.path.join('..\\data', str(x)) + '.jpg')
# df['label'] = pd.factorize(df.label)[0]
# #展示数据
# # print(df.head(n=15))
# # df.show_images(fontsize=10,resize=224)
# # plt.show()
# if __name__ == '__main__':
#     train, val, test = df.random_split()
#     predictor = ImagePredictor()
#     # since the original dataset does not provide validation split, the `fit` function splits
#     # it randomly with 90/10 ratio
#     predictor.fit(train, hyperparameters={'epochs': 2})
#     # you can trust the default config, we reduce the epoch to save some build time










