#-*-coding:utf-8-*-
#@Author:Trina
import vcf
import seaborn as sns
from subprocess import check_output
from time import time
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve, plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.inspection import permutation_importance
# from sklearn.datasets import load_boston
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict

# from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
import os
from matplotlib import pyplot as plt
# from tsne import bh_sne
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets


x_compare = pd.read_table("C:/Users/Administrator/PycharmProjects/pythonProject2/detext/data/processed/dateset_1x.sam",usecols=[0])
y_compare = pd.read_table("C:/Users/Administrator/PycharmProjects/pythonProject2/detext/data/processed/dateset_1x.sam",usecols=[2])

# x_train,x_test,y_train,y_test = train_test_split(x_compare,y_compare,test_size = 0.2,random_state = 5)

tsne = TSNE(n_components=2)
temp = []
x_compare = np.array(x_compare).tolist()
for i in x_compare:
    i = i[0]
    temp.append(eval(i))
Y = tsne.fit_transform(temp)
# plt.figure(figsize=(8,8))
# for i in range(x_tsne.shape[0]):
#     plt.text(x_tsne[i,0],x_tsne[i,1],str(y_compare[i]),color = plt.cm.Set1(y_compare[i]), fontdict ={'weight':'bold','size':9})
# plt.xticks([])
# plt.yticks([])
# plt.show()
# y_data = np.where(y_compare == 'F1')
# print(Y)
# red = y_compare == '0'
# green = y_compare == '1'
plt.scatter(Y[0],Y[1])

# plt.scatter(Y[red,0],Y[red,1],c ="gold", cmap = plt.cm.Spectral)
# plt.scatter(,c ="mediumturquoise", cmap = plt.cm.Spectral)
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# # plt.axis("tight")
plt.show()