from sklearn.datasets import load_iris # 导入鸢尾花数据集
iris = load_iris() # 载入数据集
X = iris.data
y = iris.target

print('iris数据集特征')
print(X[:10])

print('iris数据集标签')
print(y[:10])

from sklearn import decomposition
pca = decomposition.PCA(n_components=3) #加载降维模型PCA，设置降维后的维度为31.2.4 数据处理

import numpy as np
X_t = pca.fit_transform(X) #pca处理原始数据X
y = np.choose(y, [1, 2, 0]).astype(np.float)
print('处理完的数据集变成了三维的特征')
print(X_t[:5])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

fig = plt.figure(1, figsize=(15, 8)) #图标大小设置
plt.clf()
ax = fig.add_subplot(121, projection='3d') #3D图形
plt.cla()
for name, label, color in [('Setosa', 0, 'red'),     #我们将三种类别的数据分别赋予不同的颜色，红色代表类别‘0’对应的Setosa
                           ('Versicolour', 1, 'blue'), #蓝色代表类别‘1’对应的Versicolour
                           ('Virginica', 2, 'yellow')]: #黄色代表类别‘2’对应的Virginica
    ax.scatter(X_t[y == label, 0],
               X_t[y == label, 1],
               X_t[y == label, 2],
    label=name,color=color,edgecolor='k')
plt.legend(prop={'size': 15})
plt.show()

fig = plt.figure(1, figsize=(15, 8))
plt.clf()
ax = fig.add_subplot(122)
for name, label, color in [('Setosa', 0, 'red'), #我们将三种类别的数据分别赋予不同的颜色，红色代表类别‘0’对应的Setosa
                           ('Versicolour', 1, 'blue'), #蓝色代表类别‘1’对应的Versicolour
                           ('Virginica', 2, 'yellow')]: #黄色代表类别‘2’对应的Virginica
    ax.scatter(X_t[y == label, 0],
               X_t[y == label, 1],
               label=name,
               color=color,
               edgecolor='k')
plt.show()

from sklearn.datasets import load_iris # 导入鸢尾花数据集
iris = load_iris() # 载入数据集
X = iris.data
y = iris.target

print('iris数据集特征')
print(X[:10])

print('iris数据集标签')
print(y[:10])

from sklearn import manifold
isomap = manifold.Isomap(n_neighbors=10, n_components=3)

import numpy as np
y = np.choose(y, [1, 2, 0]).astype(np.float)
X_t = isomap.fit_transform(X) ##pca处理原始数据X
print('处理完的数据集变成了三维的特征')
print(X_t[:5])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

fig = plt.figure(1, figsize=(15, 8))
plt.clf()
ax = fig.add_subplot(121, projection='3d') #设置3D
plt.cla()
for name, label, color in [('Setosa', 0, 'red'), #我们将三种类别的数据分别赋予不同的颜色，红色代表类别‘0’对应的Setosa
                           ('Versicolour', 1, 'blue'), #蓝色代表类别‘1’对应的Versicolour
                           ('Virginica', 2, 'yellow')]: #黄色代表类别‘2’对应的Virginica
    ax.scatter(X_t[y == label, 0],
               X_t[y == label, 1],
               X_t[y == label, 2],
    label=name,color=color,edgecolor='k')
plt.legend(prop={'size': 15})
plt.show()

fig = plt.figure(1, figsize=(15, 8))
plt.clf()
ax = fig.add_subplot(122)
for name, label, color in [('Setosa', 0, 'red'), #我们将三种类别的数据分别赋予不同的颜色，红色代表类别‘0’对应的Setosa
                           ('Versicolour', 1, 'blue'), #蓝色代表类别‘1’对应的Versicolour
                           ('Virginica', 2, 'yellow')]: #黄色代表类别‘2’对应的Virginica
    ax.scatter(X_t[y == label, 0],
               X_t[y == label, 1],
               label=name,
               color=color,
               edgecolor='k')
plt.show()



