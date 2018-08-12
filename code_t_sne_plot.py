# coding: utf-8
from sklearn import datasets
from sklearn.manifold import TSNE
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dat10_path = "/Users/liupeng/Documents/PythonProjects/experiments/data/infer/experiment_10_batch_50/code.dat"
dat11_path = "/Users/liupeng/Documents/PythonProjects/experiments/data/infer/experiment_11_batch_50/code.dat"
dat12_path = "/Users/liupeng/Documents/PythonProjects/experiments/data/infer/experiment_12_batch_50/code.dat"

kaiti_code = np.load(dat10_path)
qigong_code = np.load(dat11_path)
songti_code = np.load(dat12_path)

# tsne = TSNE(n_components=3, random_state=0)
#
# kaiti_code_3d = tsne.fit_transform(kaiti_code)
# qigong_code_3d = tsne.fit_transform(qigong_code)
# songti_code_3d = tsne.fit_transform(songti_code)
#
# fig = plt.figure()
# ax = Axes3D(fig)
#
# ax.scatter(kaiti_code_3d[:10, 0], kaiti_code_3d[:10, 1], kaiti_code_3d[:10, 2], c='g')
# ax.scatter(qigong_code_3d[:10, 0], qigong_code_3d[:10, 1], qigong_code_3d[:10, 2], c='r')
# ax.scatter(songti_code_3d[:10, 0], songti_code_3d[:10, 1], songti_code_3d[:10, 2], c='b')

tsne = TSNE(n_components=2, random_state=0)

kaiti_code_2d = tsne.fit_transform(kaiti_code)
qigong_code_2d = tsne.fit_transform(qigong_code)
songti_code_2d = tsne.fit_transform(songti_code)

plt.scatter(songti_code_2d[:, 0], songti_code_2d[:, 1], c="b")
plt.scatter(qigong_code_2d[:, 0], qigong_code_2d[:, 1], c="r")
plt.scatter(kaiti_code_2d[:, 0], kaiti_code_2d[:, 1], c="g")
plt.show()



