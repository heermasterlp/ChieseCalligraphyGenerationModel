# coding: utf-8
from sklearn import datasets
from sklearn.manifold import TSNE
digits = datasets.load_digits()
# Take the first 500 data points: it's hard to see 1500 points
X = digits.data[:500]
y = digits.target[:500]


tsne = TSNE(n_components=3, random_state=0)

X_2d = tsne.fit_transform(X)

target_ids = range(len(digits.target_names))

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, c, label in zip(target_ids, colors, digits.target_names):
    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], X_2d[y == i, 2], c=c, label=label)
plt.legend()
plt.show()