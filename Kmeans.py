import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
import numpy as np

width = 2000
height = 2000
num_points = 10
min_distance = 100
circle_radius = 400
cluster_distance = 400

points = np.random.uniform((0, 0), (width, height), (num_points, 2))

def bisecting_kmeans(points, cluster_distance):
    if len(points) < 2:
        yield points
        return
    kmeans = KMeans(n_clusters=2, n_init=10).fit(points)
    labels = kmeans.labels_
    clusters = [points[labels == i] for i in range(2)]
    for cluster in clusters:
        distances = squareform(pdist(cluster))
        if (distances > cluster_distance).any():
            yield from bisecting_kmeans(cluster, cluster_distance)
        else:
            yield cluster

clusters = list(bisecting_kmeans(points, cluster_distance))
labels = np.zeros(num_points, dtype=int)
for i, cluster in enumerate(clusters):
    labels[np.isin(points, cluster).all(axis=1)] = i

fig = plt.figure(dpi=200)
ax = fig.add_subplot(111)
ax.scatter(points[:, 0], points[:, 1], c=labels)
for i, (x, y) in enumerate(points):
    circle = plt.Circle((x, y), circle_radius, fill=False, edgecolor='gray', linestyle='--', linewidth=0.8)
    ax.add_artist(circle)
    ax.annotate(str(i), (x, y), fontname='Times New Roman')
for i, cluster in enumerate(clusters):
    center_x, center_y = cluster.mean(axis=0)
    ax.plot(center_x, center_y, marker='+', markersize=10)
    ax.annotate(f'Cluster {i}', (center_x, center_y), fontname='Times New Roman', fontsize=12, fontweight='bold')
ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_aspect('equal')
plt.show()

print('所有点的坐标:')
for i, (x, y) in enumerate(points):
    print(f'点{i}: ({x:.2f}, {y:.2f})')

print('所有中心的坐标:')
for i, cluster in enumerate(clusters):
    center_x, center_y = cluster.mean(axis=0)
    print(f'中心{i}: ({center_x:.2f}, {center_y:.2f})')

print('最终的聚类结果:')
for i in range(len(clusters)):
    print(f'类别{i}: {np.where(labels == i)[0]}')