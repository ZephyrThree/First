import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
import numpy as np
import seaborn as sns

width = 2000
height = 2000
depth = 1000
num_points = 5
min_distance = 100
circle_radius = 400
cluster_distance = 400

points = np.random.uniform((0, 0, 0), (width, height, depth), (num_points, 3))

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

sns.set_style('darkgrid')
fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels)
for i, (x, y, z) in enumerate(points):
    ax.text(x, y, z, str(i), fontname='Times New Roman')
for i, cluster in enumerate(clusters):
    center_x, center_y, center_z = cluster.mean(axis=0)
    ax.text(center_x+120, center_y-120, center_z, f'C{i}', fontname='Times New Roman', fontsize=10, fontweight='bold')

    # u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    # x = center_x + 200 * np.cos(u) * np.sin(v)
    # y = center_y + 200 * np.sin(u) * np.sin(v)
    # z = center_z + 200 * np.cos(v)
    # ax.plot_wireframe(x, y, z, color="r", linewidth=0.5)
ax.set_xlim(0,width)
ax.set_ylim(0,height)
ax.set_zlim(0,depth)
ax.set_aspect('equal')

x_ticks = np.arange(0,width+1,width/4)
y_ticks = np.arange(0,height+1,height/4)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)

plt.show()

print("所有点的坐标")
for i,(x,y,z) in enumerate(points):
    print(f"点 {i}: ({x:.2f}, {y:.2f}, {z:.2f})")

print("\n所有聚类中心的坐标:")
for i,cluster in enumerate(clusters):
    center_x, center_y, center_z = cluster.mean(axis=0)
    print(f"中心 {i}: ({center_x:.2f}, {center_y:.2f}, {center_z:.2f})")

print('最终的聚类结果:')
for i in range(len(clusters)):
    print(f'类别{i}: {np.where(labels == i)[0]}')