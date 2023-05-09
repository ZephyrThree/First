import matplotlib.pyplot as plt
import random

width = 2000
height = 2000
num_points = 10
min_distance = 100
circle_radius = 400

points = []
for i in range(num_points):
    while True:
        x = random.uniform(0, width)
        y = random.uniform(0, height)
        if all(((x - px)**2 + (y - py)**2)**0.5 > min_distance for px, py in points):
            points.append((x, y))
            break

fig, ax = plt.subplots(dpi=200)
ax.scatter(*zip(*points), s=15)
for x, y in points:
    circle = plt.Circle((x, y), circle_radius, fill=False, edgecolor='gray',
                        linestyle='--', linewidth=0.8)
    ax.add_artist(circle)
ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_aspect('equal')
plt.show()