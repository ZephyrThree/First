import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import random
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['font.family'] = 'Times New Roman'


def CBBA(robots, tasks, max_iter=10):
    """
    CBBA algorithm for multi-robot task allocation.
    robots: list of robot objects
    tasks: list of task objects
    max_iter: maximum number of iterations
    """
    n_robots = len(robots)
    n_tasks = len(tasks)

    for robot in robots:
        robot.init(n_tasks)

    for i in range(max_iter):
        for robot in robots:
            robot.build_bundle(tasks)
            robot.broadcast()

        for robot in robots:
            robot.update_conflict()

    return robots


class Robot:
    def __init__(self, id):
        self.id = id

    def init(self, n_tasks):
        self.bundle = []
        self.path = [0] * n_tasks
        self.winner = [0] * n_tasks
        self.bids = [0] * n_tasks

    def build_bundle(self, tasks):
        available_tasks = [task for task in tasks if task.id not in self.bundle]
        available_tasks.sort(key=lambda task: -self.calc_bid(task))

        while available_tasks:
            task = available_tasks.pop(0)
            bid = self.calc_bid(task)

            if bid > self.bids[task.id]:
                self.bundle.append(task.id)
                self.path[task.id] = len(self.bundle)
                self.winner[task.id] = self.id
                self.bids[task.id] = bid

    def calc_bid(self, task):
        return 1 / (np.linalg.norm(self.pos - task.pos) + 1e-6)

    def broadcast(self):
        for robot in robots:
            if robot is not self:
                for i in range(len(self.winner)):
                    if self.bids[i] > robot.bids[i]:
                        robot.winner[i] = self.winner[i]
                        robot.bids[i] = self.bids[i]
                        if robot.winner[i] == robot.id:
                            robot.path[i] = 0
                            if i in robot.bundle:
                                robot.bundle.remove(i)

    def update_conflict(self):
        for i in range(len(self.winner)):
            if self.winner[i] != self.id and i in self.bundle:
                self.bundle.remove(i)
                self.path[i] = 0


class Task:
    def __init__(self, id, pos):
        self.id = id
        self.pos = pos

def tsp_ga(points, pop_size=10, n_generations=1000, mutation_rate=0.1):
    def calc_distance(path):
        distance = 0
        for i in range(1, len(path)):
            distance += np.linalg.norm(points[path[i]] - points[path[i-1]])
        return distance

    def crossover(parent1, parent2):
        child = [None] * len(parent1)
        a, b = sorted(random.sample(range(1, len(parent1)), 2))
        child[a:b+1] = parent1[a:b+1]
        for i in range(len(parent2)):
            if parent2[i] not in child:
                for j in range(len(child)):
                    if child[j] is None:
                        child[j] = parent2[i]
                        break
        return child

    def mutate(individual):
        a, b = sorted(random.sample(range(1, len(individual)), 2))
        individual[a], individual[b] = individual[b], individual[a]

    population = [[0] + random.sample(range(1, len(points)), len(points)-1) for _ in range(pop_size)]
    for _ in range(n_generations):
        population.sort(key=lambda x: calc_distance(x))
        new_population = population[:pop_size//2]
        for _ in range(pop_size//2, pop_size):
            parents = random.sample(new_population, 2)
            child = crossover(*parents)
            if random.random() < mutation_rate:
                mutate(child)
            new_population.append(child)
        population = new_population

    population.sort(key=lambda x: calc_distance(x))
    return population[0]

# example
n_robots = 2
n_tasks = 10

robots = [Robot(i) for i in range(n_robots)]
tasks = [Task(i, np.random.rand(3) * np.array([2000, 2000, 1000])) for i in range(n_tasks)]

for robot in robots:
    robot.pos = np.random.rand(3) * np.array([2000, 2000, 1000])

robots = CBBA(robots, tasks)

# plot
fig = plt.figure(dpi=200, figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 0.5])

for i, task in enumerate(tasks):
    ax.scatter(*task.pos, c='black', s=10)
    ax.text(*task.pos, f'T{i}', fontsize=5)

colors = ['r', 'g', 'y', 'c', 'm']
shortest_paths = []
for i, robot in enumerate(robots):
    ax.scatter(*robot.pos, c='blue', s=10)
    ax.text(*robot.pos, f'A{i}', fontsize=5)

    bundle_pos = [tasks[j].pos for j in robot.bundle]
    bundle_pos.insert(0, robot.pos)

    path = tsp_ga(bundle_pos)
    shortest_paths.append([0 if j == 0 else robot.bundle[j - 1] for j in path])

    bundle_pos = [bundle_pos[j] for j in path]

    bundle_pos = np.array(bundle_pos).T
    ax.plot(*bundle_pos, c=colors[i % len(colors)], linewidth=1)

ax.set_xlim(0, 2000)
ax.set_ylim(0, 2000)
ax.set_zlim(0, 1000)

# plt.savefig('figure.png', dpi=200)

plt.show()

# print allocation result
for i, robot in enumerate(robots):
    print(f'Robot {i} is allocated tasks {sorted(robot.bundle)}')

# print shortest path
for i, robot in enumerate(robots):
    print(f'Robot {i} shortest path: {shortest_paths[i]}')

# print shortest path length
for i, robot in enumerate(robots):
    bundle_pos = [tasks[j].pos for j in robot.bundle]
    bundle_pos.insert(0, robot.pos)

    path_length = 0
    for j in range(1, len(shortest_paths[i])):
        task_index_1 = shortest_paths[i][j - 1]
        task_index_2 = shortest_paths[i][j]

        if task_index_1 == 0:
            pos_1 = robot.pos
        else:
            pos_1 = tasks[task_index_1].pos

        if task_index_2 == 0:
            pos_2 = robot.pos
        else:
            pos_2 = tasks[task_index_2].pos

        path_length += np.linalg.norm(pos_2 - pos_1)

    print(f'Robot {i} shortest path length: {path_length}')