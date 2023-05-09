import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

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


# example
n_robots = 5
n_tasks = 20

robots = [Robot(i) for i in range(n_robots)]
tasks = [Task(i, np.random.rand(2) * 2000) for i in range(n_tasks)]

for robot in robots:
    robot.pos = np.random.rand(2) * 2000

robots = CBBA(robots, tasks)

# print allocation result
for i, robot in enumerate(robots):
    print(f'Robot {i} is allocated tasks {sorted(robot.bundle)}')

# plot
fig, ax = plt.subplots(dpi=200)
ax.set_aspect('equal')

for i, task in enumerate(tasks):
    plt.scatter(*task.pos, c='black')
    plt.annotate(f'T{i}', xy=task.pos)

colors = ['r', 'g', 'y', 'c', 'm']
for i, robot in enumerate(robots):
    plt.scatter(*robot.pos, c='blue')
    plt.annotate(f'A{i}', xy=robot.pos)

    bundle_pos = [tasks[j].pos for j in sorted(robot.bundle)]
    bundle_pos.insert(0, robot.pos)

    bundle_pos = np.array(bundle_pos).T
    plt.plot(*bundle_pos, c=colors[i], linewidth=1)

plt.xlim(0, 2000)
plt.ylim(0, 2000)

# plt.savefig('figure.png', dpi=200)

plt.show()