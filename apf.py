import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'


def CBBA(robots, tasks, obstacles, max_iter=10):
    """
    CBBA algorithm for multi-robot task allocation with obstacle avoidance.
    robots: list of robot objects
    tasks: list of task objects
    obstacles: list of obstacle objects
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


class Obstacle:
    def __init__(self, pos, r):
        self.pos = pos
        self.r = r


def VFF(pos, goal, obstacles, k_att=1.0, k_rep=100.0, rr=100.0):
    """
    Virtual force field method for obstacle avoidance.
    pos: current position
    goal: goal position
    obstacles: list of obstacle objects
    k_att: attractive force coefficient
    k_rep: repulsive force coefficient
    rr: repulsive force range
    """
    att_force = k_att * (goal - pos)

    rep_force = np.zeros(2)

    for obstacle in obstacles:
        vec_o2r = pos - obstacle.pos

        dist_o2r = np.linalg.norm(vec_o2r)

        if dist_o2r <= rr:
            rep_force += k_rep * (1 / dist_o2r - 1 / rr) * vec_o2r / dist_o2r ** 3

    force = att_force + rep_force

    return force


# example
n_robots = 5
n_tasks = 10

robots = [Robot(i) for i in range(n_robots)]
tasks = [Task(i, np.random.rand(2) * 2000) for i in range(n_tasks)]
obstacles = [Obstacle(np.random.rand(2) * 2000, np.random.rand() * 50 + 50) for _ in range(5)]

for robot in robots:
    robot.pos = np.random.rand(2) * 2000

dt = 0.1

for t in range(1000):
    robots = CBBA(robots, tasks, obstacles)