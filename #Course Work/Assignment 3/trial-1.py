import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

class RRT:
    def __init__(self, start, goal, obstacle):
        self.start = start
        self.goal = goal
        self.obstacle = obstacle
        self.tree = [start]

    def generate_random_point(self):
        x = random.uniform(0, 10)
        y = random.uniform(0, 10)
        return np.array([x, y])

    def is_collision_free(self, point):
        # Check if the point is collision-free with the obstacle
        for i in range(len(self.obstacle)):
            p1 = self.obstacle[i]
            p2 = self.obstacle[(i + 1) % len(self.obstacle)]
            if self.line_intersects(point, self.start, p1, p2):
                return False
        return True

    def line_intersects(self, p, q, r, s):
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        return ccw(p, r, s) != ccw(q, r, s) and ccw(p, q, r) != ccw(p, q, s)

    def extend_tree(self, random_point):
        nearest_point = self.find_nearest_point(random_point)
        new_point = self.steer(nearest_point, random_point)
        if self.is_collision_free(new_point):
            self.tree.append(new_point)
        return new_point

    def find_nearest_point(self, point):
        distances = [np.linalg.norm(point - p) for p in self.tree]
        nearest_index = np.argmin(distances)
        return self.tree[nearest_index]

    def steer(self, from_point, to_point):
        max_distance = 0.5  # adjust as needed
        direction = to_point - from_point
        distance = min(max_distance, np.linalg.norm(direction))
        return from_point + (direction / np.linalg.norm(direction)) * distance

    def plot_rrt(self, ax):
        ax.plot(*zip(*self.tree), marker='o', color='b', linestyle='None')
        ax.plot(*zip(*self.obstacle + [self.obstacle[0]]), color='r')

    def plot_path(self, path, ax):
        ax.plot(*zip(*path), marker='o', color='g', linestyle='-', markersize=2)

    def rrt_algorithm(self, max_steps):
        for step in range(max_steps):
            plt.clf()  # Clear the current figure

            # Plot obstacle
            plt.plot(*zip(*self.obstacle + [self.obstacle[0]]), color='r')

            # Plot RRT tree
            self.plot_rrt(plt)

            random_point = self.generate_random_point()

            if self.is_collision_free(random_point):
                new_point = self.extend_tree(random_point)
                if np.linalg.norm(new_point - self.goal) < 0.1:
                    path = self.generate_path_to_goal()
                    print(f"Path found in step {step} with {len(path)} points.")
                    self.plot_rrt(plt)
                    self.plot_path(path, plt)
                    break

            # plt.pause(0.05)  # Adjust the pause duration as needed
            # plt.show(block=False)

    def generate_path_to_goal(self):
        distances = [np.linalg.norm(point - self.goal) for point in self.tree]
        goal_index = np.argmin(distances)
        path = self.smooth_path(self.tree[:goal_index + 1])
        return path

    def smooth_path(self, path):
        # Implement your path smoothing logic (optional)
        return path

start = np.array([1, 1])
goal = np.array([5,5])
obstacle = np.array([[3, 3], [3, 6], [6, 6], [6, 3]])

rrt = RRT(start, goal, obstacle)

# Create a figure for plotting
plt.figure()

# Run the RRT algorithm and display each step
max_steps = 1000
for _ in rrt.rrt_algorithm(max_steps):
    plt.waitforbuttonpress()

# Show the final plot
plt.show()