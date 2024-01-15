import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import random

# Step 1: Get the Start , Goal and Obstacle from input file
input_file_name = "input.txt"
obstacle_coords = []
with open(input_file_name,'r') as file:
    input_file_lines = file.readlines()
    for index, line in enumerate(input_file_lines):
    #Step 1: Get the important parameters of the file.
        if(index == 0):
            start_vertex = line.strip()
            print(start_vertex)
        if(index == 1):
            goal_vertex = line.strip()
            print(goal_vertex)
        if(index > 1):
            obstacle_coords.append(line.strip())
    

# Step 2: Plot the Obstacle and Start and End Vertices:
def base_workspace_setup():
    x_coordinates, y_coordinates = zip(*[map(float, coord.split()) for coord in obstacle_coords])
    x_coordinates += (x_coordinates[0],)
    y_coordinates += (y_coordinates[0],)

    plt.plot(float(start_vertex.split()[0]),float(start_vertex.split()[1]), marker='o', color='g', label='Start Vertex')
    plt.plot(float(goal_vertex.split()[0]),float(goal_vertex.split()[1]), marker='o', color='r', label='End Vertex')
    plt.plot(x_coordinates, y_coordinates, marker='o', linestyle='-', color='gray')
    
    obstacle_polygon = plt.Polygon(list(zip(x_coordinates, y_coordinates)), edgecolor='none', facecolor='gray', alpha=0.75)
    plt.gca().add_patch(obstacle_polygon)
    
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('WorkSpace')
    plt.show()

# Call this base setup to check if its working.    
base_workspace_setup()


# Step 3: Setup RRT* Algorithm
class RRTStar:
    def __init__(self, start, goal, obstacle):
        self.start = start
        self.goal = goal
        self.obstacle = obstacle
        self.tree = [start]
        self.cost = {tuple(start): 0.0}
        
    def generate_random_point(self):
        x = random.randint(-10, 15)
        y = random.randint(-10, 15)
        # print(f'{x} - {y}')
        return np.array([x, y])
    
    def is_collision_free(self, point):
        for i in range(len(self.obstacle)):
            p1 = self.obstacle[i]
            p2 = self.obstacle[(i + 1) % len(self.obstacle)]
            if self.line_intersects(point, self.start, p1, p2):
                return False
        return True
    
    def line_intersects(self, p, q, r, s):
        pass
    
    def extend_tree(self, random_point):
        nearest_point, nearest_index = self.find_nearest_point(random_point)
        new_point = self.steer(nearest_point, random_point)
        if self.is_collision_free(new_point):
            if self.is_edge_collision_free(nearest_point, new_point):
                self.tree.append(new_point)
                self.rewire(nearest_point, nearest_index, new_point)
    
        return new_point
    
    def find_nearest_point(self, point):
        distances = [np.linalg.norm(point - p) for p in self.tree]
        nearest_index = np.argmin(distances)
        return self.tree[nearest_index], nearest_index

    def steer(self, from_point, to_point):
        max_distance = 1
        direction = to_point - from_point
        distance = min(max_distance, np.linalg.norm(direction))
        return from_point + (direction / np.linalg.norm(direction)) * distance

    def is_edge_collision_free(self, from_point, to_point):
        # Check if the edge is collision-free
        for i in range(len(self.obstacle)):
            p1 = self.obstacle[i]
            p2 = self.obstacle[(i + 1) % len(self.obstacle)]
            if self.line_intersects(from_point, to_point, p1, p2):
                return True
        return True

    def rewire(self, nearest_point, nearest_index, new_point):
        pass # TO DO

                
    def plot_rrt(self, ax):
        ax.plot(*zip(*self.tree), marker='o', color='b', linestyle='None')

    def plot_path(self, path, ax):
        ax.plot(*zip(*path), marker='o', color='g', linestyle='-', markersize=2)
        
    def plot_rrt_star(self, ax):
        ax.clear()
        # ax.fill(*zip(*self.obstacle), color='gray', alpha=0.5)
        ax.plot(*zip(*self.tree), marker='o', color='b', linestyle='None')
        ax.plot(*self.goal, marker='o', color='r', label='Goal Vertex')
        ax.plot(*self.start, marker='o', color='g', label='Start Vertex')
        ax.set_xlim([-10, 15])
        ax.set_ylim([-10, 15])
        ax.legend()
        plt.pause(0.01)

    def rrt_star_algorithm(self, max_steps):
        for step in range(max_steps):
            random_point = self.generate_random_point()

            if self.is_collision_free(random_point):
                new_point = self.extend_tree(random_point)
                self.plot_rrt_star(plt.gca())
                plt.draw()

                if np.linalg.norm(new_point - self.goal) < 2:
                    print("Goal reached!")
                    break

    def generate_path_to_goal(self):
        distances = [np.linalg.norm(point - self.goal) for point in self.tree]
        goal_index = np.argmin(distances)
        path = self.smooth_path(self.tree[:goal_index + 1])
        return path

    def smooth_path(self, path):
        # Implement your path smoothing logic (optional)
        return path
    

# Example usage:
start = np.array([0.00, 0.00])
goal = np.array([12.00, 12.00])
obstacle = np.array([
    [-5.00, 5.00],
    [4.00, 5.00],
    [4.00, -5.00],
    [-5.00, -5.00],
    [-5.00, -0.00],
    [-4.00, -0.00],
    [-4.00, -4.00],
    [3.00, -4.00],
    [3.00, 4.00],
    [-4.00, 4.00],
    [-4.00, 1.00],
    [-5.00, 1.00]
])

rrt_star = RRTStar(start, goal, obstacle)
rrt_star.rrt_star_algorithm(10000)
plt.show()

