import numpy as np
import matplotlib.pyplot as plt
import random

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

def distance(node1, node2):
    return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

def nearest_node(nodes, target_node):
    return min(nodes, key=lambda node: distance(node, target_node))

def steer(node1, node2, delta_q):
    if distance(node1, node2) < delta_q:
        return node2
    else:
        theta = np.arctan2(node2.y - node1.y, node2.x - node1.x)
        new_x = node1.x + delta_q * np.cos(theta)
        new_y = node1.y + delta_q * np.sin(theta)
        return Node(new_x, new_y)

def is_obstacle_free(node1, node2):
    # Add your obstacle checking logic here
    return True

def extend(tree, q_rand, delta_q):
    q_near = nearest_node(tree, q_rand)
    q_new = steer(q_near, q_rand, delta_q)

    if is_obstacle_free(q_near, q_new):
        q_new.parent = q_near
        q_new.cost = q_near.cost + distance(q_near, q_new)

        # Rewire step
        near_nodes = [node for node in tree if distance(q_new, node) < delta_q]
        for near_node in near_nodes:
            if q_new.cost + distance(q_new, near_node) < near_node.cost:
                near_node.parent = q_new
                near_node.cost = q_new.cost + distance(q_new, near_node)

        tree.append(q_new)

def rrt_star(start, goal, num_iterations, delta_q):
    tree = [start]

    for _ in range(num_iterations):
        q_rand = Node(random.uniform(0, 10), random.uniform(0, 10))
        extend(tree, q_rand, delta_q)

    # Connect to goal
    extend(tree, goal, delta_q)

    # Final path
    path = []
    current_node = goal
    while current_node:
        path.append((current_node.x, current_node.y))
        current_node = current_node.parent

    return tree, path

def plot_rrt(tree, path):
    plt.figure(figsize=(8, 8))
    for node in tree:
        if node.parent:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], color='blue')
    plt.scatter(*zip(*[(node.x, node.y) for node in tree]), color='blue')
    plt.scatter(*zip(*path), color='red', marker='x')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('RRT* Algorithm')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    start_node = Node(1, 1)
    goal_node = Node(9, 9)

    iterations = 500
    delta = 0.5

    tree, path = rrt_star(start_node, goal_node, iterations, delta)

    plot_rrt(tree, path)
