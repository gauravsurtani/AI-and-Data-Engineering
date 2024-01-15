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

def is_obstacle_free(node1, node2, obstacle_vertices):
    # Check if the line segment between node1 and node2 intersects with the obstacle
    for i in range(len(obstacle_vertices)):
        x1, y1 = obstacle_vertices[i]
        x2, y2 = obstacle_vertices[(i + 1) % len(obstacle_vertices)]

        # Check if the line segment intersects with the obstacle edge
        if do_segments_intersect(node1.x, node1.y, node2.x, node2.y, x1, y1, x2, y2):
            return False

    return True

def do_segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    def orientation(x, y, x1, y1, x2, y2):
        return (y2 - y1) * (x - x2) - (x2 - x1) * (y - y2)

    o1 = orientation(x3, y3, x4, y4, x1, y1)
    o2 = orientation(x3, y3, x4, y4, x2, y2)
    o3 = orientation(x1, y1, x2, y2, x3, y3)
    o4 = orientation(x1, y1, x2, y2, x4, y4)

    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True

    return False

def extend(tree, q_rand, delta_q, obstacle_vertices):
    q_near = nearest_node(tree, q_rand)
    q_new = steer(q_near, q_rand, delta_q)

    if is_obstacle_free(q_near, q_new, obstacle_vertices):
        q_new.parent = q_near
        q_new.cost = q_near.cost + distance(q_near, q_new)

        # Rewire step
        near_nodes = [node for node in tree if distance(q_new, node) < delta_q]
        for near_node in near_nodes:
            if q_new.cost + distance(q_new, near_node) < near_node.cost:
                near_node.parent = q_new
                near_node.cost = q_new.cost + distance(q_new, near_node)

        tree.append(q_new)

def rrt_star(start, goal, num_iterations, delta_q, obstacle_vertices):
    tree = [start]

    for _ in range(num_iterations):
        q_rand = Node(random.uniform(-10, 15), random.uniform(-10, 15))
        extend(tree, q_rand, delta_q, obstacle_vertices)

    # Connect to goal
    extend(tree, goal, delta_q, obstacle_vertices)

    # Final path
    path = []
    current_node = goal
    while current_node:
        path.append((current_node.x, current_node.y))
        current_node = current_node.parent

    return tree, path

def plot_rrt(tree, path, obstacle_vertices):
    plt.figure(figsize=(8, 8))

    # Plot obstacle
    obstacle_vertices.append(obstacle_vertices[0])  # Close the polygon
    obstacle_x, obstacle_y = zip(*obstacle_vertices)
    plt.fill(obstacle_x, obstacle_y, color='gray', alpha=0.5)

    for node in tree:
        if node.parent:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], color='blue')
    plt.scatter(*zip(*[(node.x, node.y) for node in tree]), color='blue')
    plt.scatter(*zip(*path), color='red', marker='o',zorder=2)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('RRT* Algorithm with Obstacle')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    start_node = Node(0, 0)
    goal_node = Node(12, 12)

    iterations = 2000
    delta = 0.75

    obstacle_vertices = [
        (-5.00, 5.00), (4.00, 5.00), (4.00, -5.00),
        (-5.00, -5.00), (-5.00, 0.00), (-4.00, 0.00),
        (-4.00, -4.00), (3.00, -4.00), (3.00, 4.00),
        (-4.00, 4.00), (-4.00, 1.00), (-5.00, 1.00)
    ]

    tree, path = rrt_star(start_node, goal_node, iterations, delta, obstacle_vertices)

    plot_rrt(tree, path, obstacle_vertices)
