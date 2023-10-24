import numpy as np
import matplotlib.pyplot as plt
import random
from shapely.geometry import Polygon, LineString
random.seed(3)

# A node in our RRT (Randomly-exploring Random Tree)
class RRTNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

# Calculate the Euclidean distance between two nodes
def distance(node1, node2):
    return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

# Find the node in 'nodes' that is nearest to 'target_node'
def find_nearest(nodes, target_node):
    return min(nodes, key=lambda node: distance(node, target_node))

# Steer towards the target node with a maximum distance
def steer(node1, node2, max_distance):
    if distance(node1, node2) < max_distance:
        return node2
    else:
        angle = np.arctan2(node2.y - node1.y, node2.x - node1.x)
        new_x = node1.x + max_distance * np.cos(angle)
        new_y = node1.y + max_distance * np.sin(angle)
        return RRTNode(new_x, new_y)

# Check if the path between two nodes is obstacle-free
def is_path_clear(node1, node2, obstacle_vertices):
    line = LineString([(node1.x, node1.y), (node2.x, node2.y)])
    obstacle_polygon = Polygon(obstacle_vertices)

    # Check if the line segment intersects with the obstacle
    if line.intersects(obstacle_polygon):
        return False
    else:
        return True

# Extend the tree with a new node
def extend_tree(tree, target_node, max_distance, obstacle_vertices, rewire_radius):
    nearest_node = find_nearest(tree, target_node)
    new_node = steer(nearest_node, target_node, max_distance)

    if is_path_clear(nearest_node, new_node, obstacle_vertices):
        new_node.parent = nearest_node
        new_node.cost = nearest_node.cost + distance(nearest_node, new_node)

        # Rewire the tree
        near_nodes = [node for node in tree if distance(new_node, node) < rewire_radius]
        for near_node in near_nodes:
            if new_node.cost + distance(new_node, near_node) < near_node.cost:
                near_node.parent = new_node
                near_node.cost = new_node.cost + distance(new_node, near_node)

        tree.append(new_node)

# Run the RRT* algorithm
def rrt_star(start, goal, iterations, max_distance, obstacle_vertices, rewire_radius):
    tree = [start]

    for _ in range(iterations):
        random_node = RRTNode(random.uniform(-10, 15), random.uniform(-10, 15))
        extend_tree(tree, random_node, max_distance, obstacle_vertices, rewire_radius)

    # Connect to the goal
    extend_tree(tree, goal, max_distance, obstacle_vertices, rewire_radius)

    # Generate the final path
    final_path = []
    current_node = goal
    while current_node:
        final_path.append((current_node.x, current_node.y))
        current_node = current_node.parent

    return tree, final_path

# Visualize the RRT* tree and final path
def visualize_rrt(tree, final_path, obstacle_vertices):
    plt.figure(figsize=(8, 8))

    # Draw the obstacle
    obstacle_vertices.append(obstacle_vertices[0])  # Close the polygon
    obstacle_x, obstacle_y = zip(*obstacle_vertices)
    plt.fill(obstacle_x, obstacle_y, color='gray', alpha=0.5)

    # Draw the tree edges
    for node in tree:
        if node.parent:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], color='blue')

    # Scatter plot for tree nodes
    plt.scatter(*zip(*[(node.x, node.y) for node in tree]), color='blue')

    # Scatter plot for the final path
    plt.scatter(*zip(*final_path), color='red', marker='x')

    # Labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('RRT* Algorithm with Obstacle')
    plt.grid(True)
    plt.show()

# Main program
if __name__ == "__main__":
    # Start and goal nodes
    start_node = RRTNode(0, 0)
    goal_node = RRTNode(12, 12)

    # Algorithm parameters
    iterations = 2000
    max_dist = 0.75
    rewire_radius = 0.75

    # Obstacle definition
    obstacle_vertices = [
        (-5.00, 5.00), (4.00, 5.00), (4.00, -5.00),
        (-5.00, -5.00), (-5.00, 0.00), (-4.00, 0.00),
        (-4.00, -4.00), (3.00, -4.00), (3.00, 4.00),
        (-4.00, 4.00), (-4.00, 1.00), (-5.00, 1.00)
    ]

    # Run the RRT* algorithm
    final_tree, path_to_goal = rrt_star(start_node, goal_node, iterations, max_dist, obstacle_vertices, rewire_radius)

    # Visualize the result
    visualize_rrt(final_tree, path_to_goal, obstacle_vertices)
