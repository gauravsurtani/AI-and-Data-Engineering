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
    x_difference = node1.x - node2.x
    y_difference = node1.y - node2.y
    distance_squared = x_difference ** 2 + y_difference ** 2
    distance_node = np.sqrt(distance_squared)
    return distance_node

# Find the node in 'nodes' that is nearest to 'target_node'
def find_nearest(nodes, target_node):
    min_node = None
    min_distance = float('inf')

    for node in nodes:
        current_distance = distance(node, target_node)
        if current_distance < min_distance:
            min_distance = current_distance
            min_node = node

    return min_node

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

# To fix for RRT* Algorithm
def extend_tree_RRT(tree, target_node, max_distance, obstacle_vertices, rewire_radius):
    nearest_node = find_nearest(tree, target_node)
    new_node = steer(nearest_node, target_node, max_distance)
    # print(nearest_node)
    # print(new_node)
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
        
        if distance(new_node, target_node) < max_distance:
            goal_distance.append(new_node.cost + distance(new_node, target_node))
            if target_node == goal_node:
                pass       
                ## everytime distance to target node changes we add a goal distance

def extend_tree(tree, target_node, max_distance, obstacle_vertices, rewire_radius, goal_distance):
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

        if new_node.x == goal_node.x and new_node.y == goal_node.y:
            goal_distance.append(new_node.cost)
        # Check if the new node is close to the goal
        if distance(new_node, target_node) < max_distance:
            current_distance = new_node.cost + distance(new_node, target_node)
            # Add the distance to the goal only if it changes
            if not goal_distance or current_distance != goal_distance[-1]:
                pass
                # goal_distance.append(current_distance)
        # Visualize the current state of the RRT
        # visualize_rrt(tree, obstacle_vertices, target_node)

# Run the RRT* algorithm
def rrt_star(start, goal, iterations, max_distance, obstacle_vertices, rewire_radius):
    tree = [start]

    for _ in range(iterations):
        
        random_node = RRTNode(random.uniform(-10, 15), random.uniform(-10, 15))
        #print(random_node)
        # random number should not be start or end
        if random_node == start_node :
            pass
        if random_node == RRTNode(12, 12) :
            pass        
        extend_tree(tree, random_node, max_distance, obstacle_vertices, rewire_radius, goal_distance)

    # Connect to the goal
    extend_tree(tree, goal, max_distance, obstacle_vertices, rewire_radius, goal_distance)

    # Generate the final path
    final_path = []
    current_node = goal
    while current_node:
        final_path.append((current_node.x, current_node.y))
        current_node = current_node.parent

    return tree, final_path

# Visualize the RRT* tree and final path
def visualize_rrt_final(tree, final_path, obstacle_vertices):
    plt.clf()
    plt.figure(figsize=(10, 10))
    
   
    # Draw the obstacle
    obstacle_vertices.append(obstacle_vertices[0])  # Close the polygon
    obstacle_x, obstacle_y = zip(*obstacle_vertices)
    plt.fill(obstacle_x, obstacle_y, color='gray', alpha=0.5)

    # Draw the tree edges
    for node in tree:
        if node.parent:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], color='gray')

    # Scatter plot for tree nodes
    plt.scatter(*zip(*[(node.x, node.y) for node in tree]), color='lightgray')

    # Scatter plot for the final path
    plt.scatter(*zip(*final_path), color='green', marker='o',zorder=2)

    # Labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('RRT* Algorithm with a polygon obstacle')
    plt.grid(True)
    plt.show()
    
# Function to visualize the RRT
def visualize_rrt(tree, obstacle_vertices, target_node):
    plt.clf()  # Clear the previous plot 
    
    # Plot obstacles
    obstacle_vertices.append(obstacle_vertices[0])  # Close the polygon
    obstacle_x, obstacle_y = zip(*obstacle_vertices)
    plt.fill(obstacle_x, obstacle_y, color='gray', alpha=0.5)

    # Plot tree edges
    for node in tree:
        if node.parent:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], color='blue')

    # Plot target node
    # plt.scatter(goal_node.x, goal_node.y, color='blue', marker='x', label='Random Node')
    
    # Plot start node
    plt.scatter(start_node.x, start_node.y, color='green', marker='x', label='Start Node')
    
    # Plot target node
    plt.scatter(goal_node.x, goal_node.y, color='red', marker='x', label='Goal Node')

    # Set axis limits
    plt.xlim(-12, 15)
    plt.ylim(-12, 15)

    # Add labels and legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

    # Show the plot
    plt.pause(0.01)

# Main program
if __name__ == "__main__":
    
    #Get the Start , Goal and Obstacle from input file
    goal_distance = []
    input_file_name = "input.txt"
    obstacle_coords = []
    with open(input_file_name,'r') as file:
        input_file_lines = file.readlines()
        for index, line in enumerate(input_file_lines):
        #Step 1: Get the important parameters of the file.
            if(index == 0):
                start_vertex = line.strip()
                sv1 = int(float(start_vertex.strip().split()[0]))
                sv2 = int(float(start_vertex.strip().split()[1]))
                start_node = RRTNode(sv1, sv2)
            if(index == 1):
                goal_vertex = line.strip()
                gv1 = int(float(goal_vertex.strip().split()[0]))
                gv2 = int(float(goal_vertex.strip().split()[1]))
                goal_node = RRTNode(gv1, gv2)
            if(index > 1):
                obstacle_coords.append(line.strip())

    # Algorithm parameters
    
    max_dist = 1
    print(f'{max_dist}: length of node we are using')
    rewire_radius = 0.75
    print(f'{rewire_radius}: Radius used to find nearest node')
    iterations = 2000
    print(f'{iterations}: Max Iterations')

    # Obstacle definition
    obstacle_vertices = [tuple(map(float, point.split())) for point in obstacle_coords]

    # Run the RRT* algorithm
    final_tree, path_to_goal = rrt_star(start_node, goal_node, iterations, max_dist, obstacle_vertices, rewire_radius)

    # Visualize the result
    visualize_rrt_final(final_tree, path_to_goal, obstacle_vertices)
    # print(f'{len(goal_distance)}: Goal distance')
    # print(f'{goal_distance}: Goal distance')
    # print(f'{path_to_goal}: Goal Nodes')
    