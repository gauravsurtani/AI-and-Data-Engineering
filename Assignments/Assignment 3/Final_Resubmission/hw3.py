import numpy as np
import matplotlib.pyplot as plt
import random
from shapely.geometry import Polygon, LineString
from matplotlib.patches import Circle
import math


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

def is_inside_circle(point, center, radius):
    distance_squared = (point[0] - center[0])**2 + (point[1] - center[1])**2
    return distance_squared <= radius**2

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

def extend_tree(tree, target_node, max_distance, obstacle_vertices, rewire_radius, goal_distance, goal_radius, paths):
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

        # Check if the new node is close to the goal
        if is_inside_circle((new_node.x, new_node.y), (goal_node.x, goal_node.y), goal_radius):
            current_distance = new_node.cost + distance(new_node, target_node)
            # Add the distance to the goal only if it changes
            if not goal_distance or current_distance != goal_distance[-1]:
                goal_distance.append(current_distance)
                # Store the path for visualization
                current_path = []
                current_path_node = new_node
                while current_path_node:
                    current_path.append((current_path_node.x, current_path_node.y))
                    current_path_node = current_path_node.parent
                paths.append(current_path)
                visualize_rrt_final(tree, paths[-1], obstacle_vertices, paths)

        # Visualize the current state of the RRT
        # visualize_rrt(tree, obstacle_vertices, target_node)

# Run the RRT* algorithm
def rrt_star(start, goal, iterations, max_distance, obstacle_vertices, rewire_radius, goal_radius, goal_distance):
    tree = [start]

    for _ in range(iterations):
        random_node = RRTNode(random.uniform(-10, 15), random.uniform(-10, 15))
        extend_tree(tree, random_node, max_distance, obstacle_vertices, rewire_radius, goal_distance, goal_radius, paths)

    # Continue extending the tree towards the goal even after initial connection
    while not any(node.x == goal.x and node.y == goal.y for node in tree):
        extend_tree(tree, goal, max_distance, obstacle_vertices, rewire_radius, goal_distance, goal_radius, paths)

    # Generate the final path
    final_path = []
    current_node = goal
    while current_node:
        final_path.append((current_node.x, current_node.y))
        current_node = current_node.parent

    return tree, final_path

# Visualize the RRT* tree and final path
# Updated visualize_rrt_final function
def visualize_rrt_final(tree, final_path, obstacle_vertices, paths, target_radius=0.5):
    plt.clf()
    plt.figure(figsize=(10, 10))
    
    # Draw the obstacle
    obstacle_vertices.append(obstacle_vertices[0])  # Close the polygon
    obstacle_x, obstacle_y = zip(*obstacle_vertices)
    plt.fill(obstacle_x, obstacle_y, color='gray', alpha=0.5)
    
    # Plot start node
    plt.plot(start_node.x, start_node.y, color='green', marker='x', label='Start Node',zorder=2)
    
    # Plot target node
    plt.plot(goal_node.x, goal_node.y, color='red', marker='x', label='Goal Node',zorder=2)

    # Draw the tree edges
    for node in tree:
        if node.parent:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], color='gray')

    # Scatter plot for tree nodes
    plt.scatter(*zip(*[(node.x, node.y) for node in tree]), color='lightgray')

    # Scatter plot for the final path
    plt.scatter(*zip(*final_path), color='green', marker='o',zorder=2)
    
    
    # Plot the circle around the target node
    target_circle = Circle((final_path[0][0], final_path[0][1]), target_radius, fill=False, color='red',linewidth=2)
    plt.gca().add_patch(target_circle)

    # Plot all stored paths
    for path in paths:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, color='green', linestyle='dashed')

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
    goal_radius = 0.5
    paths = []  # Store paths for visualization

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
    iterations = 4000
    print(f'{iterations}: Max Iterations')

    # Obstacle definition
    obstacle_vertices = [tuple(map(float, point.split())) for point in obstacle_coords]

    # Run the RRT* algorithm
    final_tree, path_to_goal = rrt_star(start_node, goal_node, iterations, max_dist, obstacle_vertices, rewire_radius, goal_radius, goal_distance)
    # Visualize the result
    # visualize_rrt_final(final_tree, path_to_goal, obstacle_vertices,paths)
    # print(f'{len(goal_distance)}: Goal distance')
    # print(f'{goal_distance}: Goal distance')
    # print(f'{path_to_goal}: Goal Nodes')
    
    def distance_between_points(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Given path coordinates
    path = [(0, 0), (-5, 1), (-5, 5), (12, 12)]
    
    # Calculate total distance
    total_distance = sum(distance_between_points(path[i], path[i+1]) for i in range(len(path)-1))
    
    print(f'Total Distance: {total_distance}')

    
    output_file_name = "output.txt"
    with open(output_file_name, 'w') as output_file:
        output_file.write(f"{goal_distance[-1]}\n") 
        output_file.write(f"{goal_distance[0]}\n")
        output_file.write(f"{total_distance}\n")

    print(f'Goal Distances: {goal_distance}')
    print(f'Goal distances are saved to {output_file_name}')
    
    