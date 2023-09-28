import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
from IPython.display import display, clear_output

pointer = 1
pointer_ref_array = []
X_coords = []
Y_coords = []
dist_array = []
coords_array = []
all_x = []
all_y = []

input_file_name = "input.txt"
with open(input_file_name,'r') as file:
    input_file_lines = file.readlines()
    for index, line in enumerate(input_file_lines):
    #Step 1: Get the important parameters of the file.
        if(index == 0):
            vertice_count = line.strip()
        if(index == 1):
            starting_vertex = line.strip()
        if(index == 2):
            target_vertex = line.strip()
            break

    new_file_lines = input_file_lines[3:]    
    with open('FormattedInput.txt','w') as newfile:
        newfile.writelines(new_file_lines)

#print(f'Count of Vertices: {vertice_count}')
#print(f'Starting Vertex: {starting_vertex}')
#print(f'Target Vertex: {target_vertex}')

connection_array = []
with open('FormattedInput.txt','r') as fFile:
    for line in fFile:
            SINGLE_LINE = line.strip().split(' ')
            connection_array.append(SINGLE_LINE)

# Convert all strings to either to Integer or Float            
for index,eachline in enumerate(connection_array):
    connection_array[index][0] = int(connection_array[index][0])
    connection_array[index][1] = int(connection_array[index][1])
    connection_array[index][2] = float(connection_array[index][2])

# Create a scatter plot    
# for connecting 1 -> 2  [0,0] -> [0,2] 

with open('coords.txt', 'r') as fFile:
    for line in fFile:
        SINGLE_LINE = line.strip().split(' ')
        coords_array.append(SINGLE_LINE)

pointer_dict = [{}]
for point in coords_array:
    x, y = float(point[0]), float(point[1])
    all_x.append(int(x))
    all_y.append(int(y))
    pointer_ref_array.append([pointer,(int(x),int(y))])
    pointer_dict.append({'x': x , 'y': y})
    if int(pointer) == int(starting_vertex):
        plt.scatter(x, y, marker='x', color='red')
        plt.annotate(f'{pointer}', (x, y), textcoords="offset points", xytext=(0, 5), ha='right',size=7)
        pointer += 1 
        continue
        
    if int(pointer) == int(target_vertex):
        plt.scatter(x, y, marker='x', color='green')
        plt.annotate(f'{pointer}', (x, y), textcoords="offset points", xytext=(0, 5), ha='left',size=7)
        pointer += 1 
        continue
        
    plt.scatter(x, y, marker='o', color='gray')
    # Add labels to the points
    plt.annotate(f'{pointer}', (x, y), textcoords="offset points", xytext=(0, 5), ha='center',size=7,zorder=1)    
    pointer += 1    
    
plt.xticks(range(-2,24,2))
plt.yticks(range(-2,24,2))

# Set labels for x and y axes
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Dijkstra algorithm')

def setup_initial_plots():
    for each_connection in connection_array:
        source = int(float(each_connection[0]))
        destination = int(float(each_connection[1]))
        sourcePoint = pointer_ref_array[source-1][1]
        destinationPoint = pointer_ref_array[destination-1][1]
        
        #     Simplifying the variables
        sourceX = sourcePoint[0]
        sourceY = sourcePoint[1]
        destX = destinationPoint[0]
        destY = destinationPoint[1]    
        #     print(f'{sourceX},{sourceY} --> {destX},{destY}');
        plt.plot([sourceX,destX],[sourceY,destY],c='lightgray', zorder=0)
        
setup_initial_plots();

class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v, w):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append((v, w))

    def min_distance(self, dist, visited):
        min_dist = float('inf')
        min_vertex = None
        for vertex in self.graph:
            if not visited[vertex] and dist[vertex] < min_dist:
                min_dist = dist[vertex]
                min_vertex = vertex
        return min_vertex

    def dijkstra(self, start):
        visited = {vertex: False for vertex in self.graph}
        dist = {vertex: float('inf') for vertex in self.graph}
        parent = {vertex: None for vertex in self.graph}  # Store parent vertices

        dist[start] = 0

        for _ in range(len(self.graph)):
            u = self.min_distance(dist, visited)
            visited[u] = True
            visited_vertices.append(u)

            for v, w in self.graph[u]:
                if not visited[v] and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    parent[v] = u  
                    
                    # Uncomment this to remove all the multiple plots    
                    fig = plt.subplot()
                    setup_initial_plots()
                    plt.scatter([all_x], [all_y], color='lightgray', marker='o')
                    plt.plot([pointer_dict[u]['x'],pointer_dict[v]['x']],[pointer_dict[u]['y'],pointer_dict[v]['y']])
                    for index,vnum in enumerate(visited_vertices):
                        plt.scatter(pointer_dict[visited_vertices[index]]['x'], pointer_dict[visited_vertices[index]]['y'],  color='gray', marker='x')
                    plt.xlabel('X-axis')
                    plt.ylabel('Y-axis')
                    plt.xlim(-2, 22)
                    plt.ylim(-2, 22)
                    clear_output(wait=True)
                    display(fig)
                    plt.legend()
                    plt.show()  
                    
        return dist, parent

    def shortest_path_with_distances(self, start, end, parent):
        path = []
        current = end
        distances = {}

        while current is not None:
            path.append(current)
            current = parent[current]

        path.reverse()

        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            for neighbor, weight in self.graph[u]:
                if neighbor == v:
                    distances[v] = weight
                    break
        return path, distances

if __name__ == "__main__":
    visited_vertices = []
    graph = Graph()
    for each_path in connection_array:
        graph.add_edge(each_path[0],each_path[1],each_path[2])
     
    start_vertex = int(starting_vertex)
    end_vertex = int(target_vertex)

    distances, parents = graph.dijkstra(start_vertex)
    path, distances_between = graph.shortest_path_with_distances(start_vertex, end_vertex, parents)
    
    #print(f'Shortest distance from {start_vertex} to {end_vertex} is {distances[end_vertex]}')
    #print(*path)
    #print('Distances between nodes on the path:')
    intermediate_distances = 0
    dist_array = []
    for node, distance in distances_between.items():
        #print(f'Distance from {path[path.index(node) - 1]} to {node} is {distance}')
        intermediate_distances = distance + intermediate_distances
        #intermediate_distances = "{0:.4f}".format(intermediate_distances)
        dist_array.append(intermediate_distances)
        #print(distance,end=' ')

#print(*path)
#print(*dist_array)
    
# FINAL PLOT
pointer = 1
for point in coords_array:
    x, y = float(point[0]), float(point[1])
    if int(pointer) == int(starting_vertex):
        plt.scatter(x, y, marker='x', color='red')
        plt.annotate(f'{pointer}', (x, y), textcoords="offset points", xytext=(0, 5), ha='right',size=7)
        pointer += 1 
        continue
        
    if int(pointer) == int(target_vertex):
        plt.scatter(x, y, marker='x', color='green')
        plt.annotate(f'{pointer}', (x, y), textcoords="offset points", xytext=(0, 5), ha='left',size=7)
        pointer += 1 
        continue
        
    plt.scatter(x, y, marker='o', color='gray')
    plt.annotate(f'{pointer}', (x, y), textcoords="offset points", xytext=(0, 5), ha='center',size=7,zorder=1)    
    pointer += 1  
      
plt.xticks(range(-2,24,2))
plt.yticks(range(-2,24,2))

# Set labels for x and y axes
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Dijkstra algorithm')

for each_connection in connection_array:
    source = int(float(each_connection[0]))
    destination = int(float(each_connection[1]))
    sourcePoint = pointer_ref_array[source-1][1]
    destinationPoint = pointer_ref_array[destination-1][1]
    
    #     Simplifying the variables
    sourceX = sourcePoint[0]
    sourceY = sourcePoint[1]
    destX = destinationPoint[0]
    destY = destinationPoint[1]    
    #     print(f'{sourceX},{sourceY} --> {destX},{destY}');    
    plt.plot([sourceX,destX],[sourceY,destY],c='lightgray', zorder=0)

plt.scatter([all_x], [all_y], color='lightgray', marker='o')
plt.scatter([pointer_dict[start_vertex]['x']], [pointer_dict[start_vertex]['y']], color='green', marker='x',zorder=1)
plt.scatter([pointer_dict[end_vertex]['x']], [pointer_dict[end_vertex]['y']], color='red', marker='x',zorder=1)
for index,src in enumerate(path[:-1]):
    src = pointer_dict[path[index]]
    dest = pointer_dict[path[index+1]]
    if(dest != pointer_dict[end_vertex]):
        plt.scatter(dest['x'],dest['y'],color='blue',zorder=1)
    plt.plot([src['x'],dest['x']],[src['y'],dest['y']],c='blue', zorder=0)
    
# Display the plot
print("Program Completed and Graph Generated.")
plt.show()

with open('output.txt', 'w') as f:
    for index,p in enumerate(path):
        f.write(str(path[index]) + ' ')
    f.write('\n')
    f.write('0 ')
    for index,d in enumerate(dist_array):
        dist_array[index] = "{:.4f}".format(dist_array[index])
        f.write(str(dist_array[index]) + ' ')
        
print("Output File Generated.")