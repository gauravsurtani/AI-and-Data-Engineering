import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
from IPython.display import display, clear_output
import math


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
plt.title('Shortest Path')

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

heuristic_dict = [{}]
def setup_heuristic_dict(source,destination,weight):
    destination_point = pointer_dict[destination]
    for each_point in pointer_dict[1:]:
        x1,y1 = each_point['x'],each_point['y']
        x2,y2 = destination_point['x'],destination_point['y']
        heuristic_distance = (math.sqrt((x2 - x1)**2 + (y2 - y1)**2)) * weight
        heuristic_dict.append(heuristic_distance)
    

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
        no_of_iteration = 0
        dist[start] = 0

        for _ in range(len(self.graph)):
            u = self.min_distance(dist, visited)
            visited[u] = True
            visited_vertices.append(u)
            

            for v, w in self.graph[u]:
                if not visited[v] and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    parent[v] = u  
                
                # print(f'Iteration Number:{no_of_iteration}')
                # Uncomment this to remove all the multiple plots    
                # fig = plt.subplot()
                # setup_initial_plots()
                # plt.scatter([all_x], [all_y], color='lightgray', marker='o')
                # plt.plot([pointer_dict[u]['x'],pointer_dict[v]['x']],[pointer_dict[u]['y'],pointer_dict[v]['y']])
                # for index,vnum in enumerate(visited_vertices):
                #     plt.scatter(pointer_dict[visited_vertices[index]]['x'], pointer_dict[visited_vertices[index]]['y'],  color='gray', marker='x')
                # plt.xlabel('X-axis')
                # plt.ylabel('Y-axis')
                # plt.xlim(-2, 22)
                # plt.ylim(-2, 22)
                # clear_output(wait=True)
                # display(fig)
                # plt.legend()
                # plt.show()
                no_of_iteration +=1                    
        return dist, parent, no_of_iteration
        
    # def weighted_a_star(self, start, weight=1):
    #     visited = {vertex: False for vertex in self.graph}
    #     dist = {vertex: float('inf') for vertex in self.graph}
    #     parent = {vertex: None for vertex in self.graph}  # Store parent vertices
    #     no_of_iteration = 0
    #     dist[start] = 0

    #     for _ in range(len(self.graph)):
    #         u = self.min_distance(dist, visited)
    #         visited[u] = True
    #         visited_vertices.append(u)
    #         if u == end_vertex:
    #             # Early exit if the destination is reached
    #             break

    #         for v, w in self.graph[u]:
    #             if not visited[v] and dist[u] + w < dist[v]:
    #                 dist[v] = dist[u] + w + (heuristic_dict[v] * weight)
    #                 parent[v] = u  
    #             no_of_iteration +=1
                
    #             # Uncomment this to remove all the multiple plots    
    #             fig = plt.subplot()
    #             setup_initial_plots()
    #             plt.scatter([all_x], [all_y], color='lightgray', marker='o')
    #             plt.plot([pointer_dict[u]['x'],pointer_dict[v]['x']],[pointer_dict[u]['y'],pointer_dict[v]['y']])
    #             for index,vnum in enumerate(visited_vertices):
    #                 plt.scatter(pointer_dict[visited_vertices[index]]['x'], pointer_dict[visited_vertices[index]]['y'],  color='gray', marker='x')
    #             plt.xlabel('X-axis')
    #             plt.ylabel('Y-axis')
    #             plt.xlim(-2, 22)
    #             plt.ylim(-2, 22)                
    #             # clear_output(wait=True)
    #             display(fig)
    #             plt.legend()
    #             plt.show()           
    #     return dist, parent, no_of_iteration

    def a_star2(self, start, goal, weight):
        visited = {vertex: False for vertex in self.graph}
        g = {vertex: float('inf') for vertex in self.graph}  # Actual cost from start to vertex
        f = {vertex: float('inf') for vertex in self.graph}  # g + heuristic cost from vertex to goal
        parent = {vertex: None for vertex in self.graph}  # Store parent vertices
        no_of_iteration = 0
        g[start] = 0
        f[start] = heuristic_dict[start]
    
        for _ in range(len(self.graph)):
            u = self.min_distance(f, visited)  # Use f instead of g for minimum distance
            visited[u] = True
    
            if u == goal:  
                break
    
            for v, w in self.graph[u]:
                if not visited[v] and (g[u] + w) < g[v]:
                    g[v] = (g[u] + w)
                    f[v] = g[v] + (heuristic_dict[v] * weight) 
                    parent[v] = u  
                
                # Uncomment this to remove all the multiple plots    
                # fig = plt.subplot()
                # setup_initial_plots()
                # plt.scatter([all_x], [all_y], color='lightgray', marker='o')
                # plt.plot([pointer_dict[u]['x'],pointer_dict[v]['x']],[pointer_dict[u]['y'],pointer_dict[v]['y']])
                # for index,vnum in enumerate(visited_vertices):
                #     plt.scatter(pointer_dict[visited_vertices[index]]['x'], pointer_dict[visited_vertices[index]]['y'],  color='gray', marker='x')
                # plt.xlabel('X-axis')
                # plt.ylabel('Y-axis')
                # plt.xlim(-2, 22)
                # plt.ylim(-2, 22)
                # # clear_output(wait=True)
                # display(fig)
                # plt.legend()
                # plt.show()  
                no_of_iteration +=1                 
                      
        return g, parent, no_of_iteration

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
    
# FINAL PLOT
def final_plot(path,dist_array,title):
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
    plt.title(title)
    
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
    print("Program and Graph Generation in progress...")
    plt.show()
    
    with open('output.txt', 'a') as f:
        for index,p in enumerate(path):
            f.write(str(path[index]) + ' ')
        f.write('\n')
        f.write('0 ')
        for index,d in enumerate(dist_array):
            dist_array[index] = "{:.4f}".format(dist_array[index])
            f.write(str(dist_array[index]) + ' ')
        f.write('\n')
            
    

def create_empty_output_file():
    with open('output.txt', 'w') as f:
        pass

def run_particular_algorithm(algo_used,weight=1):
    if algo_used=='D':
        distances, parents,iterations = graph.dijkstra(start_vertex)    
        path, distances_between = graph.shortest_path_with_distances(start_vertex, end_vertex, parents)    
        intermediate_distances = 0
        dist_array = []
        for node, distance in distances_between.items():
            intermediate_distances = distance + intermediate_distances
            dist_array.append(intermediate_distances)
        final_plot(path,dist_array,'Dijkstra`s Algorithm   #Iterations:'+str(iterations))
       
    if algo_used=='WA':
        # distances, parents,iterations = graph.weighted_a_star(start_vertex,weight)
        distances, parents,iterations = graph.a_star2(start_vertex, end_vertex, weight)
        path, distances_between = graph.shortest_path_with_distances(start_vertex, end_vertex, parents)    
        intermediate_distances = 0
        dist_array = []
        for node, distance in distances_between.items():
            intermediate_distances = distance + intermediate_distances
            dist_array.append(intermediate_distances)
        title = 'Weighted A* | Weights ='+str(weight) + '  #Iterations:'+str(iterations)
        if weight==1:
            title = 'A*' + '  #Iterations:='+str(iterations)
        final_plot(path,dist_array,title)
        

if __name__ == "__main__":
    visited_vertices = []
    graph = Graph()
    create_empty_output_file()
    for each_path in connection_array:
        graph.add_edge(each_path[0],each_path[1],each_path[2])
     
    start_vertex = int(starting_vertex)
    end_vertex = int(target_vertex)
    
    # Setup Heuristic Distances from every node to the target node
    setup_heuristic_dict(start_vertex, end_vertex, 1)    
    
    run_particular_algorithm('D')
    run_particular_algorithm('WA')
    run_particular_algorithm('WA',2)
    run_particular_algorithm('WA',3)
    run_particular_algorithm('WA',4)
    run_particular_algorithm('WA',5)   
    
    # distances, parents = graph.dijkstra(start_vertex)    
    # path, distances_between = graph.shortest_path_with_distances(start_vertex, end_vertex, parents)    
    # intermediate_distances = 0
    # dist_array = []
    # for node, distance in distances_between.items():
    #     intermediate_distances = distance + intermediate_distances
    #     dist_array.append(intermediate_distances)
        
    
    # distances, parents = graph.weighted_a_star(start_vertex, 2)
    # distances, parents = graph.weighted_a_star(start_vertex, 3)
    # distances, parents = graph.weighted_a_star(start_vertex, 4)
    # distances, parents = graph.weighted_a_star(start_vertex, 5)
    
    
#print(*path)
#print(*dist_array)
print("Output File Generated.")

    
