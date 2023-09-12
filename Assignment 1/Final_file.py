import matplotlib.pyplot as plt

pointer = 1
pointer_ref_array = []
X_coords = []
Y_coords = []
dist_array = []
coords_array = []

input_file_name = "input.txt"
with open(input_file_name,'r') as file:
#   file_contents = file.read();
    input_file_lines = file.readlines()
    for index, line in enumerate(input_file_lines):
#       Step 1: Get the important parameters of the file.
#       print(f'{index}, {line.strip()}')
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
            # Process each line here
            # print(line.strip())
            SINGLE_LINE = line.strip().split(' ')
            connection_array.append(SINGLE_LINE)

# Convert all strings to either to Integer or Float            
for index,eachline in enumerate(connection_array):
    connection_array[index][0] = int(connection_array[index][0])
    connection_array[index][1] = int(connection_array[index][1])
    connection_array[index][2] = float(connection_array[index][2])

with open('coords.txt','r') as fFile:
    for line in fFile:
            SINGLE_LINE = line.strip().split(' ')       
            #plt.plot(float(SINGLE_LINE[0]),float(SINGLE_LINE[1]),marker='*', color='blue',)

# Create a scatter plot    
# for connecting 1 -> 2  [0,0] -> [0,2] 

with open('coords.txt', 'r') as fFile:
    for line in fFile:
        SINGLE_LINE = line.strip().split(' ')
        coords_array.append(SINGLE_LINE)

for point in coords_array:
    x, y = float(point[0]), float(point[1])
    pointer_ref_array.append([pointer,(int(x),int(y))])
    if int(pointer) == int(starting_vertex):
        plt.scatter(x, y, marker='x', color='blue')
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


# Setting up all the connections between all points in the system
for each_connection in connection_array:
    #     print(each_connection)
    source = int(float(each_connection[0]))
    destination = int(float(each_connection[1]))
    sourcePoint = pointer_ref_array[source-1][1]
    destinationPoint = pointer_ref_array[destination-1][1]
    #print(f'{source} --> {destination}')
    #print(f'{sourcePoint} --> {destinationPoint}')

    #     print(f'Source:\t\t{pointer_ref_array[(int(each_connection[0])-1)][1]}')
    #     print(f'Desination:\t{pointer_ref_array[(int(each_connection[1])-1)][1]}')
    #     print(f'SourceX:\t{pointer_ref_array[(int(each_connection[0])-1)][1][0]}')
    #     print(f'DesinationX:\t{pointer_ref_array[(int(each_connection[1])-1)][1][0]}')
    #     print(f'SourceY:\t{pointer_ref_array[(int(each_connection[0])-1)][1][1]}')
    #     print(f'DesinationY:\t{pointer_ref_array[(int(each_connection[1])-1)][1][1]}')
    
    #     Simplifying the variables
    sourceX = sourcePoint[0]
    sourceY = sourcePoint[1]
    destX = destinationPoint[0]
    destY = destinationPoint[1]    
    #     print(f'{sourceX},{sourceY} --> {destX},{destY}');
    plt.plot([sourceX,destX],[sourceY,destY],c='lightgray', zorder=0)

# Display the plot
plt.show()


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

            for v, w in self.graph[u]:
                if not visited[v] and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    parent[v] = u  # Update parent vertex

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

# Example usage:
if __name__ == "__main__":
    graph = Graph()
    for each_path in connection_array:
        graph.add_edge(each_path[0],each_path[1],each_path[2])
     
    start_vertex = 6
    end_vertex = 94  # Replace with the desired end vertex

    distances, parents = graph.dijkstra(start_vertex)
    path, distances_between = graph.shortest_path_with_distances(start_vertex, end_vertex, parents)
    
    #print(f'Shortest distance from {start_vertex} to {end_vertex} is {distances[end_vertex]}')
    print(f'{" ".join(map(str, path))}')
    #print('Distances between nodes on the path:')
    
    #dist_array = []
    for node, distance in distances_between.items():
        #print(f'Distance from {path[path.index(node) - 1]} to {node} is {distance}')
        #dist_array.append(distance)
        print(distance,end=' ')
