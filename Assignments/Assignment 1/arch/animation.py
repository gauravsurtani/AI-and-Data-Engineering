import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
from IPython.display import display, clear_output

coords_array = []
pointer_ref_array = []
pointer = 1
starting_vertex = 6
target_vertex = 94

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

with open('coords.txt', 'r') as fFile:
    for line in fFile:
        SINGLE_LINE = line.strip().split(' ')
        coords_array.append(SINGLE_LINE)
        
pointer_dict = [{}]
all_x = []
all_y = []
for point in coords_array:
    x, y = float(point[0]), float(point[1])
    pointer_ref_array.append([pointer,(int(x),int(y))])
    all_x.append(int(x))
    all_y.append(int(y))
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

# Add labels and a legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.title('Scatter Plot Example')
plt.legend()

for each_connection in connection_array:
    source = int(float(each_connection[0]))
    destination = int(float(each_connection[1]))
    sourcePoint = pointer_ref_array[source-1][1]
    destinationPoint = pointer_ref_array[destination-1][1]
    
    sourceX = sourcePoint[0]
    sourceY = sourcePoint[1]
    destX = destinationPoint[0]
    destY = destinationPoint[1]    
    plt.plot([sourceX,destX],[sourceY,destY],c='lightgray', zorder=0)
    


#for i in range(len(x)):
 #   animate(i)
  #  clear_output(wait=True)
   # display(fig)

plt.show()