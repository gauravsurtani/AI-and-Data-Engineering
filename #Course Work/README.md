# AI-and-Data-Engineering

## Algorithm Documentation
### Dijkstra's Algorithm / A* Algorithm / Weighted A* Algorithm
---
### How to Run

Ensure you have Python installed on your system.
Check for the input.txt and coords.txt file which contain your nodes and its distances.
Open a terminal or command prompt and type the below
---
## Introduction

This document provides an overview and documentation for the Python code implementing Dijkstra's algorithm for finding the shortest path between nodes in a weighted graph. The code is organized around a `Graph` class that encapsulates the necessary data structures and methods for performing Dijkstra's algorithm.

The code provided uses several data structures:

`Dictionary (self.graph)`: Used to represent the graph. Keys represent vertices, and values are lists of tuples representing the neighboring vertices and their corresponding edge weights.

`Dictionary (visited)`: Used in Dijkstra's algorithm to keep track of visited vertices.

`Dictionary (dist)`: Used in Dijkstra's algorithm to store the shortest distances from the start vertex to each vertex.

`Dictionary (parent)`: Used in Dijkstra's algorithm to store the parent vertices in the shortest path tree.

`List (visited_vertices)`: Used to keep track of visited vertices during the execution of Dijkstra's algorithm.

`List (path)`: Used to store the vertices in the shortest path.

`List (distances_between)`: Used to store distances between nodes on the path.

## Class: Graph

### Constructor: `__init__(self)`

- Initializes a `Graph` object with an empty dictionary to store the graph representation.

### Method: `add_edge(self, u, v, w)`

- Adds a weighted edge from node `u` to node `v` with weight `w`. If node `u` is not already in the graph, it is added.

### Method: `min_distance(self, dist, visited)`

- Finds the vertex with the minimum distance value from the set of vertices that have not been visited.

### Method: `dijkstra(self, start)`

- Implements Dijkstra's algorithm to find the shortest paths from a specified starting vertex to all other vertices in the graph.
- Returns a tuple containing two dictionaries:
  - `dist`: Maps each vertex to its shortest distance from the starting vertex.
  - `parent`: Maps each vertex to its parent vertex in the shortest path tree.

### Method: `shortest_path_with_distances(self, start, end, parent)`

- Computes the shortest path from a specified starting vertex to an end vertex, along with the distances between nodes on the path.
- Returns a tuple containing two dictionaries:
  - `path`: Contains the vertices in the shortest path from `start` to `end`.
  - `distances`: Maps each vertex on the path to the distance from the previous vertex.

## Example Usage

```python
if __name__ == "__main__":
    visited_vertices = []
    graph = Graph()
    for each_path in connection_array:
        graph.add_edge(each_path[0], each_path[1], each_path[2])
     
    start_vertex = 6
    end_vertex = 94

    distances, parents = graph.dijkstra(start_vertex)
    path, distances_between = graph.shortest_path_with_distances(start_vertex, end_vertex, parents)
```

In this example, the code initializes a `Graph` object, adds edges, and then applies Dijkstra's algorithm to find the shortest path and distances between two specified vertices.

## Note

- The code contains commented sections for visualization and debugging purposes, such as plotting and displaying intermediate steps of the algorithm. These sections can be uncommented for more detailed visualization.

- Some print statements are commented out, which can be used for debugging or to provide additional information during execution.

- The code assumes that certain variables (`visited_vertices`, `connection_array`, etc.) are defined elsewhere in the program. These variables should be provided with appropriate values before running the code.

### A* Algorithm

A* Algorithm is a heuristic search algorithm that is similar to Dijkstra's Algorithm, but uses a heuristic function to guide the search towards the target node. The heuristic function estimates the distance from a node to the target node, and is used to prioritize the nodes in the search. A* Algorithm is often faster than Dijkstra's Algorithm, but requires a good heuristic function to be effective.

### Weighted A* Algorithm

Weighted A* Algorithm is a variant of A* Algorithm that allows the user to adjust the weight of the heuristic function. This can be useful in situations where the heuristic function is not accurate, or when the user wants to prioritize certain paths over others.

## Visualization

###  Dijkstra's Algorithm / A* Algorithm / Weighted A* Algorithm Example

![ Dijkstra's Algorithm / A* Algorithm / Weighted A* Algorithm ](/Assignment%202/dijkstras-Astar-weightedAstar.gif)

### Final Path Example

![Final Path Example](/#Course%20Work//Assignment%202/End.png)

## Implementation

The program is implemented in Python, and consists of several classes and functions:

- `Graph`: A class that represents a graph and provides methods for adding nodes and edges.
- `Node`: A class that represents a node in the graph and stores its coordinates and edges.
- `InputProcessor`: A class that reads input data from `input.txt` and `coords.txt` and generates `FormattedInput.txt`.
- `HeuristicDictionary`: A class that calculates heuristic distances for Weighted A* Algorithm.
- `ShortestPathFinder`: A class that implements Dijkstra's Algorithm, A* Algorithm, and Weighted A* Algorithm.
- `Visualizer`: A class that uses `matplotlib` to create visual representations of the graph and algorithm steps.
- `OutputGenerator`: A class that generates output data in `output.txt`.

To run the program, make sure `input.txt`, `coords.txt`, `FormattedInput.txt`, and `output.txt` are available, and run the program using `python

**--------------------------------------------------------------------------------------------------------------------------------------------------------------------**


# RRT and RRT* Algorithm Implementation

## Overview

This repository contains the implementation of the Rapidly-exploring Random Tree (RRT) and RRT* (RRT Star) algorithms for path planning. The implementation is written in Python and uses the matplotlib library for visualization.

## Visualization

### RRT* Tree Example

![RRT* Tree Example](/Assignment%203/rrt_animation.gif)

### Final Path Example

![Final Path Example](/Assignment%203/submissions/Figure%202023-10-23%20193614%20(892).png)


## Files

- **rrt_star.py**: Python script containing the implementation of the RRT and RRT* algorithms.
- **input.txt**: Input file specifying the start and goal coordinates, as well as the vertices of obstacles.
- **output.txt**: Output file containing goal distances and additional information.

## How to Run

1. Ensure you have Python installed on your machine.
2. Run the `rrt_star.py` script:
   ```bash
   python rrt_star.py
   ```
3. The program will read input from `input.txt`, perform the RRT* algorithm, visualize the results, and save goal distances to `output.txt`.

## Input File (input.txt)

The input file contains the following information:

- **Start Node**: Coordinates of the start node.
- **Goal Node**: Coordinates of the goal node.
- **Obstacle Vertices**: Vertices of polygonal obstacles.


## Visualization

The program generates visualizations of the RRT* tree, final path, and obstacles using matplotlib.

## Dependencies

- Python 3.x
- matplotlib library
- shapely library

Install dependencies using:
```bash
pip install matplotlib shapely
```

## Results

The RRT* algorithm successfully plans a path from the start node to the goal node while avoiding obstacles. The visualizations and goal distances are saved for analysis.

Feel free to modify parameters in the script for experimentation and testing.

For any questions or issues, please contact me at gaurav.surtani@sjsu.edu.

---