# Dijkstra's Algorithm Documentation

---
## How to Run

Ensure you have Python installed on your system.
Check for the input.txt and coords.txt file which contain your nodes and its distances.
Open a terminal or command prompt and type the below
---

```
python .\Final_file.py
```

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
