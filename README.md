# Reinforcement Learning for Algorithm Optimization – Optimize sorting or searching techniques
# Q-Learning and A* Pathfinding Comparison

This Python project compares the performance of **Q-Learning** and **A\*** algorithms in finding the shortest path between two nodes in a weighted graph. The project utilizes **Q-Learning** as a reinforcement learning algorithm to iteratively learn and improve its pathfinding abilities, and **A\*** as a classical heuristic-based pathfinding algorithm. The comparison is visualized using **NetworkX** and **Matplotlib**.

## Overview

The code demonstrates two popular pathfinding algorithms:

- **Q-Learning**: A reinforcement learning algorithm that learns the optimal path by receiving rewards based on actions and exploring various paths.
- **A\***: A well-known heuristic-based search algorithm that finds the shortest path by evaluating nodes based on their current cost and a heuristic estimate of the remaining cost.

Both algorithms are implemented and compared using the same graph structure, and their results are visualized for comparison.

## Requirements

To run this project, you'll need to have the following Python libraries installed:

- `numpy`
- `heapq` (for priority queue operations)
- `networkx` (for graph creation and visualization)
- `matplotlib` (for plotting the graph and paths)
- `random` (for random action selection in Q-Learning)

Install the required dependencies using `pip`:

```bash
pip install numpy networkx matplotlib
```

## Components

### 1. **QLearningAgent Class**

The `QLearningAgent` class implements the Q-Learning algorithm for pathfinding. The key parameters for the agent are:

- **alpha (learning rate)**: Controls how quickly the agent updates its knowledge. A value between 0 and 1.
- **gamma (discount factor)**: Determines the importance of future rewards. A value of 0 means the agent focuses on immediate rewards, while a value of 1 prioritizes long-term rewards.
- **epsilon (exploration probability)**: Probability that the agent will explore random actions instead of exploiting the best-known action.
- **episodes**: The number of times the agent will explore and learn from the graph.

The agent uses a Q-table to store the learned values for each node pair (state-action), updating them over episodes using the **Q-learning formula**.

### 2. **A* Algorithm**

The `a_star` function implements the A\* search algorithm, which finds the shortest path between two nodes. It uses:

- **g_score**: The cost to reach a node from the start node.
- **f_score**: The total estimated cost (g_score + heuristic) to reach the goal node.
- **came_from**: A dictionary that tracks the path leading to the goal node.

The heuristic function uses **Dijkstra's algorithm** to estimate the cost from a node to the goal node.

### 3. **Graph Visualization**

The `visualize` function creates a visual comparison of the paths found by the Q-Learning and A\* algorithms using **NetworkX** and **Matplotlib**.

- Nodes are drawn as circles, and edges are drawn with weights.
- The path found by Q-Learning is shown in **green**, and the path found by A\* is shown in **red**.

### 4. **Heuristic Function**

The `heuristic` function computes an estimate of the shortest path using Dijkstra's algorithm. It is used by the A\* algorithm to evaluate node states.

### 5. **Dijkstra's Algorithm**

The `dijkstra` function implements Dijkstra’s shortest path algorithm, which is used by the A\* heuristic to calculate the shortest distance between two nodes in the graph.

---

## Graph Representation

The graph is represented as a dictionary of nodes. Each node points to a dictionary of its neighbors with edge weights.

```python
graph = {
    0: {1: 2, 2: 4},
    1: {0: 2, 3: 1},
    2: {0: 4, 3: 3, 4: 2},
    3: {1: 1, 2: 3, 5: 5},
    4: {2: 2, 5: 1},
    5: {3: 5, 4: 1}
}
```

In this graph:
- Node 0 is connected to nodes 1 and 2 with weights 2 and 4, respectively.
- Node 1 is connected to nodes 0 and 3 with weights 2 and 1, respectively, and so on.

### Start and End Nodes

The agent starts at `start_node = 0` and tries to reach `end_node = 5`.

## How to Run

1. Define your graph in the `graph` variable. It should be a dictionary of nodes where each node has a dictionary of neighboring nodes with edge weights.
2. Set your `start_node` and `end_node`.
3. Run the script. The Q-Learning agent will train and find a path using the Q-learning algorithm, while the A\* algorithm will calculate the shortest path using its heuristic approach.
4. The paths found by both algorithms will be printed, and a visual comparison of the two paths will be displayed.

```python
start_node = 0
end_node = 5

# Create and train the Q-Learning agent
q_agent = QLearningAgent(graph)
q_agent.train(start_node, end_node)
q_path = q_agent.get_best_path(start_node, end_node)

# Find the shortest path using the A* algorithm
a_path = a_star(graph, start_node, end_node)

# Print paths and their lengths and costs
print("Q-Learning Path:", q_path)
print("A* Path:", a_path)
print("Q-Learning Path Length:", len(q_path))
print("A* Path Length:", len(a_path))
print("Q-Learning Path Cost:", sum(graph[q_path[i]][q_path[i+1]] for i in range(len(q_path)-1)))
print("A* Path Cost:", sum(graph[a_path[i]][a_path[i+1]] for i in range(len(a_path)-1)))

# Visualize the paths
visualize(graph, q_path, a_path)
```

## Output

- **Q-Learning Path**: The path discovered by the Q-Learning agent.
- **A* Path**: The path found by the A\* algorithm.
- **Q-Learning Path Cost**: The total cost of the path discovered by Q-Learning.
- **A* Path Cost**: The total cost of the path discovered by A\*.

A graphical comparison will show the **Q-Learning path** in **green** and the **A\* path** in **red**.

## Notes

- The Q-Learning agent will improve its pathfinding over multiple episodes of training. The more episodes, the better it will get at finding the shortest path.
- The A\* algorithm uses a heuristic (Dijkstra's algorithm) to guide its search for the shortest path.
- The visualization helps compare the paths and the effectiveness of the two algorithms.
