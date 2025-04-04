import numpy as np
import heapq
import networkx as nx
import matplotlib.pyplot as plt
import random

""" 
alpha: Learning Rate - Determines how quickly the agent updates its knowledge. 
       A low value makes the agent learn slowly, while a high value allows faster learning.
gamma: Discount Factor - Determines the importance of future rewards. 
       A value of 0 means focusing only on immediate rewards, while a value of 1 focuses on long-term rewards.
epsilon: Exploration Probability - This value controls how often the agent explores random actions instead of exploiting known ones.
         Ensures the agent explores new options and avoids getting stuck in local optimal solutions.
episodes: Number of Episodes - The number of times the agent will train (learn) on the graph to improve its knowledge.
Q-value table: This stores the learned values for each possible action (path) for each state (node) in the graph.
"""


class QLearningAgent:
    def __init__(self, graph, alpha=0.9, gamma=0.5, epsilon=0.1, episodes=100):
        self.graph = graph  # The graph representing the environment
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration probability
        self.episodes = episodes  # Number of episodes (trials) to learn
        # Initialize the Q-table with zero values for all nodes and their neighbors
        self.q_table = {
            node: {neighbor: 0 for neighbor in graph[node]} for node in graph}

    def choose_action(self, state):
        """Choose an action based on exploration vs exploitation."""
        if random.uniform(0, 1) < self.epsilon:  # Exploration: select a random action
            return random.choice(list(self.q_table[state].keys()))
        # Exploitation: select the action with the highest Q-value (best learned action)
        else:
            return max(self.q_table[state], key=self.q_table[state].get)

    def train(self, start_node, end_node):
        """Train the agent to learn the optimal path."""
        for _ in range(self.episodes):  # Train for a set number of episodes
            state = start_node  # Initialize the state (starting node)

            # Keep going until the goal (end node) is reached
            while state != end_node:
                action = self.choose_action(state)  # Select an action
                # The action taken leads to the next state (neighbor)
                next_state = action

                # Define the reward: a positive reward if we reach the end node, else a negative reward based on the edge weight
                reward = 100 if next_state == end_node else - \
                    self.graph[state][next_state]

                # Retrieve the current Q-value for this state-action pair
                old_value = self.q_table[state][action]
                next_max = max(self.q_table[next_state].values(
                )) if next_state in self.q_table else 0  # Max Q-value of the next state

                # Q-learning update rule: Update the Q-value for the state-action pair
                new_value = (1 - self.alpha) * old_value + \
                    self.alpha * (reward + self.gamma * next_max)
                # Update the Q-table with the new Q-value
                self.q_table[state][action] = new_value

                state = next_state  # Transition to the next state

    def get_best_path(self, start_node, end_node):
        """Retrieve the best path from start to end using the Q-table."""
        path = [start_node]
        state = start_node

        while state != end_node:  # Follow the actions with the highest Q-value to reach the end node
            if state not in self.q_table or not self.q_table[state]:
                break  # If there's no valid action, exit the loop

            # Always pick the action with the highest Q-value
            state = max(self.q_table[state], key=self.q_table[state].get)
            path.append(state)

        return path


def a_star(graph, start_node, end_node):
    """
    Implements the A* search algorithm to find the shortest path in a weighted graph.
    This method calculates the best path using both cost and heuristic values.
    """
    open_set = {start_node}  # Initialize open set with the start node
    came_from = {}  # Keeps track of the path (node -> previous node)
    # Initialize all nodes to infinity (unreachable)
    g_score = {node: float('inf') for node in graph}
    g_score[start_node] = 0  # Cost to reach the start node is 0

    # Initialize heuristic scores to infinity
    f_score = {node: float('inf') for node in graph}
    # Use heuristic to estimate total cost from start to end
    f_score[start_node] = heuristic(start_node, end_node, graph)

    while open_set:  # Continue until the open set is empty
        # Get the node with the lowest f_score
        current = min(open_set, key=lambda node: f_score[node])

        if current == end_node:  # If we reached the end node, reconstruct and return the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_node)  # Add the start node at the end
            return path[::-1]  # Reverse the path to get it in correct order

        open_set.remove(current)  # Remove the current node from open set

        # Explore neighbors of the current node
        for neighbor, weight in graph[current].items():
            # Calculate the cost to reach the neighbor
            tentative_g_score = g_score[current] + weight

            # If this new path to the neighbor is better (shorter), update its scores
            if tentative_g_score < g_score[neighbor]:
                # Record how we reached the neighbor
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score  # Update the cost
                # Update estimated total cost
                f_score[neighbor] = g_score[neighbor] + \
                    heuristic(neighbor, end_node, graph)
                # Add the neighbor to open set for further evaluation
                open_set.add(neighbor)

    return []  # Return an empty list if no path is found


def dijkstra(graph, start_node, end_node):
    """
    Dijkstra's algorithm to find the shortest path between nodes in a graph.
    This method does not use heuristics but relies solely on path costs.
    """
    distances = {node: float('inf')
                 for node in graph}  # Set initial distances to infinity
    distances[start_node] = 0  # Distance to the start node is 0
    # Initialize the priority queue with the start node
    priority_queue = [(0, start_node)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(
            priority_queue)  # Get the node with the smallest distance

        if current_node == end_node:  # If we reached the end node, return the distance
            return current_distance

        # Explore the neighbors of the current node
        for neighbor, weight in graph[current_node].items():
            # Calculate the tentative distance to the neighbor
            distance = current_distance + weight
            # If this path is shorter, update the distance
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                # Add the neighbor to the priority queue
                heapq.heappush(priority_queue, (distance, neighbor))

    return float('inf')  # Return infinity if no path is found


def heuristic(start_node, end_node, graph):
    """Heuristic function used by A* to estimate the cost from the start node to the end node."""
    return dijkstra(graph, start_node, end_node)  # Use Dijkstra as the heuristic (shortest path cost)


def visualize(graph_dict, q_path, a_path):
    """
    Visualizes the graph and the paths found by Q-Learning and A* algorithms.
    Uses NetworkX and Matplotlib to draw the graph and paths.
    """
    G = nx.Graph()  # Create a NetworkX graph
    for node in graph_dict:
        for neighbor, weight in graph_dict[node].items():
            G.add_edge(node, neighbor, weight=weight)  # Add edges to the graph

    pos = nx.spring_layout(G)  # Layout the graph visually
    plt.figure(figsize=(10, 6))  # Set the figure size

    # Draw nodes and labels
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue")

    # Draw edge labels (weights)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Draw paths for Q-Learning (green) and A* (red)
    q_edges = [(q_path[i], q_path[i+1]) for i in range(len(q_path)-1)]
    a_edges = [(a_path[i], a_path[i+1]) for i in range(len(a_path)-1)]

    nx.draw_networkx_edges(G, pos, edgelist=q_edges,
                           edge_color='green', width=2, label="Q-Learning Path")
    nx.draw_networkx_edges(G, pos, edgelist=a_edges,
                           edge_color='red', width=2, label="A* Path")

    plt.legend(loc="upper left")  # Display the legend
    # Title for the plot
    plt.title("Comparison of Q-Learning and A* Algorithm Paths with Weights")
    plt.show()  # Show the plot


if __name__ == "__main__":
    # Define the graph as an adjacency list with weights
    graph = {
        0: {1: 2, 2: 4},
        1: {0: 2, 3: 1},
        2: {0: 4, 3: 3, 4: 2},
        3: {1: 1, 2: 3, 5: 5},
        4: {2: 2, 5: 1},
        5: {3: 5, 4: 1}
    }

    start_node = 0
    end_node = 5

    # Q-Learning Algorithm
    q_agent = QLearningAgent(graph)
    q_agent.train(start_node, end_node)
    q_path = q_agent.get_best_path(start_node, end_node)

    # A* Algorithm
    a_path = a_star(graph, start_node, end_node)

    # Output results
    print("Q-Learning Path:", q_path)
    print("A* Path:", a_path)
    print("Q-Learning Path Length:", len(q_path))
    print("A* Path Length:", len(a_path))
    print("Q-Learning Path Cost:",
          sum(graph[q_path[i]][q_path[i+1]] for i in range(len(q_path)-1)))
    print("A* Path Cost:", sum(graph[a_path[i]]
          [a_path[i+1]] for i in range(len(a_path)-1)))
    print("Q-Learning Q-Table:", q_agent.q_table)

    # Visualize the comparison between Q-Learning and A* paths
    visualize(graph, q_path, a_path)
