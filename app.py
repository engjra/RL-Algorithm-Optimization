import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from rl_algorithm_optimization import QLearningAgent, a_star

# Visualization function
def visualize(graph_dict, q_path, a_path):
    G = nx.Graph()
    for node in graph_dict:
        for neighbor, weight in graph_dict[node].items():
            G.add_edge(node, neighbor, weight=weight)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue")
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    if q_path:
        q_edges = [(q_path[i], q_path[i+1]) for i in range(len(q_path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=q_edges, edge_color='green', width=3, label='Q-Learning')

    if a_path:
        a_edges = [(a_path[i], a_path[i+1]) for i in range(len(a_path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=a_edges, edge_color='red', width=3, style='dashed', label='A*')

    plt.legend()
    plt.title("Path Comparison: Q-Learning vs A*")
    st.pyplot(plt)


# Streamlit UI
def main():
    st.set_page_config(page_title="RL Pathfinding: Q-Learning vs A*", layout="centered")
    st.title("🔍 Reinforcement Learning: Q-Learning vs A* Pathfinding")

    with st.sidebar:
        st.header("📌 Graph Configuration")
        nodes_input = st.text_input("Nodes (comma-separated)", "0,1,2,3,4,5")
        edges_input = st.text_area("Edges (format: node1 node2 weight)", 
                                   "0 1 2\n1 2 3\n2 3 1\n3 4 2\n4 5 1\n0 5 10")

    try:
        nodes = list(map(int, nodes_input.strip().split(',')))
        edges = edges_input.strip().split('\n')
        graph = {node: {} for node in nodes}

        for edge in edges:
            n1, n2, w = map(int, edge.strip().split())
            graph[n1][n2] = w
            graph[n2][n1] = w  # undirected

        st.success("✅ Graph created successfully!")
    except Exception as e:
        st.error(f"❌ Error parsing input: {e}")
        return

    start_node = st.selectbox("Start Node", options=nodes, index=0)
    end_node = st.selectbox("End Node", options=nodes, index=len(nodes)-1)

    if st.button("🚀 Run Algorithms"):
        # Q-Learning
        q_agent = QLearningAgent(graph)
        q_agent.train(start_node, end_node)
        q_path = q_agent.get_best_path(start_node, end_node)

        # A* Search
        a_path = a_star(graph, start_node, end_node)

        # Results
        st.subheader("🧠 Q-Learning Results")
        st.write("Path:", q_path)
        st.write("Path (Readable):", " → ".join(map(str, q_path)))
        st.write("Cost:", sum(graph[q_path[i]][q_path[i+1]] for i in range(len(q_path)-1)))
        st.write("Q-Table:")
        st.json(q_agent.q_table)

        st.subheader("🔺 A* Search Results")
        st.write("Path:", a_path)
        st.write("Path (Readable):", " → ".join(map(str, a_path)))
        st.write("Cost:", sum(graph[a_path[i]][a_path[i+1]] for i in range(len(a_path)-1)))

        st.subheader("📊 Visualization")
        visualize(graph, q_path, a_path)

    st.markdown("---")
    st.caption("Developed for project: Reinforcement Learning for Algorithm Optimization – Q-Learning vs A*")

if __name__ == "__main__":
    main()
