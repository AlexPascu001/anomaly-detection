import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

# generate a random graph with 100 nodes and degree 3
G = nx.random_regular_graph(3, 100)

# generate graph with 10 cliques of 20 nodes each
H = nx.connected_caveman_graph(10, 20)

# union of G and H
I = nx.union(G, H, rename=('G-', 'H-'))

# add 10 random edges to I
for _ in range(10):
    u, v = np.random.choice(I.nodes(), 2, replace=False)  # Select two distinct nodes
    I.add_edge(u, v)

# draw graph I
plt.figure()
nx.draw(I)
plt.show()

for node in I.nodes():
    ego = nx.ego_graph(I, node)
    num_neighbors = len(list(ego.neighbors(node)))
    num_edges = ego.number_of_edges()

    I.nodes[node]['num_neighbors'] = num_neighbors
    I.nodes[node]['num_edges'] = num_edges

neighbors = nx.get_node_attributes(I, 'num_neighbors')
edges = nx.get_node_attributes(I, 'num_edges')

n = np.array(list(neighbors.values())).reshape(-1, 1)
e = np.array(list(edges.values())).reshape(-1, 1)

model = np.load('power_law_model.npy', allow_pickle=True).item()
theta = model['theta']
C = model['C']

predicted_e = C * (n ** theta)

anomaly_scores = []
for xi, yi, predicted_yi in zip(n, e, predicted_e):
    score = (max(yi, predicted_yi) / min(yi, predicted_yi)) * np.log(abs(yi - predicted_yi) + 1)
    anomaly_scores.append(score)

for i, node in enumerate(I.nodes()):
    I.nodes[node]['anomaly_score'] = anomaly_scores[i]

sorted_nodes = sorted(I.nodes(data=True), key=lambda x: x[1]['anomaly_score'], reverse=True)
top_10_nodes = [node for node, data in sorted_nodes[:10]]

plt.figure(figsize=(10, 10))
pos = nx.spring_layout(I)

nx.draw(I, pos, node_color='green', node_size=20, edge_color='gray')

nx.draw(I.subgraph(top_10_nodes), pos, node_color='red', node_size=50)

plt.title("Graph with Top 10 Clique-Like Nodes Highlighted")
plt.show()

G1 = nx.random_regular_graph(3, 100)
G2 = nx.random_regular_graph(5, 100)

G = nx.union(G1, G2, rename=('G1-', 'G2-'))

for u, v in G.edges():
    G.add_edge(u, v, weight=1)

random_nodes = np.random.choice(G.nodes(), 2, replace=False)
for node in random_nodes:
    for u, v in nx.ego_graph(G, node).edges():
        G[u][v]['weight'] += 10

for node in G.nodes():
    ego = nx.ego_graph(G, node)
    total_weight = sum(G[u][v]['weight'] for u, v in ego.edges())
    num_edges = ego.number_of_edges()

    G.nodes[node]['total_weight'] = total_weight
    G.nodes[node]['num_edges'] = num_edges

weights = nx.get_node_attributes(G, 'total_weight')
edges = nx.get_node_attributes(G, 'num_edges')

w = np.array(list(weights.values())).reshape(-1, 1)  # Wi
e = np.array(list(edges.values())).reshape(-1, 1)  # Ei

model = np.load('power_law_model.npy', allow_pickle=True).item()
theta = model['theta']
C = model['C']

predicted_w = C * (e ** theta)

anomaly_scores = []
for yi, predicted_yi in zip(w, predicted_w):
    score = (max(yi, predicted_yi) / min(yi, predicted_yi)) * np.log(abs(yi - predicted_yi) + 1)
    anomaly_scores.append(score)

for i, node in enumerate(G.nodes()):
    G.nodes[node]['anomaly_score'] = anomaly_scores[i]

sorted_nodes = sorted(G.nodes(data=True), key=lambda x: x[1]['anomaly_score'], reverse=True)
top_4_nodes = [node for node, data in sorted_nodes[:4]]

plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G)

nx.draw(G, pos, node_color='green', node_size=20, edge_color='gray')

nx.draw(G.subgraph(top_4_nodes), pos, node_color='red', node_size=50)
edge_labels = nx.get_edge_attributes(G, 'weight')

nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)


plt.title("Graph with Top 4 HeavyVicinity Nodes Highlighted")
plt.show()