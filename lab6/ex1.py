import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

import matplotlib
matplotlib.use('TkAgg')

with open('ca-AstroPh.txt', 'r') as f:
    lines = f.readlines()

for line in lines:
    if line[0] == '#':
        lines.remove(line)

G = nx.Graph()

for line in lines[:1500]:
    edge = line.split()
    G.add_edge(edge[0], edge[1])
    if 'weight' not in G[edge[0]][edge[1]]:
        G[edge[0]][edge[1]]['weight'] = 1
    else:
        G[edge[0]][edge[1]]['weight'] += 1

print('Number of nodes:', G.number_of_nodes())
print('Number of edges:', G.number_of_edges())

for node in G.nodes():
    # get ego network of node
    ego = nx.ego_graph(G, node)
    # get number of neighbors of node
    num_neighbors = len(list(ego.neighbors(node)))
    # get number of edges in ego network
    num_edges = ego.number_of_edges()
    # get total weight of edges in ego network
    total_weight = sum([ego[node][neighbor]['weight'] for neighbor in ego.neighbors(node)])
    # get principal eigenvalue of weighted adjacency matrix of ego network
    principal_eigenvalue = max(np.linalg.eigvals(nx.adjacency_matrix(ego).todense()))

    nx.set_node_attributes(G, {node: {'num_neighbors': num_neighbors, 'num_edges': num_edges, 'total_weight': total_weight, 'principal_eigenvalue': principal_eigenvalue}})

neighbors = nx.get_node_attributes(G, 'num_neighbors')
edges = nx.get_node_attributes(G, 'num_edges')
weights = nx.get_node_attributes(G, 'total_weight')
eigenvalues = nx.get_node_attributes(G, 'principal_eigenvalue')

n = np.array(list(neighbors.values())).reshape(-1, 1)
e = np.array(list(edges.values())).reshape(-1, 1)

log_n = np.log(n + 1e-10)
log_e = np.log(e + 1e-10)

model = LinearRegression()
model.fit(log_n, log_e)

theta = model.coef_[0]
C = np.exp(model.intercept_)
print('Power law parameters: theta =', theta, 'C =', C)

# save the model
np.save('power_law_model.npy', {'theta': theta, 'C': C})

anomaly_scores = []
for xi, yi in zip(n, e):
    predicted_yi = C * (xi ** theta)
    score = (max(yi, predicted_yi) / min(yi, predicted_yi)) * np.log(abs(yi - predicted_yi) + 1)
    anomaly_scores.append(score)

for i, node in enumerate(G.nodes()):
    G.nodes[node]['anomaly_score'] = anomaly_scores[i]

# sort nodes by anomaly score in descending order
sorted_nodes = sorted(G.nodes(data=True), key=lambda x: x[1]['anomaly_score'], reverse=True)

# draw graph, draw top 10 nodes with highest anomaly scores in red
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G)
nx.draw(G, pos, node_color='green', node_size=20, edge_color='gray')
nx.draw(G.subgraph([node for node, data in sorted_nodes[:10]]), pos, node_color='red', node_size=50)
plt.title("Graph with Top 10 Nodes Highlighted Based on Anomaly Scores")
# plt.savefig('anomaly_scores.png')
plt.show()

features = np.hstack((n, e))  # Combine the features N_i and E_i
lof = LocalOutlierFactor(n_neighbors=20)
lof_scores = -lof.fit_predict(features)  # LOF gives -1 for outliers, use the negative scores

scaler = MinMaxScaler()
normalized_anomaly_scores = scaler.fit_transform(np.array(anomaly_scores).reshape(-1, 1)).flatten()

final_scores = normalized_anomaly_scores + lof_scores

for i, node in enumerate(G.nodes()):
    G.nodes[node]['final_score'] = final_scores[i]

sorted_final_nodes = sorted(G.nodes(data=True), key=lambda x: x[1]['final_score'], reverse=True)

plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G)

nx.draw(G, pos, node_color='green', node_size=20, edge_color='gray')

top_10_nodes = [node for node, data in sorted_final_nodes[:10]]
nx.draw(G.subgraph(top_10_nodes), pos, node_color='red', node_size=50)

plt.title("Graph with Top 10 Nodes Highlighted Based on Final Scores (Anomaly Scores + LOF Scores)")
# plt.savefig('final_scores.png')
plt.show()