#In[0]
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

#In[1]

def spectral_cluster(G, k):
    e, v = graph_eigs(G)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(v[:, -k:])
    return kmeans.labels_

def graph_eigs(G):
    A = nx.to_numpy_array(G)
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigenValues, eigenVectors = np.linalg.eig(L)
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    return eigenValues, eigenVectors

#Show that spectral clustering works
test = nx.karate_club_graph()
clusters = spectral_cluster(test, 4)
nx.draw_spring(test, with_labels=True, node_color=clusters)
plt.show()

#In[2]
def compose_n(graphlist, depth=0):
    firstGraph = graphlist[0]
    #Relabel so union doesn't destroy degenerate nodes
    nodes = firstGraph.nodes()
    mapping = {index: index + len(nodes) * depth for index in nodes}
    firstGraph = nx.relabel_nodes(firstGraph, mapping)
    if len(graphlist) == 1:
        return firstGraph
    return nx.compose(firstGraph, compose_n(graphlist[1:], depth+1))

def n_karates(n):
    return [nx.karate_club_graph() for _ in range(n)]

#In[2]
#First experiment: spectral clustering with k=3 on two connected components
#We concatenate 2 karate club graphs together
plt.close()
n_graphs = 2
k = 3
G = compose_n(n_karates(n_graphs))

nx.draw_spring(G, with_labels=True)
plt.show()

#In[3]
plt.close()
clusters = spectral_cluster(G, k)
nx.draw_spring(G, with_labels=True, node_color=clusters)
plt.show()

#In[4]
#k=2 on 3
plt.close()
n_graphs = 3
k = 2
G = compose_n(n_karates(n_graphs))
clusters = spectral_cluster(G, k)
nx.draw_spring(G, with_labels=True, node_color=clusters)
plt.show()

#In[5]
#k=2 on 5
plt.close()
n_graphs = 5
k = 2
G = compose_n(n_karates(n_graphs))
clusters = spectral_cluster(G, k)
nx.draw_spring(G, with_labels=True, node_color=clusters)
plt.show()

#In[6]
#visualize eigenvector components for n_graphs = 3, k = 2
n_graphs = 3
k = 2
G = compose_n(n_karates(n_graphs))
e, v = graph_eigs(G)
v = v[:, -k:]
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Values of each point in the first 2 eigenvectors; 3 connected components, 2 clusters')
ax1.bar(list(range(len(v[:, 0]))), v[:, 0])
ax2.bar(list(range(len(v[:, 1]))), v[:, 1])


#In[7]
#visualize 2D eigen embeddings for n_graphs = 3, k = 2
plt.close()
plt.scatter(v[:, 0], v[:, 1])


#In[8]
plt.close()
kmeans = KMeans(n_clusters=k+1, random_state=0).fit(v)
clusters = kmeans.labels_
nx.draw_spring(G, with_labels=True, node_color=clusters)
plt.show()