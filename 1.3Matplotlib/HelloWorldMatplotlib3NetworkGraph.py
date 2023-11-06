# https://plotly.com/python/network-graphs/
import matplotlib.pyplot as plt
import networkx as nx

G = nx.Graph()
G.add_node(1)
G.add_nodes_from([2, 3])
G.add_nodes_from([
    (4, {"color": "red"}),
    (5, {"color": "green"}),
    ])
G.add_edge(1, 2)
e = (2, 3)
G.add_edge(*e)
G.add_edges_from([(1, 2), (1, 3)])

ax = plt.subplot(111)
nx.draw_networkx(G, with_labels=True)
plt.show()
plt.close()

G.clear()
G.add_edges_from([(1, 2), (1, 3)])
G.add_node(1)
G.add_edge(1, 2)
G.add_node("spam")        # adds node "spam"
G.add_nodes_from("spam")  # adds 4 nodes: 's', 'p', 'a', 'm'
G.add_edge(3, 'm')
ax = plt.subplot(111)
nx.draw_networkx(G, with_labels=True)
plt.show()
plt.close()

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

