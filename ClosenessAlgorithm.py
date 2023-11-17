import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np


def haversine(lon1, lat1, lon2, lat2):
    
    
    lon1 = np.radians(float(lon1))
    lat1 = np.radians(float(lat1))
    lon2 = np.radians(float(lon2))
    lat2 = np.radians(float(lat2))

    r = 6371
    
    
    dlon = np.subtract(lon2, lon1)
    dlat = np.subtract(lat2, lat1)

    a = np.add(np.power(np.sin(np.divide(dlat, 2)), 2),
               np.multiply(np.cos(lat1),
                           np.multiply(np.cos(lat2),
                                       np.power(np.sin(np.divide(dlon, 2)), 2))
                           )
              )
    c = np.multiply(2, np.arcsin(np.sqrt(a)))

    return c*r

G = nx.Graph()

nodes = [
    (-12.088458678552852, -76.99995413516139),
    (-12.090380562754838, -76.99966032710414),
    (-12.092966168683475, -76.99931586248526),
    (-12.088378443303547, -76.99937003526539),
    (-12.08826178845815, -76.99876807584465),
    (-12.08822301245964, -76.99832195365843),
    (-12.088148691780125, -76.99778660703494),
    (-12.088056530949354, -76.99720403754591),
    (-12.087968546211531, -76.99656132430854),
    (-12.089675534451413, -76.9991519521514),
    (-12.089551777094766, -76.99857450208236),
    (-12.089497633233227, -76.99811570613711),
    (-12.089458959039707, -76.99760153826745),
    (-12.090308175716393, -76.99907571953004),
    (-12.090245174257388, -76.99849417959894),
    (-12.090082510227436, -76.99750638496573),
    (-12.089992243978523, -76.99690634527937),
    (-12.089901977699176, -76.99634323111216),
    (-12.091471281814615, -76.99893086822658),
    (-12.091381614578621, -76.99832334463916),
    (-12.09212136837659, -76.99887355468059),
    (-12.091673032985709, -76.99729170081149),
    (-12.092714502764341, -76.99873886670235),
    (-12.09226863928874, -76.99720467799904),
    (-12.092197551760778, -76.99660251434683),
    (-12.092167457482644, -76.99596805144829)
]

G.add_nodes_from(nodes)

# Agregar aristas
edges = [
    ((-12.088458678552852, -76.99995413516139), (-12.088378443303547, -76.99937003526539)),
((-12.088458678552852, -76.99995413516139), (-12.090380562754838, -76.99966032710414)),
((-12.088378443303547, -76.99937003526539), (-12.08826178845815, -76.99876807584465)),
((-12.088378443303547, -76.99937003526539), (-12.089675534451413, -76.9991519521514)),
((-12.08826178845815, -76.99876807584465), (-12.08822301245964, -76.99832195365843)),
((-12.08826178845815, -76.99876807584465), (-12.089551777094766, -76.99857450208236)),
((-12.08822301245964, -76.99832195365843), (-12.088148691780125, -76.99778660703494)),
((-12.08822301245964, -76.99832195365843), (-12.089497633233227, -76.99811570613711)),
((-12.088148691780125, -76.99778660703494), (-12.089458959039707, -76.99760153826745)),
((-12.088148691780125, -76.99778660703494), (-12.088056530949354, -76.99720403754591)),
((-12.088056530949354, -76.99720403754591), (-12.087968546211531, -76.99656132430854)),

((-12.089675534451413, -76.9991519521514), (-12.089551777094766, -76.99857450208236)),
((-12.089551777094766, -76.99857450208236), (-12.089497633233227, -76.99811570613711)),
((-12.089497633233227, -76.99811570613711), (-12.089458959039707, -76.99760153826745)),

((-12.090308175716393, -76.99907571953004), (-12.090380562754838, -76.99966032710414)),
((-12.090308175716393, -76.99907571953004), (-12.090245174257388, -76.99849417959894)),
((-12.090245174257388, -76.99849417959894), (-12.090082510227436, -76.99750638496573)),
((-12.090082510227436, -76.99750638496573), (-12.089992243978523, -76.99690634527937)),
((-12.089992243978523, -76.99690634527937), (-12.089901977699176, -76.99634323111216)),
((-12.090308175716393, -76.99907571953004), (-12.089675534451413, -76.9991519521514)),
((-12.090082510227436, -76.99750638496573), (-12.089458959039707, -76.99760153826745)),
((-12.089992243978523, -76.99690634527937), (-12.088056530949354, -76.99720403754591)),
((-12.089901977699176, -76.99634323111216), (-12.087968546211531, -76.99656132430854)),

((-12.091471281814615, -76.99893086822658), (-12.091381614578621, -76.99832334463916)),
((-12.091381614578621, -76.99832334463916), (-12.090245174257388, -76.99849417959894)),
((-12.091471281814615, -76.99893086822658), (-12.090308175716393, -76.99907571953004)),

((-12.09212136837659, -76.99887355468059), (-12.091673032985709, -76.99729170081149)),
((-12.091673032985709, -76.99729170081149), (-12.090082510227436, -76.99750638496573)),
((-12.09212136837659, -76.99887355468059), (-12.091471281814615, -76.99893086822658)),

((-12.092714502764341, -76.99873886670235), (-12.092966168683475, -76.99931586248526)),
((-12.092714502764341, -76.99873886670235), (-12.09212136837659, -76.99887355468059)),
((-12.09226863928874, -76.99720467799904), (-12.091673032985709, -76.99729170081149)),
((-12.09226863928874, -76.99720467799904), (-12.092197551760778, -76.99660251434683)),
((-12.092197551760778, -76.99660251434683), (-12.089992243978523, -76.99690634527937)),
((-12.092197551760778, -76.99660251434683), (-12.092167457482644, -76.99596805144829)),
((-12.092167457482644, -76.99596805144829), (-12.089901977699176, -76.99634323111216)),
((-12.092966168683475, -76.99931586248526), (-12.090380562754838, -76.99966032710414)),
((-12.092714502764341, -76.99873886670235
), (-12.09226863928874, -76.99720467799904))
]

G.add_edges_from(edges)

for edge in edges:
    node1, node2 = edge
    lat1, lon1 = node1
    lat2, lon2 = node2
    weight = haversine(lat1, lon1, lat2, lon2)
    G.add_edge(node1, node2, weight=weight)

# Dibujar el grafo
pos = {node: node for node in G.nodes()}

nx.draw(G, pos, with_labels=False, font_weight='bold', node_size=700, node_color='skyblue', font_size=8, font_color='black', font_family='Arial')
edge_labels = {(n1, n2): f"{weight:.3f}" for (n1, n2, weight) in G.edges(data='weight')}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)


# Función para calcular la distancia ponderada entre dos nodos utilizando Dijkstra
# En esta función se calcula en la variable distances las distancias menores de un nodo a todos los demas por ejemplo:
# {'A': 0, 'B': 3, 'C': 2, 'D': 6, 'E': 3, 'J': 6}
# De A a B el camino más corto es 3. De A a D el camino más 6.
def camino_mas_corto(graph, source, weight='weight'):

    distances = {node: float('inf') for node in graph.nodes}
    distances[source] = 0
    visited = set()


    while visited != set(graph.nodes): #Basicamente mientras no se hayan visitado todos los nodos
        current_node = None
        #Se verifica cual e es el más corto y se pone como current node
        for node in graph.nodes:
            if node not in visited and (current_node is None or distances[node] < distances[current_node]):
                current_node = node

        visited.add(current_node)
        neighbors = set(graph[current_node]) - visited
        for neighbor in neighbors:
            potential = distances[current_node] + graph[current_node][neighbor].get(weight, 1)
            if potential < distances[neighbor]:
                distances[neighbor] = potential

    return distances

# Calcula la centralidad de cercanía
closeness_centrality = {}

targets={'A': (-12.09212136837659, -76.99887355468059), 
         'B': (-12.092966168683475, -76.99931586248526), 
         'C': (-12.089992243978523, -76.99690634527937)}

for node in targets:
    total_distance = 0
    shortest_path_length = camino_mas_corto(G, targets[node], weight='weight')
    for _node in G.nodes:
        total_distance+=shortest_path_length[_node]
    closeness_centrality[node] = 1 / total_distance

# Imprime los valores de centralidad de cercanía
for node, closeness in closeness_centrality.items():
    print(f'Nodo: {node}, Centralidad de Cercanía: {closeness}')

closest_node = None
for node in targets:
    if closest_node is None or closeness_centrality[closest_node] < closeness_centrality[node]:
        closest_node = node

print(f"El nodos más cercano a los demas es: {closest_node}")
for label, node in targets.items():
    nx.draw_networkx_labels(G, pos, labels={node: label})
plt.show()