import networkx as nx
import matplotlib.pyplot as plt

# Crea un grafo ponderado
G = nx.Graph()

# Agrega nodos
G.add_node('A')
G.add_node('B')
G.add_node('C')
G.add_node('D')
G.add_node('E')
G.add_node('J')


# Agrega aristas con pesos
G.add_edge('A', 'B', weight=1)
G.add_edge('A', 'C', weight=2)
G.add_edge('B', 'C', weight=2)
G.add_edge('C', 'D', weight=4)
G.add_edge('C', 'E', weight=1)
G.add_edge('D', 'J', weight=2)
G.add_edge('E', 'J', weight=3)

# Dibuja el grafo
pos = nx.spring_layout(G)  # Define una disposición para los nodos
labels = {edge: G.get_edge_data(edge[0], edge[1])['weight'] for edge in G.edges()}  # Etiquetas con los pesos
nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)



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

for node in G.nodes:
    total_distance = 0
    shortest_path_length = camino_mas_corto(G, node, weight='weight')
    for _node in G.nodes:
        total_distance+=shortest_path_length[_node]
    closeness_centrality[node] = 1 / total_distance

# Imprime los valores de centralidad de cercanía
for node, closeness in closeness_centrality.items():
    print(f'Nodo: {node}, Centralidad de Cercanía: {closeness}')

closest_node = None
for node in G.nodes:
    if closest_node is None or closeness_centrality[closest_node] < closeness_centrality[node]:
        closest_node = node

print(f"El nodos más cercano a los demas es: {closest_node}")