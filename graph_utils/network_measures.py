import networkx as nx
import numpy as np
import powerlaw
from joblib import Parallel, delayed

def get_centralization(G):
    centrality = nx.degree_centrality(G)
    c_denominator = float(1)
    n_val = float(len(centrality))	
    c_denominator = (n_val-1)*(n_val-2)
    
    #start calculations	
    c_node_max = max(centrality.values())
    c_sorted = sorted(centrality.values(),reverse=True)
    c_numerator = 0

    for value in c_sorted:
        c_numerator += (c_node_max*(n_val-1) - value*(n_val - 1))


    network_centrality = float(c_numerator/c_denominator)
		
    return network_centrality

def calculate_exponent(G):
    """
    Calculates the power law exponent for the degree distribution
    """
    M = nx.to_numpy_array(G)
    p = M.shape[0]
    M[np.abs(M) > 0] = 1
    degrees = M.sum(axis=0)

    #x_min = 1

    #alpha = 1 + p * np.reciprocal(np.log(degrees).sum())

    # Now calculate R^2
    #degree_log = np.log10(degrees)

    fit = powerlaw.Fit(degrees)
    
    return fit.power_law.alpha, fit.xmin, fit.power_law.D


def get_average_shortest_path_length(G):
    """
    Gets the longest shortest path in the graph
    """
    paths = [x for x in nx.shortest_path_length(G, weight='weight')]
    num_nodes = len(G)
    length = 0
    for path in paths:
        length += sum(path[1].values())

    return 1/(num_nodes*(num_nodes-1)) * length

def calculate_network_heterogeneity(G):
    val = 0
    for edge in G.edges():
        node1 = edge[0]
        node2 = edge[1]

        deg_node_1 = len(G[node1])
        deg_node_2 = len(G[node2])

        val += ((1/np.sqrt(deg_node_1)) - (1/np.sqrt(deg_node_2)))**2

    p = len(G)
    val /= (p - 2 * np.sqrt(p-1))

    return val

def calculate_average_nearest_neighbour_exponent(G):
    p = len(G)
    M = nx.to_numpy_array(G, weight=None)
    vals = list(nx.average_degree_connectivity(G, weight='weight').values())
    fit = powerlaw.Fit(vals)
    return fit.power_law.alpha

def calculate_diameter(G):
    if nx.number_connected_components(G) == 1:
        return nx.diameter(G)
    else:
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)       
        G0 = G.subgraph(Gcc[0])        
        return nx.diameter(G0)