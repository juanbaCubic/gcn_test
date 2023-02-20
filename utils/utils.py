import networkx
import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def returnEmbeddings():
    # load graph from networkx library

    G = nx.karate_club_graph()

    # retrieve the labels for each node
    labels = np.asarray([G.nodes[i]['club'] != 'Mr. Hi' for i in G.nodes]).astype(np.int64)

    # create edge index from
    adj = nx.to_scipy_sparse_array(G).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)

    # using degree as embedding
    first_degree = G.degree()
    dict_first_degree = dict(first_degree)
    values_dict_first_degree = dict_first_degree.values()
    list_values_dict_first_degree = list(values_dict_first_degree)
    np_array_list_values_dict_first_degree = np.array(list_values_dict_first_degree)
    embeddings = np.array(list(dict(G.degree()).values()))

    # normalizing degree values
    scale = StandardScaler()
    embeddings_reshaped = embeddings.reshape(-1, 1)
    embeddings = scale.fit_transform(embeddings_reshaped)
    return G, labels, edge_index, embeddings