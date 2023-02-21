import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T
from utils import returnEmbeddings

# custom dataset
class KarateDataset(InMemoryDataset):

    def __init__(self, transform=None):
        super(KarateDataset, self).__init__('.', transform, None , None)

        # labels, edge_index, embeddings
        G, labels, edge_index, embeddings, adj_t = returnEmbeddings()

        # generate data for the dataset
        data = Data(edge_index=edge_index)
        data.num_nodes = G.number_of_nodes()

        # embeddings
        data.x = torch.from_numpy(embeddings).type(torch.float32)

        # labels
        y = torch.from_numpy(labels).type(torch.long)
        data.y = y.clone().detach()

        data.num_classes = 2

        data.adj_t = adj_t

        # splitting the data into train, validation and test
        X_train, X_test, y_train, y_test = train_test_split(pd.Series(list(G.nodes())),
                                                            pd.Series(labels),
                                                            test_size=0.30,
                                                            random_state=42)
        n_nodes = G.number_of_nodes()


        # create train and test masks for data
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[X_train.index] = True
        test_mask[X_test.index] = True
        data['train_mask'] = train_mask
        data['test_mask'] = test_mask

        self.data, self.slices = self.collate([data])

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)