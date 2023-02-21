import torch
from datasets.dataset_karateclub import KarateDataset
from models.GCN_basic import Net
from models.GraphSAGE_basic import GraphSAGE
import torch.nn.functional as F

import os
import argparse

class Main():
    def __init__(self, train_config, debug=False):
        self.train_config = train_config

        torch.manual_seed(42)
        self.optimizer_name = "Adam"
        self.lr = 1e-1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dataset = KarateDataset()
        self.data = self.dataset[0]

        self.epochs = self.train_config['epochs']
        self.model_name = self.train_config['model']
        #TODO mirar lo de usar el getattr
        #self.model = getattr(torch.optim, self.optimizer_name)
        if self.model_name == 'gcn':
            self.model = Net(self.data).to(self.device)
        elif self.model_name == 'graphsage':
            self.model = GraphSAGE(self.data).to(self.device)

        self.optimizer = getattr(torch.optim, self.optimizer_name)(self.model.parameters(), lr=self.lr)

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        F.nll_loss(self.model()[self.data.train_mask], self.data.y[self.data.train_mask]).backward()
        self.optimizer.step()

    @torch.no_grad()
    def test(self):
        self.model.eval()
        logits = self.model()
        mask1 = self.data['train_mask']
        pred1 = logits[mask1].max(1)[1]
        acc1 = pred1.eq(self.data.y[mask1]).sum().item() / mask1.sum().item()
        mask = self.data['test_mask']
        pred = logits[mask].max(1)[1]
        acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
        return acc1, acc

    def run(self):
        for epoch in range(1, self.epochs):
            self.train()
        train_acc, test_acc = self.test()
        print('#' * 70)
        print('Train Accuracy: %s' % train_acc)
        print('Test Accuracy: %s' % test_acc)
        print('#' * 70)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-epochs', help='train epoch', type = int, default=200)
    parser.add_argument('-model', help='training model', type = str, default='graphsage')

    args = parser.parse_args()

    train_config = {
        'epochs': args.epochs,
        'model': args.model
    }


    main = Main(train_config, debug=False)
    main.run()