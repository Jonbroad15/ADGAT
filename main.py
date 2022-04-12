import numpy as np
import os
import io
import pickle
import fasttext
import dgl
# from dgl.nn.pytorch import GATv2Conv
from GATv2Conv import GATv2Conv
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from graph_utils.tools import *
from sklearn.model_selection import KFold


script_dir = os.path.realpath(os.path.dirname(__file__))

# Edit these to point to the training set transcipt locations
control_transcripts_path = "data/ADReSS-IS2020-data/train/transcription/cc/"
ad_transcripts_path = "data/ADReSS-IS2020-data/train/transcription/cd/"

# Edit this to set the co-occurrence window
o = 2

# Edit these to point to the labels
meta_data = read_meta_data("data/ADReSS-IS2020-data/train/cc_meta_data.txt", dict())
meta_data = read_meta_data("data/ADReSS-IS2020-data/train/cd_meta_data.txt", meta_data)

# Edit this to point to the test set
test_set_path = "data/ADReSS-IS2020-data/test/transcription/"
meta_data_test = read_meta_data('data/ADReSS-IS2020-data/test/meta_data_test.txt', dict())

# Word embeddings
vectors_path = 'results/data/DementiaBank.bin'

onehots = {}
# Onehot cache
# with open('results/onehots.p', 'rb') as f:
    # onehots = pickle.load(f)

def encode_onehot(w):
    if w in onehots.keys():
        return torch.unsqueeze(onehots[w], dim=0)
    else:
        return None
        for word, v in onehots.items():
            onehots[word] = torch.cat( (onehots[word], torch.zeros(1)) )
        onehots[w] = torch.zeros(len(onehots))
        onehots[w] = torch.cat( (onehots[w], torch.ones(1)) )

def load_vectors(fname, test=False):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    small = 100
    for i, line in enumerate(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
        if test and i>small:
            break
    return data

class GAT(nn.Module):
    def __init__(self, layers, linears,
                num_heads=1,
                feat_drop=0,
                attn_drop=0,
                residual=False,
                concat=False,
                negative_slope = 0.2,
                share_weights=False):
        super(GAT, self).__init__()
        self.dim = layers
        self.num_heads = num_heads
        self.concat = concat
        if self.concat:
            self.GATs = nn.ModuleList([GATv2Conv(layers[i]*(num_heads**i),
                                                layers[i+1]*(num_heads**i),
                                                    num_heads,
                                                    feat_drop=feat_drop,
                                                    attn_drop=attn_drop,
                                                    residual=residual,
                                                    negative_slope = negative_slope,
                                                    share_weights=share_weights)\
                                        for i in range(len(layers)-1)])
            self.linears = nn.ModuleList([nn.Linear(self.GATs[-1].fc_dst.out_features, linears[0])])
        else:
            self.GATs = nn.ModuleList([GATv2Conv(layers[i], layers[i+1],
                                                    num_heads,
                                                    feat_drop=feat_drop,
                                                    attn_drop=attn_drop,
                                                    residual=residual,
                                                    negative_slope = negative_slope,
                                                    share_weights=share_weights)\
                                        for i in range(len(layers)-1)])
            self.linears = nn.ModuleList([nn.Linear(layers[-1], linears[0])])
        for i in range(len(linears)-1):
            self.linears.append(nn.Linear(linears[i], linears[i+1]))

    def forward(self, g, in_feat):
        h = in_feat
        n_nodes = g.num_nodes()
        for i, layer in enumerate(self.GATs):
            h = layer(g, h)
            if self.num_heads > 1:
                if self.concat:
                    h = h.reshape(n_nodes, h.shape[1]*h.shape[2])
                else:
                    h = h.mean(dim = 1)

        g.ndata['h'] = h
        h = torch.sigmoid(dgl.mean_nodes(g, 'h'))

        for layer in self.linears:
            h = layer(h)

        return torch.sigmoid(h)

    def reset_parameters(self):
        for layer in self.GATs:
            layer.reset_parameters()


class Dataset(DGLDataset):
    def __init__(self, device,
                    test=False,
                    feat='onehot'):
        self.test = test
        self.device = device
        self.feat = feat
        if feat == 'fasttext':
            self.word_vectors = fasttext.load_model(vectors_path)
        super().__init__(name='Dataset')

    def process(self):
        self.graphs = []
        self.labels = []
        if self.test:
            graphs = load_all_files(test_set_path, occurrence_window=o)
            self.labels = torch.tensor(list(meta_data_test.values()))
        else:
            control_graphs = load_all_files(control_transcripts_path, occurrence_window=o)
            ad_graphs = load_all_files(ad_transcripts_path, occurrence_window=o)
            graphs = control_graphs + ad_graphs
            self.labels = torch.zeros(len(graphs))
            self.labels[len(control_graphs):] = 1

        self.labels = self.labels.to(self.device)

        if not self.test:
            words = set()
            for graph in graphs:
                for w in graph.nodes: words.add(w)

            for w in words:
                encode_onehot(w)

        count = 0
        not_found = set()
        for graph in graphs:
            g = dgl.from_networkx(graph.to_directed(), edge_attrs=['weight'])
            # Get a list of words in the order of the nodes
            old_labels = list(graph.nodes)
            new_labels = list(nx.convert_node_labels_to_integers(graph, ordering='sorted').nodes)
            oldnew = {k:v for k,v in zip(old_labels, new_labels)}
            newold = {k:v for v,k in oldnew.items()}
            words = sorted(oldnew, key=oldnew.get)
            # Build input features
            if self.feat == 'fasttext':
                tensors = []
                for w in words:
                    if w not in self.word_vectors.words: not_found.add(w)
                    t = torch.tensor(self.word_vectors.get_word_vector(w))
                    tensors.append(torch.unsqueeze(t, dim=0))
                x = torch.cat(tuple(tensors))
            elif self.feat == 'onehot':
                x = []
                for i, w in enumerate(words):
                    xi = encode_onehot(w)
                    if xi is None:
                        xi = torch.unsqueeze(torch.zeros(861), dim=0)
                    x.append(xi)
                x = torch.cat(tuple(x))
            else:
                x = torch.ones((int(g.num_nodes()), 32))
            # Attach input features
            try:
                g.ndata['x'] = x
            except ValueError:
                breakpoint()
                pass
            g = dgl.add_self_loop(g)
            self.graphs.append(g.to(self.device))
        print(f'{len(not_found)} not found words:', not_found)


    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


def learn(dataset, train_dataloader, valid_dataloader, model, lr,
        epochs, fold, early_stop):
    # Create the model with given dimensions
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    t = 0.5

    prev_accuracy = 0
    for epoch in range(epochs):
        num_correct = 0
        num_valids = 0
        for batched_graph, labels in train_dataloader:
            pred = model(batched_graph, batched_graph.ndata['x'].float()).squeeze()
            loss = F.binary_cross_entropy(pred.reshape(labels.shape), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_correct += ( (pred > t).float() == labels).sum().item()
            num_valids += len(labels)

        train_acc = num_correct/num_valids

        num_correct = 0
        num_valids = 0
        for batched_graph, labels in valid_dataloader:
            pred = model(batched_graph, batched_graph.ndata['x'].float()).squeeze()
            num_correct += ( (pred > t).float() == labels).sum().item()
            num_valids += len(labels)

        valid_acc = num_correct / num_valids
        print("Epoch: %d \t Loss: %.3f \t Train Acc: %.3f\tValid Acc: %.3f"\
                % (epoch, loss, train_acc, valid_acc))
        if epoch > early_stop and valid_acc < prev_accuracy -0.05:
            print("Model stopped")
            break
        torch.save(model, f'results/GAT.{fold}.pt')
        prev_accuracy = valid_acc

    return prev_accuracy

def evaluate(dataset, model):
    loader = GraphDataLoader(dataset, batch_size=1, drop_last=False)
    num_correct, true_positive, label_positives, pred_positives = 0,0,0,0
    n = 0
    t = 0.5
    for batched_graph, labels in loader:
        pred = model(batched_graph, batched_graph.ndata['x'].float()).squeeze()
        pred_b = (pred < t).float()
        num_correct += ( pred_b == labels).sum().item()
        true_positive += (pred_b == labels)*(pred_b == 1).sum().item()
        label_positives += (labels == 1).sum().item()
        pred_positives += (pred_b == 1).sum().item()
        n += len(labels)
    acc = num_correct / n
    precision = float(true_positive/pred_positives)
    recall = float(true_positive/label_positives)
    f1 = float((2*precision*recall)/(precision + recall))


    print("Test Accuracy: %.3f" % acc)

    return [acc, precision, recall, f1]

def get_loaders(dataset, train_idx, val_idx):
    num_examples = len(dataset)
    num_train = int(num_examples * 0.8)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)

    train_dataloader = GraphDataLoader(
        dataset, sampler=train_sampler, batch_size=5, drop_last=False)
    valid_dataloader = GraphDataLoader(
        dataset, sampler=valid_sampler, batch_size=5, drop_last=False)
    return train_dataloader, valid_dataloader

def train_Kfold():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GAT([861,16,8,1],
                num_heads=3,
                feat_drop=0.2,
                attn_drop=0.2,
                negative_slope=0.2,
                share_weights=True,
                residual=True).to(device)
    lr = 0.01
    epochs = 20
    torch.manual_seed(42)
    dataset = Dataset(device, feat='onehot')
    splits=KFold(n_splits=5,shuffle=True,random_state=42)
    accuracy = []

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

        model.reset_parameters()
        print("Fold: %d" % fold)
        train_loader, valid_loader = get_loaders(dataset, train_idx, val_idx)
        accuracy.append(learn(dataset, train_loader, valid_loader, model,lr,
            epochs, fold, early_stop=7))

    best_fold = int(np.argmax(accuracy))
    print("Average validation Accuracy %.3f +/- %.3f" % (np.mean(accuracy), np.std(accuracy)))
    print("Best validation Accuracy %.3f" % accuracy[best_fold])
    with open('results/onehots.p', 'wb') as f:
        pickle.dump(onehots, f)


def train(K, dataset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GAT([861] +[16 for _ in range(K)],
                [1],
                num_heads=3,
                feat_drop=0.6,
                attn_drop=0.6,
                negative_slope=0.2,
                share_weights=True,
                concat=False,
                residual=True).to(device)
    lr = 0.01
    epochs = 100
    torch.manual_seed(42)
    n = len(dataset)
    train_idx, val_idx = torch.arange(int(n*0.9)), torch.arange(int(n*0.9), n)
    train_loader, valid_loader = get_loaders(dataset, train_idx, val_idx)
    learn(dataset, train_loader, valid_loader, model,lr, epochs, 0, early_stop=20)

    with open('results/onehots.p', 'wb') as f:
        pickle.dump(onehots, f)

    return model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = Dataset(device, feat='onehot')
    test_dataset = Dataset(device, test=True, feat='onehot')
    results = {}
    model = train(1, train_dataset)
    return
    # Evaluate
    results[0] = evaluate(test_dataset, model)
    print(results)

    with open('results/h32h32h32h16h16h8-onehot.tsv', 'w') as f:
        f.write('K\taccuracy\tprecision\trecall\tf1\n')
        for k,v in results.items():
            acc, p, r, f1 = v
            f.write(f'{k}\t{acc}\t{p}\t{r}\t{f1}\n')

if __name__ == '__main__':
    main()



