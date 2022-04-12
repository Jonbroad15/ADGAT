import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy
import pdb
from nltk.corpus import stopwords
import graph_utils.network_measures as nm
import operator
import graph_utils.chat as pla
import random
from itertools import islice
import collections
from scipy.stats import spearmanr, skew, kurtosis
from sklearn.feature_extraction import DictVectorizer

def match_ad_dct_with_dataset(ad_dct, test_graphs):
    num_graphs = len(ad_dct)
    labels = np.zeros(num_graphs)

    for i,graph in enumerate(test_graphs):
        id = graph.graph['ID'] 
        labels[i] = ad_dct[id]

    return labels

def build_mmse_dataset(labels, meta_data, X, y=None):
    # Not all of the dataset has MMSE labels so we remove the values that do not
    ind_to_keep = []
    mmse_y = []
    for i,x in enumerate(labels):
        try:
            mmse_y.append(meta_data[x])
            ind_to_keep.append(i)
        except KeyError as e:
            print(x)
    X_new = X[ind_to_keep, :]

    if y is not None:
        ret_y = y[ind_to_keep]
        return X_new, np.array(mmse_y), ret_y

    return X_new, np.array(mmse_y)

def read_meta_data(path, data_dct):
    f = open(path, "r")
    txt = f.read()
    lines = txt.split('\n')

    for l in lines[1:]:
        tokens = l.split(";")
        try:
            data_dct[tokens[0].rstrip()] = int(tokens[3].rstrip().lstrip())
        except ValueError as e:
            print()

    return data_dct

def read_meta_test_data(path):
    f = open(path, "r")
    txt = f.read()
    lines = txt.split('\n')
    ad_dct = {}
    mmse_dct = {}

    for l in lines[1:]:
        tokens = l.split(";")
        try:
            ad_dct[tokens[0].rstrip()] = int(tokens[3].rstrip().lstrip())
            mmse_dct[tokens[0].rstrip()] = int(tokens[4].rstrip().lstrip())
        except ValueError as e:
            print()

    return ad_dct, mmse_dct

def random_chunk(li, min_chunk=1, max_chunk=3):
    it = iter(li)
    while True:
        nxt = list(islice(it,random.randint(min_chunk,max_chunk)))
        if nxt:
            yield nxt
        else:
            break

def file_to_sentence_graphs(path, remove_stopwords=False, remove_nonalpha=False, occurrence_window=3, pos_tags=False, shuffle=False):
    chat = pla.read_chat(path)
    if pos_tags:
        sents = chat.tagged_sents(participant='PAR')
    else:
        sents = chat.sents(participant='PAR')

    #f = open(path, "r")
    #txt = f.read()
    #lines = txt.split('\n')
    #participant = list(filter(lambda x: x.startswith("*PAR"), lines))
    #participant = list(map(lambda  x: x[6:], participant))
    G = nx.Graph()
    """
    if remove_stopwords:
        words = [word for word in words if word not in stopwords.words('english')]
    if remove_nonalpha:
        words = list([''.join(ch for ch in word if ch.isalpha()) for word in words])

    words = list(filter(lambda x: len(x) > 0, words))
    if shuffle:
        random.shuffle(words)
    """
    #print(chat.words())
    # Shuffling is a bit more involved that in a straight graph
    # Firstly we flatten the list
    if shuffle:
        words = []
        for sent in sents:
            sent = list([''.join(ch for ch in word if ch.isalpha()) for word in sent])
            words += sent
            words += ['\n']
        random.shuffle(words)
        # Find the newlines and split them back into sentences
        sents = []
        current_sent = []
        for w in words:
            if w != '\n':
                current_sent.append(w)
            else:
                sents.append(current_sent)
                current_sent = []
        if len(current_sent) > 0:
            sents.append(current_sent)

    G = nx.Graph()
    G.graph['ID'] = path[-8:-4]
    for sent in sents:
        if remove_stopwords: 
            sent = [word for word in sent if word not in stopwords.words('english')]
        if remove_nonalpha and not pos_tags:
            sent = list([''.join(ch for ch in word if ch.isalpha()) for word in sent])
            pass
        sent = list(filter(lambda x: len(x) > 0, sent))
        sentence_length = len(sent)
        for i in range(sentence_length):
            for j in range(1, occurrence_window+1):
                if i+j >= sentence_length:
                    #print(i+j)
                    break

                if pos_tags:
                    word_1 = sent[i][0]
                    word_2 = sent[i+j][0]

                    # One of the transcripts is missing the POS tags
                    if len(sent[i][1]) > 0: 
                        pos_tag_1 = sent[i][1][0]
                    else:
                        pos_tag_1 = ""
                    if len(sent[i+j][1]) > 0:
                        pos_tag_2 = sent[i+j][1][0]
                    else:
                        pos_tag_2 = ""                    

                    label_1 = word_1 + " " + pos_tag_1
                    label_2 = word_2 + " " + pos_tag_2
                else:
                    label_1 = sent[i]
                    label_2 = sent[i+j]

                if not G.has_node(label_1):
                    G.add_node(label_1)

                    if pos_tags:
                        nx.set_node_attributes(G, {label_1 : word_1}, "word")
                        nx.set_node_attributes(G, {label_1 : pos_tag_1}, "pos_tag")

                if not G.has_node(label_2):
                    G.add_node(label_2)

                    if pos_tags:
                        nx.set_node_attributes(G, {label_2 : word_2}, "word")
                        nx.set_node_attributes(G, {label_2 : pos_tag_2}, "pos_tag")

                if G.has_edge(label_1, label_2):
                    G[label_1][label_2]['weight'] += 1
                else:
                    G.add_edge(label_1, label_2, weight=1)

    return G


def file_to_graph(path, remove_stopwords=False, remove_nonalpha=True, occurrence_window=3, shuffle=False):
    chat = pla.read_chat(path)
    sents = chat.sents(participant='PAR')
    words = chat.words(participant='PAR')
    #f = open(path, "r")
    #txt = f.read()
    #lines = txt.split('\n')

    #participant = list(filter(lambda x: x.startswith("*PAR"), lines))
    #participant = list(map(lambda  x: x[6:], participant))
    G = nx.Graph()

    if remove_stopwords:
        words = [word for word in words if word not in stopwords.words('english')]
    if remove_nonalpha:
        words = list([''.join(ch for ch in word if ch.isalpha()) for word in words])

    words = list(filter(lambda x: len(x) > 0, words))
    if shuffle:
        random.shuffle(words)

    word_set = set(words)

    G = nx.Graph()
    G.add_nodes_from(word_set)
    G.graph['ID'] = path[-8:-4]
    #for sent in words:
    l = len(words)

    for i in range(l):
        for j in range(1, occurrence_window+1):
            if i+j >= l:
                break
            if G.has_edge(words[i], words[i+j]):
                G[words[i]][words[i+j]]['weight'] += 1
            else:
                G.add_edge(words[i], words[j+i], weight=1)

    return G

def load_all_files(folder, occurrence_window=2, remove_nonalpha = True, pos_tags = False, shuffle=False):
    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    graphs = []
    for f in onlyfiles:
        if not f.endswith(".cha"): continue
        try:
            G = file_to_sentence_graphs(os.path.join(folder, f), occurrence_window=occurrence_window, remove_nonalpha=remove_nonalpha, pos_tags=pos_tags, shuffle=shuffle)
        except StopIteration:
            continue
        # if not nx.is_connected(G):
            # print("Disconnected graph")
        graphs.append(G)

    return graphs

def load_shuffled_files(folder, occurrence_window,  remove_nonalpha=True, num_shuffles = 50):
    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    graphs = []
    for f in onlyfiles:
        shuffles = []
        for i in range(num_shuffles):
            G = file_to_sentence_graphs(os.path.join(folder, f), occurrence_window=occurrence_window, remove_nonalpha=remove_nonalpha, pos_tags=False, shuffle=True)
            shuffles.append(G)
        graphs.append(shuffles)

    return graphs

def vectorize_group(graphs):
    """
    Turns a list of graphs into a matrix of values we can do something with
    """
    X = np.zeros((len(graphs), 13))
    for i,G in enumerate(graphs):
        alpha, xmin, D = nm.calculate_exponent(G)
        X[i, 0] = len(G)
        X[i, 1] = G.number_of_edges()
        X[i, 2] = G.number_of_edges()/(len(G) * (len(G) - 1))
        X[i, 3] = nx.number_of_selfloops(G)/len(G)
        X[i, 4] = nx.average_clustering(G)
        X[i, 5] = nm.calculate_diameter(G)
        X[i, 6] = nm.calculate_network_heterogeneity(G)
        X[i, 7] = nm.get_centralization(G)
        X[i, 8] = nm.get_average_shortest_path_length(G)
        X[i, 9] = alpha
        #X[i, 10] = nm.sigma(G)
        X[i, 10] = xmin
        X[i, 11] = nx.degree_pearson_correlation_coefficient(G)
        X[i, 12] = nm.calculate_average_nearest_neighbour_exponent(G)
        #X[i, 12] = D
    return X

def calculate_unigram_stats(folder):
    dcts = []
    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    for f in onlyfiles:    
        chat = pla.read_chat(os.path.join(folder, f))
        dcts.append(list(chat.word_ngrams(1, participant='PAR').values()))

    X = np.zeros((len(onlyfiles), 6))
    
    for i, vals in enumerate(dcts):
        X[i, 0] = len(vals)
        X[i, 1] = sum(vals)
        X[i, 2] = np.mean(vals)
        X[i, 3] = np.std(vals)
        X[i, 4] = skew(vals)
        X[i, 5] = kurtosis(vals)

    return X


def sort_dict(dct):
    """
    Takes a dict and returns a sorted list of key value pairs
    """
    sorted_x = sorted(dct.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_x

def embed_group(graphs, k):
    """
    Embeds the group using the graph laplacian
    """
    X = np.zeros((len(graphs), k))
    for i, G in enumerate(graphs):
        X[i, :] = embed_graph(G, k)
    return X

def embed_graph(graph, embedding_dimension=16, normalized=True):
    
    # Padding with zeros
    embedding = np.zeros(embedding_dimension)
    
    # Usage of networkx graph objects
    adj_matrix = nx.adj_matrix(graph)
    n_nodes, m_nodes = adj_matrix.shape
    k = min(embedding_dimension + 1, n_nodes - 1)

    if normalized:
        laplacian = nx.normalized_laplacian_matrix(graph)
    else:
        laplacian = nx.laplacian_matrix(graph)

    # Minus the eigen decomposition of minus the Laplacian is more stable than directly
    # computing the eigen decomposition of the Laplacian
    ncv = min(n_nodes, max(2*k + 1, 20))
    while True:
        try:
            v0 = np.random.uniform(-1, 1, laplacian.shape[0])
            eigenvalues = scipy.sparse.linalg.eigsh(-laplacian, k=k, sigma=1.0, which='LM', tol=1e-6, v0=v0, ncv=ncv, return_eigenvectors=False)
            embedding[:len(eigenvalues)-1] = sorted(-eigenvalues)[1:]
            break
        except scipy.sparse.linalg.ArpackError as e:
            print(e)

            ncv += 1
    
    return embedding

def relabel_graphs(graphs):
    ret_graphs = []
    for G in graphs:
        ret_graphs.append(nx.convert_node_labels_to_integers(G))

    return ret_graphs

def padded_spectral(graph, embedding_dimension=16, normalized=True):
    
    # Padding with zeros
    embedding = np.zeros(embedding_dimension)
    
    # Usage of networkx graph objects
    adj_matrix = nx.adj_matrix(graph)
    n_nodes, m_nodes = adj_matrix.shape
    k = min(embedding_dimension + 1, n_nodes - 1)

    if normalized:
        laplacian = nx.normalized_laplacian_matrix(graph)
    else:
        laplacian = nx.laplacian_matrix(graph)

    # Minus the eigen decomposition of minus the Laplacian is more stable than directly
    # computing the eigen decomposition of the Laplacian
    
    v0 = np.random.uniform(-1, 1, laplacian.shape[0])
    eigenvalues = sparse.linalg.eigsh(-laplacian, k=k, sigma=1.0, which='LM', tol=1e-6, v0=v0, return_eigenvectors=False)
    embedding[:len(eigenvalues)-1] = sorted(-eigenvalues)[1:]
    
    return embedding
