# ADGAT
Classification of Alzheimer's Disease through Graph Attention Learning on Word Co-occurrence Networks

We construct an architecture based off [GATv2Conv](https://github.com/tech-srl/how_attentive_are_gats) developed by _Brody et al. 2022_. We modify the convolutional layers to pay attention to edge weights in our co-occurrence networks.
[GAT architecture](images/GATarchitecture.png)

### 1. Download training data
http://www.homepages.ed.ac.uk/sluzfil/ADReSS/

### 2. Build word vectors
Run `word_vectors.py` on the Dementia Bank transcripts

### 3. Train and edit hyperparameters
Use `train_KFold()` in `model.py` to experiment with desired hyperparameters on train/validation splits

### 3. Evaluate Model
Use `evaluate()` in `model.py` to test a given model on the ADReSS 2020 test set.


[h16_results](images/h16.png)
[h32_results](images/h32.png)
