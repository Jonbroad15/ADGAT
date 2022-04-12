import numpy as np
from sklearn.manifold import TSNE
import fasttext
import matplotlib.pyplot as plt
import seaborn as sns

model = fasttext.load_model('word_vectors_double.bin')

X = model.get_output_matrix()
embedded_X = TSNE(n_components=2, init='random').fit_transform(X)


x, y = embedded_X[:,0], embedded_X[:,1]

from matplotlib.colors import ListedColormap


# get colormap from seaborn
cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())


sc = plt.figure(figsize=(12,7))
sns.scatterplot(x, y,
    palette=sns.color_palette("hls", 10),
    c=x,
    cmap=cmap,
    legend="full",
    alpha=0.7
)
#legend
# plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)


for i, txt in enumerate(model.words):
    if i % 10 == 0:
        plt.annotate(txt, (x[i], y[i]))

plt.ylabel('tSNE-2d-two')
plt.xlabel('tSNE-2d-one')
plt.title('Visualization of 300D fastText Word Embeddings by 2D tSNE')

plt.savefig('/Users/jbroadbent/OneDrive/School/UofT/Winter22/csc2511/Figures/tSNE.png')
