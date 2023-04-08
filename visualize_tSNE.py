import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.manifold import TSNE

embedding_weights_file = "embedding_weights.csv"
corpus_file = "corpus.txt"
biomedical_category = "biomedical_entities.txt"
labels = []
biomedical_entities = []
entity_to_color = []
entities = []
num_show_words = 1000

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
with open(embedding_weights_file) as f:
    embedding_weights = np.loadtxt(f, delimiter=',')
embedding_weights_show = embedding_weights[range(0, num_show_words), :]
embedding_2D = tsne.fit_transform(embedding_weights_show)
with open(corpus_file, "r", encoding="utf-8") as f:
    words = f.read().split('\n')
word_to_id = {word : i for i, word in enumerate(words)}
id_to_word = {word_to_id[word] : word for word in word_to_id}
for i in range(0, num_show_words):
    labels.append(words[i])
with open(biomedical_category, "r", encoding="utf-8") as f:
    biomedical_entities = f.read().split('\n')
for entity in biomedical_entities:
    entities_plt = entity.split()
    entities.append(entities_plt[0])
    entity_to_color.append(entities_plt)

entity_to_color = dict(entity_to_color)

plt.figure(dpi=300, figsize=(30, 24))
for i, label in enumerate(labels):
    x, y = embedding_2D[i, :]
    plt.scatter(x, y, s=(5000 / math.sqrt(i + 1)))
    plt.annotate(label, (x, y), ha='center', va='top')

plt.savefig('word2vec_tSNE.png')

plt.figure(dpi=300, figsize=(15, 12))
for i, label in enumerate(labels):
    x, y = embedding_2D[i, :]
    if label in entities:
        plt.scatter(x, y, s=100, c=entity_to_color[label])
        plt.annotate(label, (x, y), ha='center', va='top')

plt.savefig('word2vec_tSNE_bio.png')

def find_near_words(word):
    index = word_to_id[word]
    embedding = embedding_weights[index]
    cos_d = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    cos_d_sort = cos_d.argsort()
    near_words = []
    for i in range(0, 50):
        # if id_to_word[cos_d_sort[i]] in entities:
        near_words.append(id_to_word[cos_d_sort[i]])
    return near_words

print(find_near_words('covid-19'))

