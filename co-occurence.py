import operator
import numpy as np
from nltk.probability import FreqDist

center_word = 'covid-19'
context_words = []
context_size = 3
data_file = "title_split_nltk.txt"
with open(data_file, "r", encoding="utf-8") as f:
    lines = f.read().strip().split('\n')

for words in lines:
    words = words.strip(' ')
    words = words.split()
    if center_word in words:
        idx = words.index(center_word)
        if idx - context_size < 0 and idx + context_size >= len(words):
            context_idx = list(range(0, idx)) + list(range(idx + 1, len(words)))
        elif idx - context_size < 0:
            context_idx = list(range(0, idx)) + list(range(idx + 1, idx + context_size + 1))
        elif idx + context_size >= len(words):
            context_idx = list(range(idx - context_size, idx)) + list(range(idx + 1, len(words)))
        else:
            context_idx = list(range(idx - context_size, idx)) + list(range(idx + 1, idx + context_size + 1))
        for i in context_idx:
            context_words.append(words[i])

fDist = FreqDist(context_words)
fDist = dict(sorted(fDist.items(), key=operator.itemgetter(1), reverse=True))
words_set = list(fDist.keys())
word_to_counts = {word : fDist[word] for word in words_set}
word_counts = np.array([word_c for word_c in word_to_counts.values()], dtype=np.float32)
word_to_freqs = {word : (fDist[word] / np.sum(word_counts)) for word in words_set}

print(list(word_to_freqs.items())[: 50])




