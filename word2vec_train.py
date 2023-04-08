import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torch
import operator
from nltk.probability import FreqDist

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

context_size = 3
negative_num = 100
batch = 128
num_epochs = 1
learning_rate = 0.001
decline_rate = 0.95
embedding_size = 128
# data_file = "abstract_split_nltk.txt"
data_file = "title_split_nltk.txt"
log_file = "train_log.txt"
corpus_file = "corpus.txt"

with open(log_file, 'w') as f:
    f.write('')

with open(data_file, "r", encoding="utf-8") as f:
    lines = f.read().strip().split('\n')

# words_set = set()
vocabulary = []
vocab_effective = []
for words in lines:
    words = words.strip(' ')
    words = words.split()
    if len(words) >= 2 * context_size + 1:    
        for word in words:
            # words_set.add(word)
            vocabulary.append(word)

fDist = FreqDist(vocabulary)
fDist = dict(sorted(fDist.items(), key=operator.itemgetter(1), reverse=True))
words_set = list(fDist.keys())
with open(corpus_file, 'w', encoding='utf-8') as f:
    for i in range(0, len(words_set)):
        f.write(words_set[i] + '\n')
# build index
word_to_id = {word : i for i, word in enumerate(words_set)}
id_to_word = {word_to_id[word] : word for word in word_to_id}
# compute word frequency
word_to_counts = {word : fDist[word] for word in words_set}
word_counts = np.array([word_c for word_c in word_to_counts.values()], dtype=np.float32)
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3./4.)
word_freqs = word_freqs / np.sum(word_freqs)
vocabulary_size = len(word_to_id)

# implement data loader
class word_embedding_dataset(D.Dataset):
    def __init__(self, vocabulary, word_to_id, id_to_word, word_freqs, word_counts):
        super(word_embedding_dataset, self).__init__()
        self.vocab_encoded = [word_to_id.get(word, vocabulary_size - 1) for word in vocabulary]
        self.vocab_encoded = torch.Tensor(self.vocab_encoded).long()
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)

    def __len__(self):
        return len(self.vocab_encoded)

    def __getitem__(self, idx):
        center_word = self.vocab_encoded[idx]
        if idx + context_size >= len(self.vocab_encoded):
            context_idx = list(range(idx - context_size, idx)) + list(range(idx + 1, len(self.vocab_encoded)))
            for i in range(0, idx + context_size + 1 - len(self.vocab_encoded)):
                context_idx.append(len(self.vocab_encoded) - 1)
        elif idx - context_size < 0:
            context_idx = list(range(0, idx)) + list(range(idx + 1, idx + context_size + 1))
            for i in range(0, 0 - (idx - context_size)):
                context_idx.append(0)
        else:
            context_idx = list(range(idx - context_size, idx)) + list(range(idx + 1, idx + context_size + 1))
        context_words = self.vocab_encoded[context_idx]
        # negative_words = multinomial(self.word_freqs, negative_num * context_words.shape[0], True)
        # negative_words = torch.distributions.multinomial(self.word_freqs, negative_num * context_words.shape[0], True)
        negative_words = torch.multinomial(self.word_freqs, negative_num * context_words.shape[0], True)
        return center_word, context_words, negative_words

# dataset = word_embedding_dataset(vocabulary, word_to_id, id_to_word, word_freqs, word_counts)
# data_loader = D.DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=0)

class embedding_model(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(embedding_model, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        init_range = 0.5 / self.embed_size
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.in_embed.weight.data.uniform_(-init_range, init_range)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, input_labels, context_labels, negative_lables):
        batch_size = input_labels.size(0)
        input_embed = self.in_embed(input_labels)
        context_embed = self.out_embed(context_labels)
        negative_embed = self.out_embed(negative_lables)
        log_context = torch.bmm(context_embed, input_embed.unsqueeze(2)).squeeze()
        log_negative = torch.bmm(negative_embed, -input_embed.unsqueeze(2)).squeeze()
        log_context = F.logsigmoid(log_context).sum(1)
        log_negative = F.logsigmoid(log_negative).sum(1)
        loss = log_context + log_negative
        return -loss
    
    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()

model = embedding_model(vocabulary_size, embedding_size)
if USE_CUDA:
    model = model.cuda()
dataset = word_embedding_dataset(vocabulary, word_to_id, id_to_word, word_freqs, word_counts)
data_loader = D.DataLoader(dataset, batch_size=batch, shuffle=True)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(0, num_epochs):
    for i, (input_labels, context_labels, negative_labels) in enumerate(data_loader):
        input_labels = input_labels.long()
        context_labels = context_labels.long()
        negative_labels = negative_labels.long()
        if USE_CUDA:
            input_labels = input_labels.cuda()
            context_labels = context_labels.cuda()
            negative_labels = negative_labels.cuda()

        optimizer.zero_grad()
        loss = model(input_labels, context_labels, negative_labels).mean()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            with open(log_file, 'a') as f:
                f.write("epoch:{}, iter:{}, loss:{}\n".format(epoch, i, loss))
            print("epoch:{}, iter:{}, loss:{}\n".format(epoch, i, loss))

        if i % 1000 == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * decline_rate

    embedding_weights = model.input_embeddings()
    np.savetxt("embedding_weights.csv", embedding_weights, delimiter=',')
    torch.save(model.state_dict(), "embedding_weights.th")



