# -*- coding: utf-8 -*-
"""A1_Section4_starter.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nlUunvG2E7Ji1jcNBiYaFFnJthc62nkg
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from nltk.tokenize import sent_tokenize
import time
import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

import nltk
nltk.download('punkt')

from torch.utils.data import Dataset, DataLoader

# prepare text using the both the nltk sentence tokenizer (https://www.nltk.org/api/nltk.tokenize.html)
# AND the spacy english pipeline (see https://spacy.io/models/en)


def prepare_texts(text, min_frequency=3):

    # Get a callable object from spacy that processes the text - lemmatizes and determines part of speech

    nlp = spacy.load("en_core_web_sm")

    # Some text cleaning. Do it by sentence, and eliminate punctuation.
    lemmas = []
    for sent in sent_tokenize(text):  # sent_tokenize separates the sentences
        for tok in nlp(sent):         # nlp processes as in Part III
            if tok.pos_ not in ["PUNCT", "SPACE", "SYM", "NUM", "X"] and tok.lemma_ not in "[]|.,/?'\"+-=":
                lemmas.append(tok.lemma_)

    # Count the frequency of each lemmatized word
    freqs = Counter()  # word -> occurrence
    for w in lemmas:
        freqs[w] += 1

    vocab = list(freqs.items())  # List of (word, occurrence)
    vocab = sorted(vocab, key=lambda item: item[1], reverse=True)  # Sort by decreasing frequency

    # per Mikolov, don't use the infrequent words, as there isn't much to learn in that case

    frequent_vocab = list(filter(lambda item: item[1]>=min_frequency, vocab))

    # Create the dictionaries to go from word to index or vice-verse

    w2i = {w[0]:i for i,w in enumerate(frequent_vocab)}
    i2w = {i:w[0] for i,w in enumerate(frequent_vocab)}

    # Create an Out Of Vocabulary (oov) token as well
    w2i["<oov>"] = len(frequent_vocab)
    i2w[len(frequent_vocab)] = "<oov>"

    # Set all of the words not included in vocabulary nuas oov
    filtered_lemmas = []
    for lem in lemmas:
        if lem not in w2i:
            filtered_lemmas.append("<oov>")
        else:
            filtered_lemmas.append(lem)

    return filtered_lemmas, w2i, i2w

"""### tokenize_and_preprocess_text creates the training samples for the model. It walks through each word in the corpus, and looks at a window (of size 'window') of words and creates input/output prediction pairs.  We need both positive (in window) samples and negative (out of window) samples."""

def tokenize_and_preprocess_text(filtered_lemmas, textlist, w2i, window=5):
    """
    Skip-gram negative sampling: Predict if the target word is in the context.
    Uses binary prediction so we need both positive and negative samples
    """
    X, T, Y = [], [], []

    # Tokenize the input
    nlp = spacy.load("en_core_web_sm")

    # sent_tokenize separates the sentences: I choose to stay within single sentence
    for sent in sent_tokenize(textlist):
        sentence_tokens = []
        for tok in nlp(sent):         # nlp processes as in Part III
            if tok.pos_ not in ["PUNCT", "SPACE", "SYM", "NUM", "X"] and tok.lemma_ not in "[]|.,/?'\"+-=":
                if tok.lemma_ not in filtered_lemmas: # filter out not frequently used words
                    continue
                if tok.lemma_ in w2i:
                    sentence_tokens.append(w2i[tok.lemma_])
                else:
                    sentence_tokens.append(w2i["<oov>"])

        #prepare X,T,Y in given single sentence tokens
        length = len(sentence_tokens)
        # Loop through each token
        for i, token in enumerate(sentence_tokens):
            # positive examples
            # we try to find context words
            # from (i-window/2) to (i+window/2), need to skip position i itself
            positive_indices = []
            for j in range(i-window//2, i+window//2+1):
                # skip if out of boundary
                if j < 0 or j >= length:
                    continue
                # skip if j=i
                if j == i:
                    continue
                # variable word is: target word
                T.append(token)
                #add context words in X
                X.append(sentence_tokens[j])
                Y.append(1)

                positive_indices.append(sentence_tokens[j])

            # negative examples
            negative_indices = []
            # generate the same number of negatives as positives
            while len(negative_indices) < len(positive_indices):
                neg_candidate = np.random.randint(0,len(w2i)-1)
                # radom candidate should not be the token
                # and should not be within positive indices
                if neg_candidate != token and neg_candidate not in positive_indices:
                    negative_indices.append(neg_candidate)

                    # variable word is: target word
                    T.append(token)
                    #add negative words in X
                    X.append(neg_candidate)
                    Y.append(-1)

    return X, T, Y

"""## Define Model that will be trained to produce word vectors"""

class SkipGramNegativeSampling(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.target_embedding = nn.Embedding(vocab_size, embedding_size)
        self.context_embedding = nn.Embedding(vocab_size, embedding_size)

    def forward(self, x, t):

        # x: torch.tensor of shape (batch_size), context word
        # t: torch.tensor of shape (batch_size), target ("output") word.
        target = self.target_embedding(t)
        context = self.context_embedding(x)

        # perform dot product in batches
        prediction = torch.mul(target, context).sum(dim=1)

        return prediction

"""#### The training function - give it the text and it does the rest"""

def get_batches(T, X, Y, batch_size):
    for i in range(0, len(T) - len(T) % batch_size, batch_size):
        target_batch = T[i:i + batch_size]
        context_batch = X[i:i + batch_size]
        labels_batch = Y[i:i + batch_size]
        yield target_batch, context_batch, labels_batch

def train_sgns(textlist, filtered_lemmas, v2i, window, embedding_size):
    # Set up a model with Skip-gram with negative sampling (predict context with word)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # textlist: a list of strings
    # Create Training Data
    X,T,Y = tokenize_and_preprocess_text(filtered_lemmas, textlist, v2i, window)

    vocab_size = len(v2i)

    # Split the training data
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.2, random_state=42)

    network = SkipGramNegativeSampling(vocab_size, embedding_size)
    network = network.to(device)
    # BCEWithLogitsLoss combines both sigmoid and binary corss-entropy
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(network.parameters(), lr=0.001)

    T_train = torch.tensor(T_train, dtype=torch.long).to(device)
    X_train = torch.tensor(X_train, dtype=torch.long).to(device)
    Y_train = torch.tensor(Y_train, dtype=torch.float).to(device)

    T_test = torch.tensor(T_test, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.long)
    Y_test = torch.tensor(Y_test, dtype=torch.float)


    # training loop
    epochs = 30
    total_trainloss = []
    total_valloss = []
    for epoch in range(epochs):
        epoch_loss = 0
        for target,context,labels in get_batches(T_train, X_train, Y_train, 4):
            target = target.to(device)
            context = context.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = network(t=target, x=context)

            # adjust labels -1 to 0 as required by BCEWithLogitsLoss
            labels_fixed = (labels+1)/2

            loss = loss_function(predictions, labels_fixed.float())
            # backpropagate and update the weights
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}, Train Loss: {epoch_loss/len(T_train)}')
        total_trainloss.append(epoch_loss/len(T_train))

        # validation
        val_loss = 0
        with torch.no_grad():
            for target,context,labels in get_batches(T_test, X_test, Y_test, 4):
                predictions = network(t=target, x=context)

                # adjust labels -1 to 0 as required by BCEWithLogitsLoss
                labels_fixed = (labels+1)/2

                loss = loss_function(predictions, labels_fixed.float())

                val_loss += loss.item()
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss/len(T_test)}')
        total_valloss.append(val_loss/len(T_test))

    plt.plot(total_trainloss, label = "Training loss")
    plt.plot(total_valloss, label = "Validation loss")
    plt.title("loss vs. Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    return network

if __name__ == "__main__":

    with open('/content/LargerCorpus.txt','r', encoding='UTF-8') as f:
        txt = f.read()
    filtered_lemmas, v2i, i2v = prepare_texts(txt)

    #Run Training and retrieve embedding
    # Run the training loop
    np.random.seed(43)
    torch.manual_seed(43)
    network = train_sgns(txt, filtered_lemmas, v2i, 5, 8)





"""### Reduce the Dimensionality of Embeddings and Display"""

from sklearn.decomposition import PCA #see https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

def visualize_embedding(embedding, i2v,most_frequent_from=20, most_frequent_to=80):
    print ("Visualizing the {} to {} most frequent words".format(most_frequent_from, most_frequent_to))

    # since the embeddings are ordered from most frequent words to least frequent,
    # we can easily select a sub range of the most frequent words:
    selected_words = embedding[most_frequent_from:most_frequent_to, :]

    # The function below will reduce a vector to 2 principle components
    pca = PCA(n_components=2)

    # Transform the selected embeddings to have 2 dimensions
    embeddings = pca.fit_transform(selected_words)

    # Plot the the reduced embeddings - a point and the word itself
    plt.figure()
    X = embeddings[:, 0]
    Y = embeddings[:, 1]
    for i, (x,y) in enumerate(embeddings):
        plt.scatter(x, y, marker='o', label=i2v[i])
        plt.text(x+0.03, y+0.03, i2v[i])

    plt.title("2-dimensional plot for embeddings")
    plt.xlabel("dimension 0")
    plt.ylabel("dimension 1")
    plt.grid(True)
    plt.show()

visualize_embedding(network.target_embedding.weight.detach().numpy(), i2v, most_frequent_from=20, most_frequent_to=55)