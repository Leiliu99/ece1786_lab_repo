# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from nltk.tokenize import sent_tokenize
import time
import torch
import spacy
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

import nltk
nltk.download('punkt')

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

if __name__ == "__main__":
    with open('/content/LargerCorpus.txt','r', encoding='UTF-8') as f:
        txt = f.read()
    filtered_lemmas, v2i, i2v = prepare_texts(txt)
    X,T,Y = tokenize_and_preprocess_text(filtered_lemmas, txt, v2i, 5)

    print("Now the examples are reduced to: "+str(len(X)))