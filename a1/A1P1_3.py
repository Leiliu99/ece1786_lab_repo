import torch
import torchtext
from torchtext import vocab
# The first time you run this will download a ~823MB file
glove = torchtext.vocab.GloVe(name="6B", # trained on Wikipedia 2014 corpus
                              dim=50)    # embedding size = 50

def print_closest_words(word, vec, n=5):
    dists = torch.norm(glove.vectors - vec, dim=1)     # compute distances to all words
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1]) # sort by distance
    print("---------- second word given: %s----------" %word)
    for idx, difference in lst[1:n+1]:                         # take the top n
        print(glove.itos[idx], "\t%5.2f" % difference)

examples = ["run", "dance", "listen", "speak", "eat", "drink", "draw", "play", "write", "walk"]

for first in examples:
    vec = glove["thinking"] - glove["think"] + glove[first]
    print_closest_words(first, vec)