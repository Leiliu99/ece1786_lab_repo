import torch
import torchtext
from torchtext import vocab
# The first time you run this will download a ~823MB file
glove = torchtext.vocab.GloVe(name="6B", # trained on Wikipedia 2014 corpus
                              dim=50)    # embedding size = 50

def print_closest_words(word, vec, n=5):
    dists = torch.norm(glove.vectors - vec, dim=1)     # compute distances to all words
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1]) # sort by distance
    print("----------Euclidean distance for word: %s----------" %word)
    for idx, difference in lst[1:n+1]:                         # take the top n
        print(glove.itos[idx], "\t%5.2f" % difference)


def print_closest_cosine_words(word, vec, n=5):
    cosine_dists = torch.cosine_similarity(glove.vectors, vec.unsqueeze(0))
    cosine_lst = sorted(enumerate(cosine_dists.numpy()), key=lambda x: x[1], reverse=True) # sort by cosine similarity
    print("----------Cosine Similarity for word: %s----------" %word)
    for idx, difference in cosine_lst[1:n+1]:                         # take the top n
        print(glove.itos[idx], "\t%5.2f" % difference)

print_closest_cosine_words("dog", glove["dog"], 5)
print_closest_words("dog", glove["dog"], 10)

print_closest_cosine_words("computer", glove["computer"], 5)
print_closest_words("computer", glove["computer"], 10)

