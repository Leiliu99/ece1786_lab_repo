
import torch
import torch.nn as nn

"""## Define Model that will be trained to produce word vectors"""
class Word2vecModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        # initialize word vectors to random numbers
        self.embeddings = nn.Embedding(vocab_size, embedding_size)

        # prediction function takes embedding as input, and predicts which word in vocabulary as output
        self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, x):
        """
        x: torch.tensor of shape (bsz), bsz is the batch size
        """
        e = self.embeddings(x)
        logits = self.linear(e)
        return logits, e