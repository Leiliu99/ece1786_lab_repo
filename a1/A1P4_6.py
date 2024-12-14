import torch
import torch.nn as nn

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