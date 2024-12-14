import torch
from torch.nn import functional as F

def generate(self, idx, device, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        max_probs = torch.tensor([]).to(device)
        six_indices = torch.tensor([], dtype=torch.int64).view(0,6).to(device)
        six_probs = torch.tensor([]).view(0,6).to(device)

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            idx_cond.to(device)
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                prob_next, idx_next = torch.topk(probs, k=6, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next[0][0].reshape(1, 1)), dim=1)

            #record the max probability in each round
            max_probs = torch.cat((max_probs, prob_next[0][0].reshape(1, 1)), dim=0)

            #record the six highest index and probability
            six_indices = torch.cat((six_indices, idx_next), dim=0)
            six_probs = torch.cat((six_probs, prob_next), dim=0)

        return idx, max_probs, six_indices, six_probs
