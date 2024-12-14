import torch 
import numpy as np

from nltk.tokenize import sent_tokenize 

from pathlib import Path 
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from mingpt.bpe import BPETokenizer 
from mingpt.utils import set_seed 

set_seed(1234)
import datasets

class SentiDataset(Dataset):
    
    def __init__(self, ds_choice="small", split="train", truncation=-1):
        
        self.ds_choice = ds_choice #still need?
        self.truncation = truncation  # int. If -1, then
        
        #get sst first 1200 train dataset
        sst = datasets.load_dataset('glue', 'sst2')
        first1200 = sst['train'][:1200]

         # Train / test split
        train_sent, val_sent, train_label, val_label = train_test_split(first1200['sentence'], first1200['label'],
                                                                test_size=0.2, shuffle=True)
        if split == "train":
            raw_data,label = train_sent, train_label
        else:
            raw_data,label = val_sent, val_label

        # Tokenize
        self.tokenizer = BPETokenizer()
        self.data = []  # List of 1-d pytorch tensor
        self.label = []

        for (x,y) in zip(raw_data,label):
            tokenized = self.tokenizer(x).view(-1)  # pytorch tensor
            if truncation >= 0:
                # self.data.append((tokenized[:truncation],y))
                self.data.append(tokenized[:truncation])
            else:
                # self.data.append((tokenized, y))
                self.data.append(tokenized)

            self.label.append(torch.tensor(y).float())

        # Count some items
        #leave it as 512 to fit trained model block size
        # self.max_sentence_length = np.max([len(d) for d in raw_data])
        self.max_sentence_length = 512

    def __len__(self):
        return len(self.data)

    def get_vocab_size(self):
        """
        We have to set this to the max vocab size (i.e., that decided by the BPE tokenizer), 
        but actually, only a small number of vocab is used, especially for the small text. 
        """
        return 50257

    def __getitem__(self, idx):
        """
        The output should be a tuple x and y, both as pytorch tensors.
        Please refer to the `run()` method in the mingpt/trainer.py script for 
        how the x and y are going to be used.
        """
        # x = self.data[idx][0]
        # y = self.data[idx][1]
        x = self.data[idx]
        y = self.label[idx]
        return (x, y)

    def get_block_size(self):
        """
        block_size is the size at which lines are truncated to ensure they are equal-length.
        """
        return self.max_sentence_length