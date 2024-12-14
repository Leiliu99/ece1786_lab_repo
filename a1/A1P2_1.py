import torch
import torchtext
from torchtext import vocab
# The first time you run this will download a ~823MB file
glove = torchtext.vocab.GloVe(name="6B", # trained on Wikipedia 2014 corpus
                              dim=50)    # embedding size = 50

def compare_words_to_category(category, target):
    #result1: calculate cosine similarity and then take average
    sum_cosine = 0
    category_len = len(category)
    for word in category:
        sum_cosine += torch.cosine_similarity(glove[word].unsqueeze(0), target.unsqueeze(0))
    result1 = sum_cosine/category_len

    #result2: take average, take cosine similarity with the average
    category_vec_list = []
    for word in category:
        category_vec_list.append(glove[word].unsqueeze(0))
    
    embedding_average = torch.mean(torch.cat(category_vec_list), 0)
    result2 = torch.cosine_similarity(embedding_average, target.unsqueeze(0))

    return result1,result2


category = ["sun", "moon", "winter", "rain", "cow", "wrist", 
            "wind", "prefix", "ghost", "glow", "heated", "cool"]
result1, result2 = compare_words_to_category(category, glove["sky"])
print(f"method1 result: {result1}, method2 result: {result2}")