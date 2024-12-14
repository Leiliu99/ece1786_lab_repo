from transformers import AutoModelForCausalLM, GPT2Tokenizer
import torch
import torch.nn.functional as F
from treelib import Tree
import numpy as np
import codecs

#load the GPT2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt_text = "It is important for all countries to try harder to reduce carbon emissions because"
input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids

#1.2 generated tokens with probabilities
max_length = 30
temperature = 0.5
top_p = 0.8
outputs = model.generate(
    input_ids,
    max_length=max_length,
    temperature=temperature,
    top_p=top_p,
    do_sample=True,
    return_dict_in_generate=True,
    output_scores=True,
)

generated_tokens = outputs.sequences
scores = outputs.scores  

# Convert logits to probabilities for each token
probabilities = [F.softmax(score, dim=-1) for score in scores]

# Decode and print the generated text
generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print("Generated Text:", generated_text)

for i, probs in enumerate(probabilities):
    token_id = generated_tokens[0][i + 14]
    token = tokenizer.decode(token_id)
    # print(probs.shape)
    token_prob = probs[0][token_id].item()
    print(f"Token: {token} | Probability: {token_prob:.4f}")

# 1.3 Tree Generation
tree = Tree()
node = 0
tree.create_node(str(tokenizer.batch_decode(generated_tokens[0:1,13], skip_special_tokens=True)), node)  # root node
parent=node

for i, score in enumerate(scores):
    generated_token = generated_tokens[0][i+14]

    prob = F.softmax(score, dim=-1)
    top_probs, top_indices = torch.topk(prob, k=2)

    for j, (t,p) in enumerate(zip(top_indices[0], top_probs[0])):
        val=str(tokenizer.decode(t.item()).strip())+": "+str(p.item())
        node += 1
        if t == generated_token:
            print(f"Generated: {tokenizer.decode(t.item()).strip()} in top {j+1}")
            parent_next=node
        
        tree.create_node(val, node, parent=parent)

    if parent==parent_next:
        print(f"Generated: {tokenizer.batch_decode(generated_tokens[0:1,i+14])} not in top 2!")
        node += 1
        prent_next=node
        tree.create_node(str(tokenizer.batch_decode(generated_tokens[0:1,i+14])), node ,parent=parent)
    
    parent = parent_next

# Display the tree
tree.show()

#hardcode here to show tree in graphics
x = b'[\' because\']\n\xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80 it: 0.5718349814414978\n\xe2\x94\x94\xe2\x94\x80\xe2\x94\x80 the: 0.1576509028673172\n    \xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80 !: 0.0\n    \xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80 [\' effects\']\n    \xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80 global: 0.041224028915166855\n    \xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80 of: 1.0\n    \xe2\x94\x82   \xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80 !: 0.0\n    \xe2\x94\x82   \xe2\x94\x94\xe2\x94\x80\xe2\x94\x80 climate: 1.0\n    \xe2\x94\x82       \xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80 !: 0.0\n    \xe2\x94\x82       \xe2\x94\x94\xe2\x94\x80\xe2\x94\x80 change: 1.0\n    \xe2\x94\x82           \xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80 are: 0.8243514895439148\n    \xe2\x94\x82           \xe2\x94\x82   \xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80 already: 0.6174745559692383\n    \xe2\x94\x82           \xe2\x94\x82   \xe2\x94\x82   \xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80 being: 0.9258776307106018\n    \xe2\x94\x82           \xe2\x94\x82   \xe2\x94\x82   \xe2\x94\x82   \xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80 !: 0.0\n    \xe2\x94\x82           \xe2\x94\x82   \xe2\x94\x82   \xe2\x94\x82   \xe2\x94\x94\xe2\x94\x80\xe2\x94\x80 felt: 1.0\n    \xe2\x94\x82           \xe2\x94\x82   \xe2\x94\x82   \xe2\x94\x82       \xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80 .: 0.3378863036632538\n    \xe2\x94\x82           \xe2\x94\x82   \xe2\x94\x82   \xe2\x94\x82       \xe2\x94\x94\xe2\x94\x80\xe2\x94\x80 in: 0.4839702546596527\n    \xe2\x94\x82           \xe2\x94\x82   \xe2\x94\x82   \xe2\x94\x82           \xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80 !: 0.0\n    \xe2\x94\x82           \xe2\x94\x82   \xe2\x94\x82   \xe2\x94\x82           \xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80 [\' other\']\n    \xe2\x94\x82           \xe2\x94\x82   \xe2\x94\x82   \xe2\x94\x82           \xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80 many: 0.4273974597454071\n    \xe2\x94\x82           \xe2\x94\x82   \xe2\x94\x82   \xe2\x94\x82           \xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80 parts: 1.0\n    \xe2\x94\x82           \xe2\x94\x82   \xe2\x94\x82   \xe2\x94\x82           \xe2\x94\x82   \xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80 !: 0.0\n    \xe2\x94\x82           \xe2\x94\x82   \xe2\x94\x82   \xe2\x94\x82           \xe2\x94\x82   \xe2\x94\x94\xe2\x94\x80\xe2\x94\x80 of: 1.0\n    \xe2\x94\x82           \xe2\x94\x82   \xe2\x94\x82   \xe2\x94\x82           \xe2\x94\x82       \xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80 !: 0.0\n    \xe2\x94\x82           \xe2\x94\x82   \xe2\x94\x82   \xe2\x94\x82           \xe2\x94\x82       \xe2\x94\x94\xe2\x94\x80\xe2\x94\x80 the: 1.0\n    \xe2\x94\x82           \xe2\x94\x82   \xe2\x94\x82   \xe2\x94\x82           \xe2\x94\x82           \xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80 !: 0.0\n    \xe2\x94\x82           \xe2\x94\x82   \xe2\x94\x82   \xe2\x94\x82           \xe2\x94\x82           \xe2\x94\x94\xe2\x94\x80\xe2\x94\x80 world: 1.0\n    \xe2\x94\x82           \xe2\x94\x82   \xe2\x94\x82   \xe2\x94\x82           \xe2\x94\x82               \xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80 ,": 0.3066892623901367\n    \xe2\x94\x82           \xe2\x94\x82   \xe2\x94\x82   \xe2\x94\x82           \xe2\x94\x82               \xe2\x94\x94\xe2\x94\x80\xe2\x94\x80 .: 0.6933107376098633\n    \xe2\x94\x82           \xe2\x94\x82   \xe2\x94\x82   \xe2\x94\x82           \xe2\x94\x94\xe2\x94\x80\xe2\x94\x80 the: 0.3571747839450836\n    \xe2\x94\x82           \xe2\x94\x82   \xe2\x94\x82   \xe2\x94\x94\xe2\x94\x80\xe2\x94\x80 happening: 0.07412238419055939\n    \xe2\x94\x82           \xe2\x94\x82   \xe2\x94\x94\xe2\x94\x80\xe2\x94\x80 so: 0.13510821759700775\n    \xe2\x94\x82           \xe2\x94\x94\xe2\x94\x80\xe2\x94\x80 will: 0.1756485253572464\n    \xe2\x94\x94\xe2\x94\x80\xe2\x94\x80 world: 0.7696937918663025\n'
s = codecs.decode(x,"utf-8")
print(s)


