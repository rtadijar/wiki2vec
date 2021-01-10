import torch
import string

trans = str.maketrans('(),_-/', '      ','\\"')

def parse_text(text, stopwords=[]): 
    tokens = text.translate(trans).split()
    tokens = [token for token in tokens if token != '']
    
    return list(dict.fromkeys(tokens))

def encode_seq(seq, word2idx, tolerate_miss=True):
    res = []
    
    for token in seq:
        if token in word2idx:
            res.append(word2idx[token])
        elif not tolerate_miss:
                return []
    
    return torch.LongTensor(res)