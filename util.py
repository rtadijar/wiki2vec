import torch

import nltk

import io

def load_embeddings(fname, get_embeddings=True, get_w2i=False, get_i2w=False, skip_first_line=True):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    
    if skip_first_line:
        fin.readline()
    
    num_embeddings = 0
    
    word2idx = {}
    idx2word = {}

    embeddings = []

    for line in fin:
        line = line.rstrip().split(' ')
        
        if get_w2i:
            word2idx[line[0]] = num_embeddings
        if get_i2w:
            idx2word[num_embeddings] = line[0]
        if get_embeddings:
            embeddings.append([float(num) for num in line[1:]])

        num_embeddings += 1
        
    fin.close()
    
    return torch.FloatTensor(embeddings), word2idx, idx2word

stopwords = set(nltk.corpus.stopwords.words('english'))
meaningless_tokens = [',', '.', '\'' , '', '"', '-', '_', '–', '&', '\'\'', '""']

def parse_text(text, stopwords=[]): 
    
    
    tokens = [word for word in nltk.word_tokenize(text) if word.lower() not in meaningless_tokens and word.lower() not in stopwords]
    
    trans = str.maketrans('', '', '[]().,;:|`')
    
    tokens = [token.translate(trans) for token in tokens]
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

def get_page_summary(id):
    query_params = {
                    'action': 'query',
                    'prop': 'extracts',
                    'exintro': '',
                    'explaintext': '',
                    'pageids': id,
                    'format': 'json'
                   }
    
    headers = {
        'User-Agent': ua
    }
    
    r = requests.get(wiki_api, params=query_params, headers=headers).json()
    return parse_text(r['query']['pages'][str(id)]['extract'], stopwords=stopwords)