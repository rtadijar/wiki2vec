from util import parse_text, encode_seq
from torch.nn.functional import cosine_similarity as similarity

def race(a, b, model, word2idx, device):
    a = a.resolve_redirect()
    
    print('starting at page `{}`'.format(a.name))
    print('trying to reach page `{}`'.format(b.name))

    goal_seq = encode_seq(parse_text(b.name), word2idx)
    
    if goal_seq == []:
        print('error: goal tokens unknown')
        return
    
    goal_seq = model(goal_seq.unsqueeze(0).to(device))
    visited_name = set()
    visited_id = set()
    
    while(a.name != b.name):
        best_link = None
        best_sim = -1e9
        
        for link in a.links(0):
            
            
            if link.name in visited_name or link.pageid in visited_id:
                continue
                
            link_seq = encode_seq(parse_text(link.name), word2idx)
            
            if len(link_seq) == 0:
                continue
            else:
                link_seq = link_seq.unsqueeze(0).to(device)
            
            link_embedding = model(link_seq)

            sim = similarity(link_embedding, goal_seq)

            if link.pageid is not None and sim > best_sim:
                best_sim = sim
                best_link = link
        
        if best_link is None:
            print('cannot find candidate link')
            break
        else:
            print('{} -> {}'.format(a.name, best_link.name))
        
            visited_name.add(best_link.name)
            visited_id.add(best_link.pageid)
            
            best_link = best_link.resolve_redirect()                
            
            visited_name.add(best_link.name)
            visited_id.add(best_link.pageid)

            a = best_link


import argparse
from client import *


parser = argparse.ArgumentParser(description='Race from A to B with a wiki2vec model!')

parser.add_argument('--model', metavar='model_path', help='path to the wiki2vec model state dicti', action='store', required=True)
parser.add_argument('--heads', metavar='num_heads', help='number of heads in the multi-head attention module', action='store', required=True, type=int)
parser.add_argument('--word2idx', metavar='w2i_path', help='path to the word/token to index mapping', action='store', required=True)

parser.add_argument('a', nargs='?', help='name of the starting point for the race (random if unspecified)', default=get_random_page())
parser.add_argument('b', nargs='?', help='name of the goal page for the race (random if unspecified)', default=get_random_page())


if __name__ == "__main__":
    import pickle
    import torch

    from model import TitleEmbedding


    args = parser.parse_args()

    with open('{}'.format(args.word2idx), 'rb') as _if:
        word2idx = pickle.load(_if)

    state_dict = torch.load('{}'.format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_embeddings = state_dict['embeddings.weight'].shape[0]
    d_model = state_dict['embeddings.weight'].shape[1]
    num_heads = args.heads

    model = TitleEmbedding(num_embeddings, d_model, num_heads, int(d_model / num_heads), int(d_model / num_heads), track_agreement=True).to(device)

    model.load_state_dict(state_dict)

    race(args.a, args.b, model, word2idx, device)