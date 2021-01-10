from client import get_random_page
from util import parse_text, encode_seq

from torch.nn.functional import cosine_similarity as similarity

def race(a, b, model, word2idx, device):
    a = a.resolve_redirect()
    
    print('starting at page `{}`'.format(a.name))
    print('trying to reach page `{}`'.format(b.name))

    goal_seq = encode_seq(parse_text(b.name))
    
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
                
            link_seq = encode_seq(parse_text(link.name))
            
            if len(link_seq) == 0:
                continue
            else:
                link_seq = link_seq.unsqueeze(0).to(device)
            
            link_embedding = F.normalize(model(link_seq))

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