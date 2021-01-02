import requests

from client import get_random_page, wiki, wiki_api, ua
from mwclient.page import Page

from pathlib import Path
from collections import defaultdict

import pickle
from multiprocessing import Pool

import nltk
import nltk.corpus

import time

stopwords = set(nltk.corpus.stopwords.words('english'))

meaningless_tokens = [',', '.', '\'' , '', '"', '-', '_', '–', '&', '\'\'', '""']

def parse_text(text): 
        
    tokens = [word for word in nltk.word_tokenize(text) if word.lower() not in stopwords and word not in meaningless_tokens]
    
    trans = str.maketrans('', '', '[]().,;:|`')
    
    tokens = [token.translate(trans) for token in tokens]
    tokens = [token for token in tokens if token != '']
    
    return list(dict.fromkeys(tokens))

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
    return parse_text(r['query']['pages'][str(id)]['extract'])


class Article:

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

    def __init__(self, page, get_links=True, get_summary=True):
        self.title = page.name
        self.id = page.pageid

        self.links = []

        for link in page.links(0, generator=True):
            self.links.append(link.name)

        self.summary = get_page_summary(self.id)




def get_pages(id, n=1000):
    visit_counts = defaultdict(int)
    links = defaultdict(list)

    while n != 0:
        pg = get_random_page()
        article = Article(pg)
        
        # fix this stupidity
        if article not in links:
            links[article] = [x.name for x in pg.links(0, generator=True)]

        visit_counts[article] += 1

        n -= 1

    
    with open('data/articles/articles{}.dat'.format(id), 'wb') as output_file:
        to_pickle = {'counts': visit_counts, 'links': links}
        pickle.dump(to_pickle, output_file)
        
        

if __name__ == '__main__':
    num_processes = 12

    Path('data/articles').mkdir(parents=True, exist_ok=True)

    
    for i in range(100):
        print("started run {}".format(i+1))
        
        start = time.time()

        with Pool(processes=num_processes) as pool:
            pool.map(get_pages, range((i + 36) * num_processes, (i + 37) * num_processes), chunksize=1)

        end = time.time()
        print('iteration took {} seconds'.format(end - start))