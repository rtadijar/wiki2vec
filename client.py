import mwclient

from mwclient.page import Page
from mwclient import Site

ua = 'WikiRacer/1.0 ()'

wiki = Site('en.wikipedia.org', clients_useragent=ua)
wiki_api = 'http://en.wikipedia.org/w/api.php'

generator = wiki.random(0, 1)

def get_random_page():
    for rand in generator:
        return Page(wiki, rand['title'])