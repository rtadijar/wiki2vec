# wiki2vec - a Wikipedia Article Embedding

wiki2vec is a machine learning model trained to create latent representations of Wikipedia articles. The latent representations retain notions of topical similarity,
and the cosine similarity between two titles can be used as a heuristic for the [Wikiracing/Wikipedia Game](https://en.wikipedia.org/wiki/Wikiracing) (get from article A to article B using exclusively in-article links).

## How it Works

The model takes as its input the title of a Wikipedia article. Each word in the title goes through an embedding layer after which follows a single transformer encoder<sup>1</sup>.
The transformed embeddings are added up using an additive attention layer, creating the representation of the given article. Subspace disagreement<sup>2</sup> is enforced in the
multi-head attention block to force different heads to attend to different semantic components of individual words.

Training is done using a dataset of Wikipedia links. Using negative sampling, the model is encouraged to lower the cosine distance between pages which link to each other,
and vice versa. In this sense wiki2vec can be seen as a graph embedding.

## Usage

A wikirace session can be run in the following manner:

`python wikirace.py --model model_path --heads num_heads --word2idx w2i_path [a] [b]`

`a` represents the starting point and `b` represents the destination. If left empty, both are initialized randomly (using Wikipedia's random article functionality).


###### <sup>1</sup> [Attention Is All You Need](https://arxiv.org/abs/1706.03762) </br> <sup>2</sup> [Multi-Head Attention with Disagreement Regularization](https://arxiv.org/abs/1810.10183)
