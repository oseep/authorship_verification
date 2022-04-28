import sys
sys.path.append("../")

import re
import json
import numpy as np
from tqdm import trange, tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from features import merge_entries
import pickle
from utills import chunker
import gc


PREPROCESSED_DATA_PATH = '../temp_data/reddit/preprocessed/'
TEMP_DATA_PATH = '../temp_data/reddit/topic_similarity/'

n_features = 10000 # number of most common words
n_topics = 150 # number of topics
max_df = 0.5 # maximum document frequency
min_df = 100 # minimum document frequency

chunk_sz = 10

ALLOWED_POS_TAGS = set(['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WP', 'WP$'])


def preprocess_for_topics(e):
    return [
        re.sub(r"[,.;@#?!&$]+\ *", " ", e['tokens'][i]).strip() 
    for i in range(len(e['tokens'])) if e['pos_tags'][i] in ALLOWED_POS_TAGS]

def noop(x):
    return x


if __name__ == "__main__":
    print('Prepping data...', flush=True)
    docs = []
    with open(PREPROCESSED_DATA_PATH + 'train.jsonl', 'r') as f:
        for l in tqdm(f):
            if np.random.rand() < 0.05:
                continue
            d = json.loads(l)
            # docs.append(preprocess_for_topics(merge_entries(d['data'])))
            prepped = [preprocess_for_topics(merge_entries(c)) for c in chunker(d['data'], chunk_sz)]
            docs.extend([c for c in prepped if len(c) > 0])
    
    print('Fitting transformer...', flush=True)       
    transformer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=n_features, analyzer=noop)
    X = transformer.fit_transform(docs)
    
    print('Fitting NMF...', flush=True)   
    nmf = NMF(n_components=n_topics, verbose=1, max_iter=100).fit(X)
    with open(TEMP_DATA_PATH + 'reddit_topic_model_chunked3.p', 'wb') as f:
        pickle.dump((nmf, transformer), f)
    