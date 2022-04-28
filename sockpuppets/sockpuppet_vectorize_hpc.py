import sys
sys.path.append("../")

import os
import pickle
import json
from tqdm import trange, tqdm
import numpy as np
import pandas as pd
from features import prepare_entry, get_transformer, merge_entries
from utills import chunker, compress_fraction, cartesian_product
from scipy.stats import entropy
from config import known_bots




USERS_TO_REMOVE = set([u.lower() for u in known_bots])

#DATA_DIR = '../data/'
#TEMP_DIR = '../temp_data/'

DATA_DIR = '/scratch/jnw301/av_public/data/'
TEMP_DIR = '/scratch/jnw301/av_public/temp_data/'

DATA_PATH = DATA_DIR + 'r_gaming/comments.jsonl'
TEMP_DATA_PATH = TEMP_DIR + 'sockpuppets/r_gaming/'


MULTIDOC_MODEL_PATH = TEMP_DIR + 'reddit/multidoc_20/model_20.p'
SINGLEDOC_MODEL_PATH = TEMP_DIR + 'reddit/unchunked/model.p'


def preprocess(comments, chunk_sz):
    return [prepare_entry('\n'.join(c), mode='accurate', tokenizer='casual') for c in chunker(comments, chunk_sz)]

def user_text_repitition_score(user_comments):
    text = '\n'.join(user_comments)
    return compress_fraction(text)

def keep_user(u, author_comments):
    if u.lower() in USERS_TO_REMOVE:
        return False
    if user_text_repitition_score(author_comments) < 0.4:
        return False
    return True


if __name__ == "__main__":
    tqdm.pandas()
    
    print('Loading model...', flush=True)
    chunk_sz = 20
    with open(MULTIDOC_MODEL_PATH, 'rb') as f:
        (clf, transformer, scaler, secondary_scaler, _) = pickle.load(f)

    with open(SINGLEDOC_MODEL_PATH, 'rb') as f:
        (clf_nc, transformer_nc, scaler_nc, secondary_scaler_nc, _) = pickle.load(f)
        
    print('Loading data...', flush=True)    
    with open(DATA_PATH, 'r') as f:
        data = []
        for l in f:
            data.append(json.loads(l))
            
    df = pd.DataFrame.from_dict(data)
    del data
    print('Data Length:', len(df))
    # Select comments from users who has at least 100 comments
    author_comment_counts = df.groupby('author').count()['body'].to_frame()
    author_comment_counts.columns = ['count']
    selected_authors = author_comment_counts.loc[author_comment_counts['count'] > 100].index.values
    df = df.loc[[r['author'] in selected_authors for i, r in df.iterrows()]]

    # Collect each author's comments in a list
    print('Preprocessing data...', flush=True)    
    author_comments = df.groupby('author')['body'].apply(list)
    author_comments = author_comments.to_frame()
    keep_mask = np.array([keep_user(u, r['body']) for u, r in tqdm(author_comments.iterrows(), total=len(author_comments))])
    
    print('Num removed spam users:', keep_mask.sum(), keep_mask.mean())
    author_comments = author_comments.loc[keep_mask]
    author_comments['body'] = author_comments['body'].progress_apply(lambda c: preprocess(c, chunk_sz))
    
    print('Vectorize...')
    # We will create one memory mapped matrix to store the features or all the chunks for all the authors.
    # author_bounds will keep track of the start and end indices of each authors' feature matrix 

    total_chunks = 0
    author_bounds = {}
    author_to_idx_nc = {}
    i = 0
    for user, chunks in author_comments['body'].iteritems():
        author_bounds[user] = (total_chunks, total_chunks + len(chunks))
        author_to_idx_nc[user] = i
        i +=1
        total_chunks += len(chunks)

    x_shape = (total_chunks, len(transformer.get_feature_names()))
    

        
    x_shape = (total_chunks, len(transformer.get_feature_names()))
    x_shape_nc = (len(author_comments), len(transformer_nc.get_feature_names()))
    XX = np.memmap(TEMP_DATA_PATH + 'XX.npy', dtype='float32', mode='w+', shape=x_shape)
    XX_nc = np.memmap(TEMP_DATA_PATH + 'XX_nc.npy', dtype='float32', mode='w+', shape=x_shape_nc)
    
 
    
    i = 0
    author_order = []
    for user, chunks in tqdm(author_comments['body'].iteritems(), total=len(author_comments)):
        s, e = author_bounds[user]
        assert author_to_idx_nc[user] == i
        XX[np.arange(s, e), :] = scaler.transform(transformer.transform(chunks).todense())
        XX_nc[i, :] = scaler_nc.transform(transformer_nc.transform([merge_entries(chunks)]).todense())[0, :]
        i += 1
        author_order.append(user)
        
    with open(TEMP_DATA_PATH + 'vectorizing_parameters.p', 'wb') as f:
        pickle.dump((author_order, total_chunks, author_bounds, author_to_idx_nc, x_shape, x_shape_nc), f)   