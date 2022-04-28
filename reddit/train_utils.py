import sys
sys.path.append("../")

import re
import numpy as np
import json
from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from features import merge_entries, get_transformer
from utills import chunker, cartesian_product, get_num_chunks
import random

def generate_unique_pairs(arr):
    ''' 
    Returns an array of tuples with all possible unique pairs
    made from the items in the given array.
    '''
    result = []
    for i in range(len(arr)):
        for j in range(i):
            result.append((arr[i], arr[j]))
    return result

def get_random_author_excluding(exclue_author, exclude_topic, authors, author_to_root, author_topics):
    '''
    Returns a random author from a list of given authors,
    excluding the given author and topic
    '''
    author_b = np.random.choice(authors)
    while author_to_root[author_b] == author_to_root[exclue_author] or exclude_topic == author_topics[author_b]:
        author_b = np.random.choice(authors)
    return author_b



def generate_doc_pairs(author_mapping, subreddit_to_author, author_to_root, author_bounds, author_subreddit):
    # Positive pairs - Same author, different subreddit
    X_idxs_pos = []
    for authors in tqdm(author_mapping.values()):
        authors = list(authors)
        for author_a, author_b in generate_unique_pairs(authors):
            start_a, end_a = author_bounds[author_a]
            start_b, end_b = author_bounds[author_b]
            idxs = cartesian_product(range(start_a, end_a), range(start_b, end_b))
            X_idxs_pos.extend(idxs)



    # Negative pairs, same subreddit 
    X_idxs_neg_same_topic = []
    for authors in tqdm(author_mapping.values()):
        for author_a in list(authors):
            possible_authors = subreddit_to_author[author_subreddit[author_a]]
            if len(possible_authors) < 10:
                continue
            author_b = np.random.choice(possible_authors)
            while author_to_root[author_b] == author_to_root[author_a]:
                author_b = np.random.choice(possible_authors)
            start_a, end_a = author_bounds[author_a]
            start_b, end_b = author_bounds[author_b]
            idxs = cartesian_product(range(start_a, end_a), range(start_b, end_b))
            X_idxs_neg_same_topic.extend(idxs)

    # Negative pairs, different subreddit       
    X_idxs_neg_diff_topic = []
    for authors in tqdm(author_mapping.values()):
        authors = list(authors)
        for author_a in authors:
            author_b = get_random_author_excluding(author_a, author_subreddit[author_a], list(author_bounds.keys()), author_to_root, author_subreddit)
            start_a, end_a = author_bounds[author_a]
            start_b, end_b = author_bounds[author_b]
            idxs = cartesian_product(range(start_a, end_a), range(start_b, end_b))
            X_idxs_neg_diff_topic.extend(idxs) 

    X_idxs_pos = np.array(X_idxs_pos)
    X_idxs_neg_diff_topic = np.array(X_idxs_neg_diff_topic)
    X_idxs_neg_same_topic = np.array(X_idxs_neg_same_topic)

    p = np.random.choice(np.arange(len(X_idxs_neg_diff_topic)), size=min(len(X_idxs_neg_diff_topic), int(len(X_idxs_pos) * 0.3)), replace=False)
    X_idxs_neg_diff_topic = X_idxs_neg_diff_topic[p]


    p = np.random.choice(np.arange(len(X_idxs_neg_same_topic)), size=min(len(X_idxs_neg_same_topic), int(len(X_idxs_pos) * 0.7)), replace=False)
    X_idxs_neg_same_topic = X_idxs_neg_same_topic[p]

    X_idxs = np.concatenate([X_idxs_pos, X_idxs_neg_diff_topic, X_idxs_neg_same_topic])
    labels = [1] * len(X_idxs_pos) + [0] * len(X_idxs_neg_diff_topic) + [0] * len(X_idxs_neg_same_topic)
    labels = np.array(labels)

    p = np.random.permutation(len(labels))
    return X_idxs[p], labels[p]


def generate_doc_pairs_no_chunking(author_mapping, subreddit_to_author, author_to_root, author_to_doc_idx, author_subreddit, return_all=False):
    print('new2')
    # Positive pairs - Same author, different subreddit
    X_idxs_pos = []
    for authors in tqdm(author_mapping.values()):
        authors = list(authors)
        for author_a, author_b in generate_unique_pairs(authors):
            X_idxs_pos.append((author_to_doc_idx[author_a], author_to_doc_idx[author_b]))

    # Negative pairs, same subreddit 
    X_idxs_neg_same_topic = []
    for authors in tqdm(author_mapping.values()):
        for author_a in list(authors):
            possible_authors = subreddit_to_author[author_subreddit[author_a]]
            if len(possible_authors) < 5:
                continue
            author_b = np.random.choice(possible_authors)
            while author_to_root[author_b] == author_to_root[author_a]:
                author_b = np.random.choice(possible_authors)
            X_idxs_neg_same_topic.append((author_to_doc_idx[author_a], author_to_doc_idx[author_b]))

    # Negative pairs, different subreddit       
    X_idxs_neg_diff_topic = []
    for authors in tqdm(author_mapping.values()):
        authors = list(authors)
        for author_a in authors:
            author_b = get_random_author_excluding(author_a, author_subreddit[author_a], list(author_to_doc_idx.keys()), author_to_root, author_subreddit)
            X_idxs_neg_diff_topic.append((author_to_doc_idx[author_a], author_to_doc_idx[author_b]))

    X_idxs_pos = np.array(X_idxs_pos)
    X_idxs_neg_diff_topic = np.array(X_idxs_neg_diff_topic)
    X_idxs_neg_same_topic = np.array(X_idxs_neg_same_topic)
    print(len(X_idxs_pos), len(X_idxs_neg_diff_topic), len(X_idxs_neg_same_topic))
    if return_all:
        return X_idxs_pos, X_idxs_neg_diff_topic, X_idxs_neg_same_topic

    p = np.random.choice(np.arange(len(X_idxs_neg_diff_topic)), size=min(len(X_idxs_neg_diff_topic), int(len(X_idxs_pos) * 0.1)), replace=False)
    X_idxs_neg_diff_topic = X_idxs_neg_diff_topic[p]


    p = np.random.choice(np.arange(len(X_idxs_neg_same_topic)), size=min(len(X_idxs_neg_same_topic), int(len(X_idxs_pos) * 0.9)), replace=False)
    X_idxs_neg_same_topic = X_idxs_neg_same_topic[p]

    X_idxs = np.concatenate([X_idxs_pos, X_idxs_neg_diff_topic, X_idxs_neg_same_topic])
    labels = [1] * len(X_idxs_pos) + [0] * len(X_idxs_neg_diff_topic) + [0] * len(X_idxs_neg_same_topic)
    labels = np.array(labels)
    print(len(X_idxs_pos), len(X_idxs_neg_diff_topic), len(X_idxs_neg_same_topic))
    p = np.random.permutation(len(labels))
    return X_idxs[p], labels[p]

def generate_same_author_pairs(author_mapping, subreddit_to_author, author_to_root, author_subreddit):
    # Positive pairs - Same author, different subreddit
    for authors in author_mapping.values():
        authors = list(authors)
        pairs = generate_unique_pairs(authors)
        for p in pairs:
            yield p
            
def generate_diff_author_same_topic_pairs(author_mapping, subreddit_to_author, author_to_root, author_subreddit):
    # Negative pairs, same subreddit 
    shuffled_author_map_values = list(author_mapping.values())
    random.shuffle(shuffled_author_map_values)
    for authors in shuffled_author_map_values:
        for author_a in list(authors):
            possible_authors = subreddit_to_author[author_subreddit[author_a]]
            
            if len(possible_authors) < 5:
                continue
            random.shuffle(possible_authors)
#            author_b = np.random.choice(possible_authors)
#            while author_to_root[author_b] == author_to_root[author_a]:
#                author_b = np.random.choice(possible_authors)
#            yield (author_a, author_b)
            for author_b in possible_authors[:100]:
                if author_to_root[author_b] == author_to_root[author_a]:
                    continue
                yield (author_a, author_b)
                
                
def generate_diff_author_diff_topic_pairs(author_mapping, subreddit_to_author, author_to_root, author_subreddit):
    shuffled_author_map_values = list(author_mapping.values())
    random.shuffle(shuffled_author_map_values)
    
    all_authors = list(author_to_root.keys())
    for authors in shuffled_author_map_values:
        authors = list(authors)
        random.shuffle(all_authors)
        for author_a in authors:
            for author_b in all_authors[:100]:
                if author_to_root[author_b] == author_to_root[author_a] or author_subreddit[author_a] == author_subreddit[author_b]:
                    continue
                yield (author_a, author_b)

def fit_transformers(preprocessed_data_path, author_mapping, chunk_sz, sample_fraction=0.05, max_comments=None, exclude_users=None):
    sampled_authors = []
    for v in author_mapping.values():
        if np.random.rand() < sample_fraction:
            if exclude_users is not None:
                sampled_authors.extend([u for u in v if u not in exclude_users])
            else:
                sampled_authors.extend(v)
    sampled_authors = set(sampled_authors)
    print('Sampled:', len(sampled_authors))
    print('Reading preprocessed data...')
    author_subreddit = {}
    author_bounds = {}
    X = []
    Y = []
    with open(preprocessed_data_path, 'r') as f:
        for l in tqdm(f):
            d = json.loads(l)
            if d['username'] in sampled_authors and len(d['data']) > chunk_sz:
                if max_comments is None:
                    doc = d['data']
                else:
                    doc = d['data'][:max_comments]
                chunks = [merge_entries(c) for c in chunker(doc, chunk_sz)]
                author_bounds[d['username']] = (len(X), len(X) + len(chunks))
                author_subreddit[d['username']] = d['subreddit']
                X.extend(chunks)
                Y.extend([d['username']] * len(chunks))

    print('Fitting transformer')
    transformer = get_transformer()
    scaler = StandardScaler()

    X = transformer.fit_transform(X).todense()
    X = scaler.fit_transform(X)

    author_mapping_sampled = defaultdict(set)
    author_to_root = {}
    for y in Y:
        u = re.search(r'(.*)_[A-Z]+$', y).group(1)
        author_mapping_sampled[u].add(y)
        author_to_root[y] = u

    subreddit_to_author = defaultdict(list)
    for k, v in author_subreddit.items():
        subreddit_to_author[v].append(k)


    print('Generating pairs')
    X_idxs, labels = generate_doc_pairs(author_mapping_sampled, subreddit_to_author, author_to_root, author_bounds, author_subreddit)

    max_sample_size = 100000
    secondary_scaler = StandardScaler()
    X_diff = np.abs(X[X_idxs[:max_sample_size, 0], :] - X[X_idxs[:max_sample_size, 1], :])

    X_diff = secondary_scaler.fit_transform(X_diff)
    return transformer, scaler, secondary_scaler

'''
def fit_transformers_no_chunking(preprocessed_data_path, selected_author_idxs, sampling_frac=0.1):

    author_to_doc_idx = {}
    author_subreddit = {}
    X = []
    Y = []

    with open(preprocessed_data_path, 'r') as f:
        i = 0
        for l in tqdm(f):
            if i not in selected_author_idxs:
                i += 1
                continue
                
            if np.random.rand() > sampling_frac:
                i += 1
                continue
            d = json.loads(l)
            doc = merge_entries(d['data'])
            author_to_doc_idx[d['username']] = len(X)
            author_subreddit[d['username']] = d['subreddit']
            X.append(doc)
            Y.append(d['username'])
            i += 1
            

    print('Fitting transformer')
    transformer = get_transformer()
    scaler = StandardScaler()

    X = transformer.fit_transform(X).todense()
    X = scaler.fit_transform(X)

    
    return transformer, scaler
'''
def fit_transformers_no_chunking(preprocessed_data_path, author_mapping, sample_fraction=0.05, min_comments=None, max_comments=None, exclude_users=None):
    sampled_authors = []
    for v in author_mapping.values():
        if np.random.rand() < sample_fraction:
            if exclude_users is not None:
                sampled_authors.extend([u for u in v if u not in exclude_users])
            else:
                sampled_authors.extend(v)
    sampled_authors = set(sampled_authors)
    print('Sampled:', len(sampled_authors))
    print('Reading preprocessed data...')
    author_to_doc_idx = {}
    author_subreddit = {}
    X = []
    Y = []

    with open(preprocessed_data_path, 'r') as f:
        for l in tqdm(f):
            d = json.loads(l)
            if d['username'] in sampled_authors:
                if min_comments is not None and len(d['data']) < min_comments:
                    continue
                if max_comments is None:
                    doc = merge_entries(d['data']) 
                else:
                    doc = merge_entries(d['data'][:max_comments]) 
                author_to_doc_idx[d['username']] = len(X)
                author_subreddit[d['username']] = d['subreddit']
                X.append(doc)
                Y.append(d['username'])

    print('Fitting transformer')
    transformer = get_transformer()
    scaler = StandardScaler()

    X = transformer.fit_transform(X).todense()
    X = scaler.fit_transform(X)

    author_mapping_sampled = defaultdict(set)
    author_to_root = {}
    for y in Y:
        u = re.search(r'(.*)_[A-Z]+$', y).group(1)
        author_mapping_sampled[u].add(y)
        author_to_root[y] = u

    subreddit_to_author = defaultdict(list)
    for k, v in author_subreddit.items():
        subreddit_to_author[v].append(k)


    print('Generating pairs')
    X_idxs, labels = generate_doc_pairs_no_chunking(author_mapping_sampled, subreddit_to_author, author_to_root, author_to_doc_idx, author_subreddit)

    max_sample_size = 100000
    secondary_scaler = StandardScaler()
    X_diff = np.abs(X[X_idxs[:max_sample_size, 0], :] - X[X_idxs[:max_sample_size, 1], :])

    X_diff = secondary_scaler.fit_transform(X_diff)
    return transformer, scaler, secondary_scaler

def vectorize(preprocessed_path, vectorized_x_path, transformer, scaler, chunk_sz, max_comments=None, exclude_users=None):

    author_subreddit = {}
    author_bounds = {}
    total_recs = 0

    print('Precomputing record size...')
    with open(preprocessed_path, 'r') as f:
        for l in tqdm(f):
            d = json.loads(l)
            if exclude_users is not None and d['username'] in exclude_users:
                continue
            if len(d['data']) < chunk_sz:
                continue
            if max_comments is None:
                doc = d['data']
            else:
                doc = d['data'][:max_comments]
            num_chunks = get_num_chunks(doc, chunk_sz)
            author_bounds[d['username']] = (total_recs, total_recs + num_chunks)
            author_subreddit[d['username']] = d['subreddit']
            total_recs += num_chunks

    x_shape = (total_recs, len(transformer.get_feature_names()))
    XX = np.memmap(vectorized_x_path, dtype='float32', mode='w+', shape=x_shape)
    i = 0        
    with open(preprocessed_path, 'r') as f:
        for l in tqdm(f, total=len(author_bounds)):
            d = json.loads(l)
            if exclude_users is not None and d['username'] in exclude_users:
                continue
            if len(d['data']) < chunk_sz:
                continue
            if max_comments is None:
                doc = d['data']
            else:
                doc = d['data'][:max_comments]
            chunks = [merge_entries(c) for c in chunker(doc, chunk_sz)]
            XX[i:(i + len(chunks))] = scaler.transform(transformer.transform(chunks).todense())

            assert author_bounds[d['username']] == (i, i + len(chunks))
            i += len(chunks)
    return XX, author_bounds, author_subreddit, x_shape
'''
def vectorize_no_chunking(preprocessed_path, vectorized_x_path, transformer, scaler, selected_author_idxs):
    print('new')
    author_to_doc_idx = {}
    total_recs = len(selected_author_idxs)

    x_shape = (total_recs, len(transformer.get_feature_names()))
    XX = np.memmap(vectorized_x_path, dtype='float32', mode='w+', shape=x_shape)
    i = 0
    j = 0
    with open(preprocessed_path, 'r') as f:
        for l in tqdm(f):
            if i not in selected_author_idxs:
                i += 1
                continue

            d = json.loads(l)
            doc = merge_entries(d['data'])
            XX[j] = scaler.transform(transformer.transform([doc]).todense())[0, :]

            author_to_doc_idx[d['username']] = j
            i += 1
            j += 1
#     assert i == total_recs
    return XX, author_to_doc_idx, x_shape
'''
def vectorize_no_chunking(preprocessed_path, vectorized_x_path, transformer, scaler, min_comments=None, max_comments=None, exclude_users=None):

    author_subreddit = {}
    author_to_doc_idx = {}
    total_recs = 0
    excluded = []
    print('Precomputing record size...')
    num_users_read = 0
    with open(preprocessed_path, 'r') as f:
        for l in tqdm(f):
            d = json.loads(l)
            num_users_read += 1
            if exclude_users is not None and d['username'] in exclude_users:
                excluded.append(d['username'])
                continue
            if min_comments is not None and len(d['data']) < min_comments:
                continue
            author_to_doc_idx[d['username']] = total_recs
            author_subreddit[d['username']] = d['subreddit']
            total_recs += 1

    x_shape = (total_recs, len(transformer.get_feature_names()))
    XX = np.memmap(vectorized_x_path, dtype='float32', mode='w+', shape=x_shape)
    i = 0        
    with open(preprocessed_path, 'r') as f:
        for l in tqdm(f, total=len(author_to_doc_idx)):
            d = json.loads(l)
            if exclude_users is not None and d['username'] in exclude_users:
                continue
            if min_comments is not None and len(d['data']) < min_comments:
                continue
            if max_comments is None:
                doc = merge_entries(d['data']) 
            else:
                doc = merge_entries(d['data'][:max_comments]) 
            XX[i] = scaler.transform(transformer.transform([doc]).todense())[0, :]

            assert author_to_doc_idx[d['username']] == i
            i += 1
    return XX, author_to_doc_idx, author_subreddit, x_shape, excluded, num_users_read