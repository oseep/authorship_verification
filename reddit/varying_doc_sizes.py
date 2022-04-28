import sys
import os
sys.path.append("../")
import warnings
warnings.filterwarnings("ignore")

import pickle
import glob
from tqdm.notebook import trange, tqdm
import json
import re
import pandas as pd
import numpy as np
from collections import defaultdict

from utills import chunker, cartesian_product, get_num_chunks
from train_utils import generate_unique_pairs, get_random_author_excluding, generate_doc_pairs_no_chunking, fit_transformers_no_chunking, vectorize_no_chunking


from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc

from sklearn.utils.fixes import loguniform
from sklearn.model_selection import RandomizedSearchCV

# BASE_PATH = '../data/reddit_2/'
'''
COMPUTED_DATA_PATH = '../temp_data/reddit/preprocessed/'
EXPERIMENT_DATA_PATH = '../temp_data/reddit/doc_size_experiments/'
'''
COMPUTED_DATA_PATH = '/scratch/jnw301/av_public/temp_data/reddit/preprocessed/'
EXPERIMENT_DATA_PATH = '/scratch/jnw301/av_public/temp_data/reddit/doc_size_experiments/'

MIN_COMMENTS = 80
SIZES = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80]

if __name__ == "__main__":
    instance_id = int(sys.argv[1])
    print('Instance ID for this machine:', instance_id, flush=True)
    with open(COMPUTED_DATA_PATH + 'metadata.p', 'rb') as f:
        (train_files, test_files, min_count, author_mapping) = pickle.load(f)
    
    num_comments = SIZES[instance_id]
    EXPERIMENT_DATA_PATH = EXPERIMENT_DATA_PATH + 'size_' + str(num_comments) + '/'
    if not os.path.exists(EXPERIMENT_DATA_PATH):
        os.makedirs(EXPERIMENT_DATA_PATH)
        
    print('Fitting transformers...', flush=True)
    transformer, scaler, secondary_scaler = fit_transformers_no_chunking(
        COMPUTED_DATA_PATH + 'train.jsonl', 
        author_mapping, 
        sample_fraction=0.1, 
        min_comments=MIN_COMMENTS,
        max_comments=num_comments
    )
    
    with open(EXPERIMENT_DATA_PATH + '.p', 'wb') as f:
        pickle.dump((transformer, scaler, secondary_scaler ), f)
        
    print('Vectorizing Train Set...', flush=True)
    XX_train, author_to_doc_idx, author_subreddit, x_shape = vectorize_no_chunking(
        preprocessed_path = COMPUTED_DATA_PATH + 'train.jsonl', 
        vectorized_x_path = EXPERIMENT_DATA_PATH + 'XX_train.npy', 
        transformer=transformer,
        scaler=scaler,
        max_comments=num_comments,
        min_comments=MIN_COMMENTS
    )
    
    
    print('Vectorizing Test Set...', flush=True)
    XX_test, author_to_doc_idx_test, author_subreddit_test, x_shape_test = vectorize_no_chunking(
        preprocessed_path = COMPUTED_DATA_PATH + 'test.jsonl', 
        vectorized_x_path = EXPERIMENT_DATA_PATH + 'XX_test.npy', 
        transformer=transformer,
        scaler=scaler,
        max_comments=num_comments,
        min_comments=MIN_COMMENTS        
    )
    
    with open(EXPERIMENT_DATA_PATH + 'training_meta_data.p', 'wb') as f:
        pickle.dump((author_to_doc_idx, author_to_doc_idx_test, author_subreddit, author_subreddit_test, x_shape, x_shape_test), f)
      
    print('Generating Pairs...', flush=True)
    author_mapping = defaultdict(set)
    author_to_root = {}
    for y in author_to_doc_idx.keys():
        u = re.search(r'(.*)_[A-Z]+$', y).group(1)
        author_mapping[u].add(y)
        author_to_root[y] = u

    subreddit_to_author = defaultdict(list)
    for k, v in author_subreddit.items():
        subreddit_to_author[v].append(k)
        
    X_idxs_train, Y_train = generate_doc_pairs_no_chunking(author_mapping, subreddit_to_author, author_to_root, author_to_doc_idx, author_subreddit)
    
    
    author_mapping_test = defaultdict(set)
    author_to_root_test = {}
    for y in author_to_doc_idx_test.keys():
        u = re.search(r'(.*)_[A-Z]+$', y).group(1)
        author_mapping_test[u].add(y)
        author_to_root_test[y] = u

    subreddit_to_author_test = defaultdict(list)
    for k, v in author_subreddit_test.items():
        subreddit_to_author_test[v].append(k)

    X_idxs_test, Y_test = generate_doc_pairs_no_chunking(author_mapping_test, subreddit_to_author_test, author_to_root_test, author_to_doc_idx_test, author_subreddit_test)
    
    print('Train Model...', flush=True)
    batch_sz = 50000
    clf = SGDClassifier(loss='log', alpha=0.01)
    x_test_diff_sample = secondary_scaler.transform(np.abs(XX_test[X_idxs_test[:batch_sz, 0]] - XX_test[X_idxs_test[:batch_sz, 1]]))
    y_test_sample = Y_test[:batch_sz]
    aucs = []
    for i in range(50):
        for idxs in chunker(np.arange(len(X_idxs_train)), batch_sz):
            x_diff = secondary_scaler.transform(np.abs(XX_train[X_idxs_train[idxs, 0]] - XX_train[X_idxs_train[idxs, 1]]))
            x_diff[np.isnan(x_diff)]=0
            y = Y_train[idxs]
            clf.partial_fit(x_diff, y, classes=[0, 1])

            probs = clf.predict_proba(x_test_diff_sample)[:, 1]

            fpr, tpr, thresh = roc_curve(y_test_sample, probs)
            roc_auc = auc(fpr, tpr)
            print('AUC:', roc_auc)
        print('~'*20, 'Epoch: ', i)
        aucs.append(roc_auc)
        
        
    x_diff = secondary_scaler.transform(np.abs(XX_test[X_idxs_test[:, 0]] - XX_test[X_idxs_test[:, 1]]))
    x_diff[np.isnan(x_diff)]=0

    probs = clf.predict_proba(x_diff)[:, 1]
    fpr, tpr, thresh = roc_curve(Y_test, probs)
    roc_auc = auc(fpr, tpr)
    print('AUC:', roc_auc)
        
    with open(EXPERIMENT_DATA_PATH + 'model.p', 'wb') as f:
        pickle.dump((clf, transformer, scaler, secondary_scaler, aucs, probs, Y_test, auc), f)