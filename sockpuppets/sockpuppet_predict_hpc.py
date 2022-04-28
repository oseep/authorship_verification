import sys
sys.path.append("../")

import os
import pickle
import json
from tqdm import trange, tqdm
import numpy as np
import pandas as pd
from features import prepare_entry, get_transformer, merge_entries
from utills import chunker, cartesian_product


#DATA_DIR = '../data/'
#TEMP_DIR = '../temp_data/'

DATA_DIR = '/scratch/jnw301/av_public/data/'
TEMP_DIR = '/scratch/jnw301/av_public/temp_data/'


#DATA_PATH = DATA_DIR + 'gamestop/comments.jsonl'
#TEMP_DATA_PATH = TEMP_DIR + 'sockpuppets/gamestop/'
#DATA_PATH = DATA_DIR + 'r_funny/comments.jsonl'
#TEMP_DATA_PATH = TEMP_DIR + 'sockpuppets/r_funny/'
DATA_PATH = DATA_DIR + 'r_gaming/comments.jsonl'
TEMP_DATA_PATH = TEMP_DIR + 'sockpuppets/r_gaming/'

MULTIDOC_MODEL_PATH = TEMP_DIR + 'reddit/multidoc_20/model_20.p'
SINGLEDOC_MODEL_PATH = TEMP_DIR + 'reddit/unchunked/model.p'


MAX_INSTANCES = 10
MAX_CHUNKS = 60


if __name__ == "__main__":
    instance_id = int(sys.argv[1])
    progress_file_is_even = True
    print('Instance ID for this machine:', instance_id)
    
    
    print('Loading model...', flush=True)
    chunk_sz = 20
    with open(MULTIDOC_MODEL_PATH, 'rb') as f:
        (clf, transformer, scaler, secondary_scaler, _) = pickle.load(f)

    with open(SINGLEDOC_MODEL_PATH, 'rb') as f:
        (clf_nc, transformer_nc, scaler_nc, secondary_scaler_nc, _) = pickle.load(f)
        
    print('Loading vectorized data...', flush=True)    
    
    with open(TEMP_DATA_PATH + 'vectorizing_parameters.p', 'rb') as f:
        selected_authors, total_chunks, author_bounds, author_to_idx_nc, x_shape, x_shape_nc = pickle.load(f)
        
    XX = np.memmap(TEMP_DATA_PATH + 'XX.npy', dtype='float32', mode='r', shape=x_shape)
    XX_nc = np.memmap(TEMP_DATA_PATH + 'XX_nc.npy', dtype='float32', mode='r', shape=x_shape_nc)
    


    author_idxs = []
    for i in range(len(selected_authors)):
        for j in range(0, i):
            author_idxs.append((i, j))
            
    print('Total number of predictions to be run:', len(author_idxs))
    job_sz = len(author_idxs) // MAX_INSTANCES
    author_idxs = author_idxs [instance_id * job_sz: (instance_id + 1) * job_sz]
    print('Number of predictions to be run on this machine:', len(author_idxs))
    '''
    try:
        with open(TEMP_DATA_PATH + 'predict_results_instance_' + str(instance_id) + '.p', 'rb') as f:
            (
                probs_nc,
                inter_probs_mean,
                inter_probs_std,
                intraA_probs_mean,
                intraA_probs_std,
                intraB_probs_mean,
                intraB_probs_std,
                pred_lengths,
                _, 
                user_pairs
            ) = pickle.load(f)
        print('Already completed:', len(user_pairs), flush=True)    
        author_idxs_remaining = author_idxs[len(user_pairs):]
    except:
        print('Failed to load prev even results')
        with open(TEMP_DATA_PATH + 'predict_results_instance_' + str(instance_id) + '_odd.p', 'rb') as f:
            (
                probs_nc,
                inter_probs_mean,
                inter_probs_std,
                intraA_probs_mean,
                intraA_probs_std,
                intraB_probs_mean,
                intraB_probs_std,
                pred_lengths,
                _, 
                user_pairs
            ) = pickle.load(f)
        print('Already completed:', len(user_pairs), flush=True)    
        author_idxs_remaining = author_idxs[len(user_pairs):]
    '''
    author_idxs_remaining = author_idxs
    
    probs_nc = []


    inter_probs_mean = []
    inter_probs_std = []

    intraA_probs_mean = []
    intraA_probs_std = []

    intraB_probs_mean = []
    intraB_probs_std = []
    pred_lengths = []
    user_pairs = []
    
    print('Remaining:', len(author_idxs_remaining), flush=True)           
    for i, j in tqdm(author_idxs_remaining):
        a = selected_authors[i]
        b = selected_authors[j]
        start_a, end_a = author_bounds[a]
        start_b, end_b = author_bounds[b]
        
        if end_a - start_a > MAX_CHUNKS:
            end_a = start_a + MAX_CHUNKS
            
        if end_b - start_b > MAX_CHUNKS:
            end_b = start_b + MAX_CHUNKS
            
        l = []
        idxs = cartesian_product(range(start_a, end_a), range(start_b, end_b))        
        x_diff = secondary_scaler.transform(np.abs(XX[idxs[:, 0]] - XX[idxs[:, 1]]))
        x_diff[np.isnan(x_diff)]=0
        p = clf.predict_proba(x_diff)[:, 1]
        inter_probs_mean.append(p.mean())
        inter_probs_std.append(p.std())
        l.append(len(p))

        idxs = cartesian_product(range(start_a, end_a), range(start_a, end_a))
        idxs = np.array([(i, j) for i, j in idxs if i != j])
        x_diff = secondary_scaler.transform(np.abs(XX[idxs[:, 0]] - XX[idxs[:, 1]]))
        x_diff[np.isnan(x_diff)]=0
        p = clf.predict_proba(x_diff)[:, 1]
        intraA_probs_mean.append(p.mean())
        intraA_probs_std.append(p.std())
        l.append(len(p))

        idxs = cartesian_product(range(start_b, end_b), range(start_b, end_b))
        idxs = np.array([(i, j) for i, j in idxs if i != j])
        x_diff = secondary_scaler.transform(np.abs(XX[idxs[:, 0]] - XX[idxs[:, 1]]))
        x_diff[np.isnan(x_diff)]=0
        p = clf.predict_proba(x_diff)[:, 1]
        intraB_probs_mean.append(p.mean())
        intraB_probs_std.append(p.std())
        l.append(len(p))

        pred_lengths.append(l)
        
        idx_a = author_to_idx_nc[selected_authors[i]]
        idx_b = author_to_idx_nc[selected_authors[j]]
        p = clf_nc.predict_proba(secondary_scaler_nc.transform(np.abs(XX_nc[[idx_a], :] - XX_nc[[idx_b], :])))[0, 1]
        probs_nc.append(p)
        
        user_pairs.append((a, b))
        
        if len(user_pairs) % 500 == 0:
            with open(TEMP_DATA_PATH + 'predict_results_instance_' + str(instance_id) + '.p', 'wb') as f:
                pickle.dump((
                    probs_nc,
                    inter_probs_mean,
                    inter_probs_std,
                    intraA_probs_mean,
                    intraA_probs_std,
                    intraB_probs_mean,
                    intraB_probs_std,
                    pred_lengths,
                    author_idxs, 
                    user_pairs
                ), f)
                print('Saved progress...', flush=True)

    with open(TEMP_DATA_PATH + 'predict_results_instance_' + str(instance_id) + '.p', 'wb') as f:
        pickle.dump((
            probs_nc,
            inter_probs_mean,
            inter_probs_std,
            intraA_probs_mean,
            intraA_probs_std,
            intraB_probs_mean,
            intraB_probs_std,
            pred_lengths,
            author_idxs, 
            user_pairs
        ), f)
        
    print('Done!', flush=True)