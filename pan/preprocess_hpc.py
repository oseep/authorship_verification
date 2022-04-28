import os
import sys
sys.path.append("../")

import pickle
import json
import glob
from tqdm.auto import trange, tqdm
import sys
import numpy as np
from features import prepare_entry
from utills import chunker
import nltk

'''
DATA_DIR = '../data/pan/'
GROUND_TRUTH_PATH = DATA_DIR + 'pan20-authorship-verification-training-large-truth.jsonl'
DATA_PATH = DATA_DIR + 'pan20-authorship-verification-training-large.jsonl'
TEMP_DATA_PATH = '../temp_data/pan/'
'''

DATA_DIR = '/scratch/jnw301/av_public/data/pan/'
GROUND_TRUTH_PATH = DATA_DIR + 'pan20-authorship-verification-training-large-truth.jsonl'
DATA_PATH = DATA_DIR + 'pan20-authorship-verification-training-large.jsonl'
TEMP_DATA_PATH = '/scratch/jnw301/av_public/temp_data/pan/'



NUM_MACHINES = 20
TOKEN_BATCH_SZ = 100

def prepare_grouped_entries(token_spans, text, tokens_per_entry=500):
    groups = chunker(token_spans, tokens_per_entry)
    return [prepare_entry(text[spans[0][0]:spans[-1][1]], mode='accurate', tokenizer='casual') for spans in groups]


if __name__ == "__main__":
    instance_id = int(sys.argv[1])
    print('Instance ID for this machine:', instance_id, flush=True)
    
    train_ids, test_ids, _, _ = pickle.load(open(TEMP_DATA_PATH + 'dataset_partition.p', 'rb'))
    
    total_recs = len(train_ids) + len(test_ids)
    job_sz = total_recs // NUM_MACHINES
    start_rec = instance_id * job_sz
    end_rec = (instance_id + 1) * job_sz
    
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    
    print('Recs on this machine:', (end_rec - start_rec), flush=True)
    i = 0
    with open(DATA_PATH, 'r') as f, \
        open(TEMP_DATA_PATH + 'preprocessed_train_' + str(instance_id) + '.jsonl', 'w') as f_train, \
        open(TEMP_DATA_PATH + 'preprocessed_test_' + str(instance_id) + '.jsonl', 'w') as f_test:
        for l in tqdm(f, total=total_recs):
            i += 1
            if i < start_rec or i > end_rec:
                continue
            d = json.loads(l)
            
            spans1 = list(tokenizer.span_tokenize(d['pair'][0]))
            spans2 = list(tokenizer.span_tokenize(d['pair'][1]))
        
            e1 = prepare_grouped_entries(spans1, d['pair'][0], tokens_per_entry=TOKEN_BATCH_SZ)
            e2 = prepare_grouped_entries(spans2, d['pair'][1], tokens_per_entry=TOKEN_BATCH_SZ)
        
            preprocessed = {
                'id': d['id'],
                'fandoms': d['fandoms'],
                'pair': [
                    e1,
                    e2
                ]
            }
            if d['id'] in train_ids:
                json.dump(preprocessed, f_train)
                f_train.write('\n')
            else:
                json.dump(preprocessed, f_test)
                f_test.write('\n')