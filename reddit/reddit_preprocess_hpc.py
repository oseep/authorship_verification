import sys
sys.path.append("../")

import pickle
import glob
from tqdm import trange, tqdm
import json
import re
import pandas as pd
import numpy as np
from features import prepare_entry

NUM_MACHINES = 10
BASE_PATH = '../data/reddit_2/'

DATA_DIR = '/scratch/jnw301/av/data/reddit_2/reddit_2/'
COMPUTED_DATA_PATH = '/scratch/jnw301/av/temp_data/april_may_experiments/reddit_preprocessed_2/'


#DATA_DIR = '../data/reddit_2/'
#COMPUTED_DATA_PATH = '../temp_data/april_may_experiments/reddit_preprocessed_2/'

def colnum_string(n):
    string = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        string = chr(65 + remainder) + string
    return string


if __name__ == "__main__":
    instance_id = int(sys.argv[1])
    print('Instance ID for this machine:', instance_id, flush=True)
    with open(COMPUTED_DATA_PATH + 'metadata.p', 'rb') as f:
        (train_files, test_files, min_count) = pickle.load(f)
        
    file_arr = train_files
    total_recs = len(file_arr)
    job_sz = total_recs // NUM_MACHINES
    start_rec = instance_id * job_sz
    end_rec = (instance_id + 1) * job_sz
    curr_files = file_arr[start_rec:end_rec]
    
    
    print('Recs on this machine:', (end_rec - start_rec), flush=True)
    i = 0
    
    
    author_mapping = {}
    with open(COMPUTED_DATA_PATH + 'train' + str(instance_id) + '.jsonl', 'w') as fout:
        
        for file in tqdm(curr_files):
            file = file.replace('../data/reddit_2/', DATA_DIR)
            with open(file, 'r') as f:
                try:
                    data = [json.loads(l) for l in f.readlines()]
                except:
                    continue
            if len(data) == 0:
                continue
            df = pd.DataFrame(data=data)
            subreddit_counts = df['subreddit'].value_counts()
            subreddit_counts = subreddit_counts[subreddit_counts > min_count]
            if len(subreddit_counts) < 2:
                continue
            username = file.replace(BASE_PATH, '').replace('.jsonl', '')
            sub_authors = []
            for i in range(len(subreddit_counts)):
                comments = df.loc[df['subreddit']==subreddit_counts.index[i]]['body'].values
                preprocessed = [prepare_entry(c, mode='accurate', tokenizer='casual') for c in comments]

                d = {
                    'username': username + '_' + colnum_string(i + 1),
                    'data': preprocessed,
                    'subreddit': subreddit_counts.index[i]

                }
                json.dump(d, fout)
                fout.write('\n')
                sub_authors.append(username + '_' + colnum_string(i + 1))
            author_mapping[username] = sub_authors
            
    with open(COMPUTED_DATA_PATH + 'author_mapping_' + str(instance_id) + '.p', 'wb') as f:
        pickle.dump(author_mapping, f)