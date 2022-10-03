import os
import sys
sys.path.append("../")

import pickle
import json
import glob
from tqdm.auto import trange, tqdm
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from features import merge_entries, prepare_entry
import nltk
from utills import chunker

PREPROCESSED_DATA_PATH = '/scratch/jnw301/av_public/temp_data/pan/'
DATA_DIR = '/scratch/jnw301/av_public//data/pan/'
GROUND_TRUTH_PATH = DATA_DIR + 'pan20-authorship-verification-training-large-truth.jsonl'
TEMP_DATA_PATH = '/scratch/jnw301/av_public/temp_data/ai/'

'''
PREPROCESSED_DATA_PATH = '../temp_data/pan/'
DATA_DIR = '../data/pan/'
GROUND_TRUTH_PATH = DATA_DIR + 'pan20-authorship-verification-training-large-truth.jsonl'
TEMP_DATA_PATH = '../temp_data/ai/'
'''


MAX_RECORDS = 100
NUM_MACHINES = 20

'''
def generate_ai_and_human_text_pair(text_generation, preprocessed_doc):
    human_texts_preprocessed = [c for i, c in enumerate(preprocessed_doc) if i % 2 > 0]
    prompt_texts = [c['preprocessed'] for i, c in enumerate(preprocessed_doc) if i % 2 == 0]
    generated_texts = text_generation(prompt_texts, max_length=300, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    generated_texts = [gt[0]['generated_text'].replace(pt, '') for gt, pt in zip(generated_texts, prompt_texts)]
    generated_texts_preprocessed = [prepare_entry(t) for t in generated_texts]
    return human_texts_preprocessed, generated_texts_preprocessed
'''

def generate_ai_and_human_text_pair(text_generation, nltk_tokenizer, preprocessed_doc):
    prompt_texts = [c['preprocessed'] for i, c in enumerate(preprocessed_doc) if i % 2 == 0]
    generated_texts = text_generation(prompt_texts, max_length=450, num_beams=5, no_repeat_ngram_size=2)
    generated_text = '\n'.join([gt[0]['generated_text'].replace(pt, '') for gt, pt in zip(generated_texts, prompt_texts)])
    
    spans = list(nltk_tokenizer.span_tokenize(generated_text))
    groups = chunker(spans, 110)
    generated_texts_preprocessed = [prepare_entry(generated_text[spans[0][0]:spans[-1][1]], mode='accurate', tokenizer='casual') for spans in groups]
    return preprocessed_doc, generated_texts_preprocessed

if __name__ == "__main__":
    instance_id = int(sys.argv[1])
    print('Instance ID for this machine:', instance_id, flush=True)
    
    
    ground_truth = {}
    with open(GROUND_TRUTH_PATH, 'r') as f:
        for l in f:
            d = json.loads(l)
            ground_truth[d['id']] = d['same']
            

    fanfic_recs = []
    with open(PREPROCESSED_DATA_PATH + 'preprocessed_test.jsonl', 'r') as f:
        for l in tqdm(f):
            d = json.loads(l)
            if ground_truth[d['id']] == True:
                fanfic_recs.append(d)
            if len(fanfic_recs) > MAX_RECORDS:
                break
    
    print('Loading models...', flush=True)            
    model = AutoModelForCausalLM.from_pretrained("gpt2-large")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    text_generation = pipeline('text-generation', model=model, tokenizer=tokenizer)

    job_sz = MAX_RECORDS // NUM_MACHINES
    start_rec = instance_id * job_sz
    end_rec = (instance_id + 1) * job_sz
    fanfic_recs = fanfic_recs[start_rec:end_rec]
    nltk_tokenizer = nltk.tokenize.WhitespaceTokenizer()

    processed_ids = []
    with open(TEMP_DATA_PATH + 'human_ai_preprocessed' + str(instance_id) + '.jsonl', 'r') as f:
        for l in f:
            d = json.loads(l)
            processed_ids.append(d['id'])
            
    print(processed_ids, flush=True)        
    print('Recs on this machine:', (end_rec - start_rec), flush=True)
    with open(TEMP_DATA_PATH + 'human_ai_preprocessed' + str(instance_id) + '.jsonl', 'a') as f_out:
        for d in tqdm(fanfic_recs):
            print(d['id'], d['id'] in processed_ids, flush=True)
            if d['id'] in processed_ids:
                continue
            d1_human, d1_ai = generate_ai_and_human_text_pair(text_generation, nltk_tokenizer, d['pair'][0])
            d2_human, d2_ai = generate_ai_and_human_text_pair(text_generation, nltk_tokenizer, d['pair'][1])

            preprocessed = {
                'id': d['id'],
                'fandoms': d['fandoms'],
                'pair': [
                    {'human': d1_human, 'ai': d1_ai},
                    {'human': d2_human, 'ai': d2_ai}
                ]
            }
            json.dump(preprocessed, f_out)
            f_out.write('\n')
            f_out.flush()