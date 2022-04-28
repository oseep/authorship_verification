import sys
sys.path.append("../")
import os
import pickle
import numpy as np
from tqdm.auto import trange, tqdm
from features import get_transformer, merge_entries
import json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from utills import chunker
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import RandomizedSearchCV


'''
DATA_DIR = '../data/pan/'
GROUND_TRUTH_PATH = DATA_DIR + 'pan20-authorship-verification-training-large-truth.jsonl'
DATA_PATH = DATA_DIR + 'pan20-authorship-verification-training-large.jsonl'
PREPROCESSED_DATA_PATH = '../temp_data/pan/'
EXPERIMENT_DATA_PATH = '../temp_data/pan/doc_size_experiments/'

'''
DATA_DIR = '/scratch/jnw301/av_public/data/pan/'
GROUND_TRUTH_PATH = DATA_DIR + 'pan20-authorship-verification-training-large-truth.jsonl'
DATA_PATH = DATA_DIR + 'pan20-authorship-verification-training-large.jsonl'
PREPROCESSED_DATA_PATH = '/scratch/jnw301/av_public/temp_data/pan/'
EXPERIMENT_DATA_PATH = '/scratch/jnw301/av_public/temp_data/pan/doc_size_experiments/'

SIZES = [1, 2, 5, 10, 20, 30, 40]


train_sz = 193536
test_sz = 81963
    
    
def fit_transformers(num_chunks, data_fraction=0.01):
    docs_1 = []
    docs_2 = []

    with open(PREPROCESSED_DATA_PATH + 'preprocessed_train.jsonl', 'r') as f:
        for l in tqdm(f):
            if np.random.rand() < data_fraction:
                d = json.loads(l)
                docs_1.append(merge_entries(d['pair'][0][:num_chunks]))
                docs_2.append(merge_entries(d['pair'][1][:num_chunks]))
                
    transformer = get_transformer()
    scaler = StandardScaler()
    secondary_scaler = StandardScaler()

    X = transformer.fit_transform(docs_1 + docs_2).todense()
    X = scaler.fit_transform(X)
    X1 = X[:len(docs_1)]
    X2 = X[len(docs_1):]
    secondary_scaler.fit(np.abs(X1 - X2))
    
    return transformer, scaler, secondary_scaler


def vectorize(XX, Y, ordered_idxs, transformer, scaler, secondary_scaler, preprocessed_path, vector_Sz, num_chunks):
    with open(preprocessed_path, 'r') as f:
        batch_size = 10000
        i = 0;
        docs1 = []
        docs2 = []
        idxs = []
        labels = []
        for l in tqdm(f, total=vector_Sz):
            d = json.loads(l)
            
            docs1.append(merge_entries(d['pair'][0][:num_chunks]))
            docs2.append(merge_entries(d['pair'][1][:num_chunks]))

            labels.append(ground_truth[d['id']])
            idxs.append(ordered_idxs[i])
            i += 1
            if len(labels) >= batch_size:
                x1 = scaler.transform(transformer.transform(docs1).todense())
                x2 = scaler.transform(transformer.transform(docs2).todense())
                XX[idxs, :] = secondary_scaler.transform(np.abs(x1-x2))
                Y[idxs] = labels

                docs1 = []
                docs2 = []
                idxs = []
                labels = []

        x1 = scaler.transform(transformer.transform(docs1).todense())
        x2 = scaler.transform(transformer.transform(docs2).todense())
        XX[idxs, :] = secondary_scaler.transform(np.abs(x1-x2))
        Y[idxs] = labels
        XX.flush()
        Y.flush()
        
        
if __name__ == "__main__":
    instance_id = int(sys.argv[1])
    print('Instance ID for this machine:', instance_id, flush=True)

    
    num_chunks = SIZES[instance_id]
    EXPERIMENT_DATA_PATH = EXPERIMENT_DATA_PATH + 'size_' + str(num_chunks) + '/'
    if not os.path.exists(EXPERIMENT_DATA_PATH):
        os.makedirs(EXPERIMENT_DATA_PATH)
        
    ground_truth = {}
    with open(GROUND_TRUTH_PATH, 'r') as f:
        for l in f:
            d = json.loads(l)
            ground_truth[d['id']] = d['same']  
            
    print('Fitting transformers...', flush=True)
    transformer, scaler, secondary_scaler = fit_transformers(num_chunks=num_chunks, data_fraction=0.05)
    feature_sz = len(transformer.get_feature_names())
    
    
    with open(EXPERIMENT_DATA_PATH + '.p', 'wb') as f:
        pickle.dump((transformer, scaler, secondary_scaler ), f)
        
    print('Vectorizing train set...', flush=True)
    XX_train = np.memmap(EXPERIMENT_DATA_PATH + 'vectorized_XX_train.npy', dtype='float32', mode='w+', shape=(train_sz, feature_sz))
    Y_train = np.memmap(EXPERIMENT_DATA_PATH + 'Y_train.npy', dtype='int32', mode='w+', shape=(train_sz))
    train_idxs = np.array(range(train_sz))
    np.random.shuffle(train_idxs)

    vectorize(
        XX_train, 
        Y_train, 
        train_idxs, 
        transformer, 
        scaler, 
        secondary_scaler, 
        PREPROCESSED_DATA_PATH + 'preprocessed_train.jsonl',
        train_sz,
        num_chunks=num_chunks
    )
    
    print('Vectorizing test set...', flush=True)
    XX_test = np.memmap(EXPERIMENT_DATA_PATH + 'vectorized_XX_test.npy', dtype='float32', mode='w+', shape=(test_sz, feature_sz))
    Y_test = np.memmap(EXPERIMENT_DATA_PATH + 'Y_test.npy', dtype='int32', mode='w+', shape=(test_sz))
    test_idxs = np.array(range(test_sz))
    np.random.shuffle(test_idxs)

    vectorize(
        XX_test, 
        Y_test, 
        test_idxs, 
        transformer, 
        scaler, 
        secondary_scaler, 
        PREPROCESSED_DATA_PATH + 'preprocessed_test.jsonl',
        test_sz,
        num_chunks=num_chunks
    )
    
    
    print('Training classifier...', flush=True)
    clf = SGDClassifier(loss='log', alpha=0.01)
    batch_size=50000
    num_epochs = 50
    aucs = []
    for i in trange(num_epochs):
        print('Epoch - ', i)
        print('-' * 30)
        for idxs in chunker(range(train_sz), batch_size):
            clf.partial_fit(XX_train[idxs, :], Y_train[idxs], classes=[0, 1])

        probs = clf.predict_proba(XX_test)[:, 1]
        fpr, tpr, thresh = roc_curve(Y_test, probs)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        print('AUC: ', roc_auc)
        
        
    with open(EXPERIMENT_DATA_PATH + 'experiment_data.p', 'wb') as f:
        pickle.dump((
            aucs,
            clf,
            roc_auc,
            transformer, 
            scaler,
            secondary_scaler,
            feature_sz,
            train_sz,
            train_idxs,
            test_sz,
            test_idxs
        ), f)