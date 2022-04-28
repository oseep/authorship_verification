
import pickle
import numpy as np
from tqdm.auto import tqdm, trange
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from features import prepare_entry, get_transformer, merge_entries
from utills import chunker, cartesian_product

NUM_MACHINES = 10

def generate_doc_pairs(r_usernames, t_usernames, r_user_docs, t_user_docs, negative_twitter_sample_order=None):
    all_docs = []
    reddit_author_bounds = {}
    twitter_author_bounds = {}
    n_user_pairs = len(t_usernames)
    
    # Create a list of all the docs and save mappings
    for i in range(n_user_pairs):
        docs = r_user_docs[i]
        reddit_author_bounds[r_usernames[i]] = (len(all_docs), len(all_docs) + len(docs))
        all_docs.extend(docs)

        docs = t_user_docs[i]
        twitter_author_bounds[t_usernames[i]] = (len(all_docs), len(all_docs) + len(docs))
        all_docs.extend(docs)

    all_docs = np.array(all_docs)
    
    # Generate doc pairs

    idxs = []
    labels = []
    # Positive samples
    for i in range(n_user_pairs):
        r_s, r_e = reddit_author_bounds[r_usernames[i]]
        t_s, t_e = twitter_author_bounds[t_usernames[i]]

        idx_pairs = cartesian_product(range(r_s, r_e), range(t_s, t_e))
        idxs.extend(idx_pairs)
        labels.extend([1]*len(idx_pairs))

    # Negative samples
    if negative_twitter_sample_order is None:
        shuffled_order = np.random.permutation(len(t_usernames))
    for i in range(n_user_pairs):
        
        if negative_twitter_sample_order is not None:
            r_s, r_e = reddit_author_bounds[r_usernames[negative_twitter_sample_order[i][0]]]
            t_s, t_e = twitter_author_bounds[t_usernames[negative_twitter_sample_order[i][1]]]
        else:
            r_s, r_e = reddit_author_bounds[r_usernames[i]]
            t_s, t_e = twitter_author_bounds[t_usernames[shuffled_order[i]]]
            
        idx_pairs = cartesian_product(range(r_s, r_e), range(t_s, t_e))
        idxs.extend(idx_pairs)
        labels.extend([0]*len(idx_pairs))

    idxs = np.array(idxs)
    labels = np.array(labels)
    p = np.random.permutation(len(idxs))
    idxs = idxs[p]
    labels = labels[p]
    
    return all_docs, idxs, labels


if __name__ == "__main__":
    this_instance = int(sys.argv[1])
    print('Instance ID for this machine:', this_instance, flush=True)
    
    
    with open('twitter_reddit_preprocessed_data_manual_dataset.p', 'rb') as f:
        (
            reddit_usernames,
            twitter_usernames,
            reddit_to_twitter,
            reddit_docs_preprocessed,
            twitter_docs_preprocessed
        ) = pickle.load(f)
    
    reddit_usernames_ordered = []
    twitter_usernames_ordered = []
    for r, t in reddit_to_twitter.items():
        reddit_usernames_ordered.append(r)
        twitter_usernames_ordered.append(t)

    reddit_usernames_ordered = np.array(reddit_usernames_ordered)
    twitter_usernames_ordered = np.array(twitter_usernames_ordered)

    twitter_docs_preprocessed = np.array(twitter_docs_preprocessed)
    reddit_docs_preprocessed = np.array(reddit_docs_preprocessed)

    kfold = KFold(n_splits=10)
    folds = list(kfold.split(range(len(reddit_usernames_ordered))))
    folds_per_machine = len(folds)//NUM_MACHINES
    folds = folds[folds_per_machine * this_instance : folds_per_machine * (this_instance + 1)]
    
    print('Num folds on this machine:', len(folds), flush=True)
    
    results = []
    for train_user_index, test_user_index in folds:

        r_usernames = reddit_usernames_ordered[train_user_index]
        t_usernames = twitter_usernames_ordered[train_user_index]

        r_user_docs = reddit_docs_preprocessed[train_user_index]
        t_user_docs  = twitter_docs_preprocessed[train_user_index]


        train_docs, train_pair_idxs, train_labels = generate_doc_pairs(r_usernames, t_usernames, r_user_docs, t_user_docs)

        # Fit transformers
        transformer = get_transformer()
        primary_scaler = StandardScaler()
        XX_train = transformer.fit_transform(train_docs)
        XX_train = primary_scaler.fit_transform(XX_train.todense())

        XX_train[np.isnan(XX_train)] = 0

        batch_size = 50000
        secondary_scaler = StandardScaler()
        XX_diff_train = np.abs(XX_train[train_pair_idxs[:batch_size, 0], :] - XX_train[train_pair_idxs[:batch_size, 1], :])
        secondary_scaler.fit_transform(XX_diff_train)


        # Prepare validation/(test) set

        r_usernames_test = reddit_usernames_ordered[test_user_index]
        t_usernames_test = twitter_usernames_ordered[test_user_index]
        r_user_docs_test = reddit_docs_preprocessed[test_user_index]
        t_user_docs_test  = twitter_docs_preprocessed[test_user_index]


        test_docs, test_pair_idxs, test_labels = generate_doc_pairs(r_usernames_test, t_usernames_test, r_user_docs_test, t_user_docs_test)
        XX_test = primary_scaler.transform(transformer.transform(test_docs).todense())

        XX_test[np.isnan(XX_test)] = 0

        XX_diff_test = np.abs(XX_test[test_pair_idxs[:, 0], :] - XX_test[test_pair_idxs[:, 1], :])
        XX_diff_test = secondary_scaler.transform(XX_diff_test)


        print('Training classifier...', flush=True)
        clf = SGDClassifier(loss='log', alpha=0.1)
        aucs = []
        num_epochs = 20
        for i in trange(num_epochs):
            print('Epoch - ', i)
            print('-' * 30)
            for j in tqdm(chunker(range(len(train_labels)), batch_size), total=len(train_labels)//batch_size):
                x_idxs = train_pair_idxs[j]
                y = train_labels[j]
                x = np.abs(XX_train[x_idxs[:, 0], :] - XX_train[x_idxs[:, 1], :])
                clf.partial_fit(secondary_scaler.transform(x), y, classes=[0, 1])

            probs = clf.predict_proba(XX_diff_test)[:, 1]
            fpr, tpr, thresh = roc_curve(test_labels, probs)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            print('AUC: ', roc_auc)

        result = (
            train_user_index,
            test_user_index,
            aucs,
            roc_auc,
            probs,
            test_labels
        )
        results.append(result)
        
    with open('results_' + str(this_instance) + '.p', 'wb') as f:
        pickle.dump(results, f)