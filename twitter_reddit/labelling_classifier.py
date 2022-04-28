import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import nltk
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import re
import pickle

def extract_username(text):
    text = text.replace('\_', '_')
    match = re.search('twitter.com/([a-zA-Z0-9_]+)/status', text)
    if match:
        return match.group(1)
    match = re.search('twitter.com/#!/([a-zA-Z0-9_]+)', text)
    if match:
        return match.group(1)
    match = re.search('twitter.com/([a-zA-Z0-9_]*)', text)
#     match = re.search('.*twitter.com/([a-zA-Z0-9_]+)?([^?#]*)(\?([^#]*)).*', text)
    if match:
        return match.group(1)
    match = re.search('twitter.com/([a-zA-Z0-9_]+)$', text)
    if match:
        return match.group(1)

    return None


class CustomFuncTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer_func, fnames=None):
        self.transformer_func = transformer_func
        self.fnames = fnames
        
    def fit(self, x, y=None):
        return self;
    
    def transform(self, x):
        xx = np.array([self.transformer_func(entry) for entry in x])
        if len(xx.shape) == 1:
            return xx[:, None]
        else:
            return xx
    
    def get_feature_names(self):
        if self.fnames is None:
            return ['']
        else:
            return self.fnames
        
class CustomTfIdfTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(min_df=0.01, stop_words='english')

    def fit(self, x, y=None):
        self.vectorizer.fit([r['body'] for r in x], y)
        return self

    def transform(self, x):
        return self.vectorizer.transform([r['body'] for r in x])
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names()

class CustomPhraseTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, phrases):
        self.phrases = phrases

    def fit(self, x, y=None):
        return self

    def __process(self, r):
        text = r['body'].lower()
        ret = []
        for p in self.phrases:
            if p in text:
                ret.append(1)
            else:
                ret.append(0)
        return ret
    
    
    def transform(self, x):
        return [self.__process(xx) for xx in x]
    
    def get_feature_names(self):
        return self.phrases
    
    
class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()
    
def edit_distance(r):
    twitter_un = extract_username(r['body'])
    reddit_un = r['author']
    try:
        d = nltk.edit_distance(twitter_un.lower(), reddit_un.lower())
        return d / (0.5 * (len(twitter_un) + len(reddit_un)))
    except:
        return 1000
    
def get_r_body(r):
    return [rr['body'] for rr in r]


def update_model(new_positives, new_negatives):
    with open('labelling_model.p', 'rb') as f:
        (
            clf,
            positive_posts,
            negative_posts
        ) = pickle.load(f)
    positive_posts += new_positives
    negative_posts += new_negatives
    X = positive_posts + negative_posts
    Y = [1] * len(positive_posts) + [0] * len(negative_posts)
    clf = get_classifier_pipeline()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresh = roc_curve(y_test, preds)
    print('New AUC:', auc(fpr, tpr))
    
    with open('labelling_model.p', 'wb') as f:
        pickle.dump((
            clf,
            positive_posts,
            negative_posts
        ), f)
    return clf

def get_classifier_pipeline():
    edit_dist_transformer = CustomFuncTransformer(edit_distance, fnames=['edit_dist'])
    tf_idf_transformer = CustomTfIdfTransformer()
    phrases = CustomPhraseTransformer(['my twitter account', 'my twitter', 'twitter account', 'not my', 'isnt my', 'isn\'t my'])
    featuresets = [
            ('edit_dist', edit_dist_transformer),
            ('tfidf', tf_idf_transformer),
            ('phrases', phrases)
        ]
    transformer = FeatureUnion(featuresets)
    clf = make_pipeline(transformer, DenseTransformer(), StandardScaler(), RandomForestClassifier())
    return clf
