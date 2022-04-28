import itertools
import numpy as np
import re
import nltk
import gzip

def compress_fraction(text):
    text = bytes(text,'utf-8')
    compressed_value = gzip.compress(text)
    return len(compressed_value)/ len(text)

def cartesian_product(*arrays):
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=int)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        
        
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def get_num_chunks(arr, chunk_sz):
    """
    Returns the number of chunks that would be generated from the given array and chunk size
    """
    return int(len(arr)//chunk_sz if len(arr) % chunk_sz == 0 else np.floor(len(arr)/chunk_sz) + 1)

def binarize(y, threshold=0.5):
    y = np.array(y)
    y = np.ma.fix_invalid(y, fill_value=threshold)
    y[y >= threshold] = 1
    y[y < threshold] = 0
    return y

def c_at_1(true_y, pred_y, threshold=0.5):
    """
    Calculates the c@1 score, an evaluation method specific to the
    PAN competition. This method rewards predictions which leave
    some problems unanswered (score = 0.5). See:
        A. Peñas and A. Rodrigo. A Simple Measure to Assess Nonresponse.
        In Proc. of the 49th Annual Meeting of the Association for
        Computational Linguistics, Vol. 1, pages 1415-1424, 2011.
    Parameters
    ----------
    prediction_scores : array [n_problems]
        The predictions outputted by a verification system.
        Assumes `0 >= prediction <=1`.
    ground_truth_scores : array [n_problems]
        The gold annotations provided for each problem.
        Will always be `0` or `1`.
    Returns
    ----------
    c@1 = the c@1 measure (which accounts for unanswered
        problems.)
    References
    ----------
        - E. Stamatatos, et al. Overview of the Author Identification
        Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.
        - A. Peñas and A. Rodrigo. A Simple Measure to Assess Nonresponse.
        In Proc. of the 49th Annual Meeting of the Association for
        Computational Linguistics, Vol. 1, pages 1415-1424, 2011.
    """

    n = float(len(pred_y))
    nc, nu = 0.0, 0.0

    for gt_score, pred_score in zip(true_y, pred_y):
        if pred_score == 0.5:
            nu += 1
        elif (pred_score > 0.5) == (gt_score > 0.5):
            nc += 1.0
    
    return (1 / n) * (nc + (nu * nc / n))


def f_05_u_score(true_y, pred_y, pos_label=1, threshold=0.5):
    """
    Return F0.5u score of prediction.
    :param true_y: true labels
    :param pred_y: predicted labels
    :param threshold: indication for non-decisions (default = 0.5)
    :param pos_label: positive class label (default = 1)
    :return: F0.5u score
    """

    pred_y = binarize(pred_y)

    n_tp = 0
    n_fn = 0
    n_fp = 0
    n_u = 0

    for i, pred in enumerate(pred_y):
        if pred == threshold:
            n_u += 1
        elif pred == pos_label and pred == true_y[i]:
            n_tp += 1
        elif pred == pos_label and pred != true_y[i]:
            n_fp += 1
        elif true_y[i] == pos_label and pred != true_y[i]:
            n_fn += 1

    return (1.25 * n_tp) / (1.25 * n_tp + 0.25 * (n_fn + n_u) + n_fp)


def edit_distance(t1, t2):
    try:
        d = nltk.edit_distance(t1.lower(), t2.lower())
        return d / (0.5 * (len(t1) + len(t2)))
    except:
        return 1000
    
    
def preprocess_twitter(text):
    regex_username = r'(^|[^@\w])@(\w{1,15})\b'
    regex_hashtag = r'\#([A-Za-z]+[A-Za-z0-9-_]+)'
    regex_url = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(regex_url + '|' + regex_username + '|' + regex_hashtag, '', text)

def extract_twitter_username(text):
    text = text.replace('\_', '_')
    match = re.search('twitter.com/([a-zA-Z0-9_]+)/status', text)
    if match:
        return match.group(1)
    match = re.search('twitter.com/#!/([a-zA-Z0-9_]*)', text)
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


def colnum_string(n):
    '''
    Converts an integer to an excel column type string
    1 -> A, 2->B, etc
    '''
    string = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        string = chr(65 + remainder) + string
    return string


class ReservoirSample(object):
    def __init__(self, n):
        self.reservoir = []
        self.n = n
        self.i = 0

    def add(self, item):
        self.i += 1
        if len(self.reservoir) < self.n:
            self.reservoir.append(item)
        else:
            m = np.random.randint(0, self.i)
            if m < self.n:
                self.reservoir[m] = item