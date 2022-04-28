import tweepy
import time
import requests
import glob
import json

# Code modified from https://twittercommunity.com/t/get-all-historical-tweets-of-a-user-limits-me-only-to-3200-tweets/145610
def get_all_tweets(api, screen_name):
    alltweets = []
    try:
        new_tweets = api.user_timeline(screen_name=screen_name, tweet_mode="extended", count=200)
    except tweepy.RateLimitError:
        print('Rate limitted. Wating...')
        time.sleep(15 * 60)
        
    #save most recent tweets
    alltweets.extend(new_tweets)
    if len(alltweets) == 0:
        return alltweets
    
    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
#         print("getting tweets before %s" % (oldest))

        #all subsiquent requests use the max_id param to prevent duplicates
        try:
            new_tweets = api.user_timeline(screen_name = screen_name, count=200, max_id=oldest, tweet_mode="extended")
        except tweepy.RateLimitError:
            print('Rate limitted. Wating...')
            time.sleep(15 * 60)
        #save most recent tweets
        alltweets.extend(new_tweets)

        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

#         print("...%s tweets downloaded so far" % (len(alltweets)))
    return alltweets



def call_reddit_user_comment_search(author_name, after, before):
    url = 'https://api.pushshift.io/reddit/search/comment/?author=' + str(author_name) + '&size=1000&after=' + str(after) + '&before=' + str(before)
#     print(url)
    r = requests.get(url)
    
    while str(r) == '<Response [429]>':
        time.sleep(10)
        r = requests.get(url)
#         print(r)
        
    data = json.loads(r.text)
    
    return data['data']

def reddit_user_comments(author_name, after_date, before_date):
    
    data = call_reddit_user_comment_search(author_name, after_date, before_date)
    
    all_data_user = []

    while len(data) > 0:
        """
            Calls getPushshiftData() with the created date of the last submission
        """
        for com in data:
            all_data_user.append(com['body'])

        ##Change the after
        after = data[-1]['created_utc']
        data = call_reddit_user_comment_search(author_name, after, before_date)
        
    return all_data_user


def call_reddit_user_submissions_search(author_name, after, before):
    url = 'https://api.pushshift.io/reddit/search/submission/?author=' + str(author_name) + '&size=1000&after=' + str(after) + '&before=' + str(before)
#     print(url)
    r = requests.get(url)
    
    while str(r) == '<Response [429]>':
        time.sleep(10)
        r = requests.get(url)
#         print(r)
        
    data = json.loads(r.text)
    
    return data['data']

def reddit_user_submissions(author_name, after_date, before_date, max_data=10000):
    data = call_reddit_user_submissions_search(author_name, after_date, before_date)
    
    all_data_user = []

    while len(data) > 0:
        """
            Calls getPushshiftData() with the created date of the last submission
        """
        for com in data:
            text = com['title']
            if 'selftext' in com:
                text += '\n' + com['selftext']
            all_data_user.append(text)

        ##Change the after
        after = data[-1]['created_utc']
        data = call_reddit_user_submissions_search(author_name, after, before_date)
        if len(all_data_user) > max_data:
            break
        
    return all_data_user





def query_reddit_comments(query, before, after):
    url = 'https://api.pushshift.io/reddit/search/comment/?q=' + urllib.parse.quote(query) + '&after='+str(after)+'&before='+str(before)
    r = requests.get(url)
    
    while str(r) == '<Response [429]>':
        time.sleep(10)
        r = requests.get(url)
    data = json.loads(r.text)
    return data['data']


def query_reddit_submissions(query, before, after):
    url = 'https://api.pushshift.io/reddit/search/submission/?q=' + urllib.parse.quote(query) + '&after='+str(after)+'&before='+str(before)
    r = requests.get(url)
    
    while str(r) == '<Response [429]>':
        time.sleep(10)
        r = requests.get(url)
    data = json.loads(r.text)
    return data['data']