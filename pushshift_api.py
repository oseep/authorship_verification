import requests
import json
import time

def get_user_comments(user, after, before):
    url = 'https://api.pushshift.io/reddit/search/comment/?author='+str(user)+'&size=1000&after='+str(after)+'&before='+str(before)
    r = requests.get(url)
    
    while str(r) == '<Response [429]>':
        time.sleep(10)
        r = requests.get(url)
        print(r)
        
    data = json.loads(r.text)
    return data['data']