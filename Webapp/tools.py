import requests

def is_url_ok(url):
    return 200 == requests.head(url).status_code