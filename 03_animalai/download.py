from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys

key = "3dd99ac105fa7e445f065bf5256623af"
secret = "e0fd3a12eb2c65d6"
wait_time = 1
animalname = sys.argv[1]
savedir = "./" + animalname

flickr = FlickrAPI(key, secret, format="parsed-json")
result = flickr.photos.search(
    text = animalname,
    per_page = 10,
    media = "photos",
    sort = "relevance",
    safe_search = 1,
    extras = "url_l, licence"
)
# URL of large square 150x150 size image
# extras = "url_q, licence"

photos = result["photos"]
# pprint(photos)

for i, photo in enumerate(photos["photo"]):
    url_q = photo["url_l"]
    filepath = savedir + "/" + photo["id"] + ".jpg"
    if os.path.exists(filepath):
        continue
    urlretrieve(url_q, filepath)
    time.sleep(wait_time)
