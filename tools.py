#import numpy as np
#import pandas as pd
import spotipy 
from spotipy.oauth2 import SpotifyClientCredentials

client_id = '1dd679e02daa4de4bf2de3d89a86180d'
client_secret = '88c84edcdcaf4f999fc3d11eed560856'
pl_id = '2sjVpVZpcje4dIznYcQfAC'
#get the data
col_names = ['acousticness', 'danceability', 'duration','energy','instrumentalness','liveliness','tempo','valence','genre']
data = [{},{},{},{},{},{},{},{},{}]
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id,
                                                            client_secret=client_secret))
def get_audio_features(track_id):
    audio_features = sp.audio_features(track_id)
    audio_features = audio_features[0]
    danceability = audio_features['danceability']
    energy = audio_features['energy']
    loudness = audio_features['loudness']
    liveness = audio_features['liveness']
    instrumentalness = audio_features['instrumentalness']
    acousticness = audio_features['acousticness']
    speechiness = audio_features['speechiness'] 
    duration = audio_features['duration_ms']
    tempo = audio_features['tempo']
    valence = audio_features['valence']
    arr = [acousticness,danceability,duration,energy,instrumentalness,liveness,loudness,speechiness,tempo,valence]
    return arr

playList = sp.playlist(pl_id, fields=None, market=None, additional_types=('track', ))

def cleanUpLink(link):
    question_mark_index = link.find("?")
    link.split('https://open.spotify.com/track')
    cleaned = link[:question_mark_index] 
    return str(cleaned)

track_id = cleanUpLink('https://open.spotify.com/track/2yAVzRiEQooPEJ9SYx11L3?si=2afb5278a3564f55')



#get_audio_features(track_id)