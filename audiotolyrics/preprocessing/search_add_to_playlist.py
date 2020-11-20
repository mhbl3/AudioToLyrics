import spotipy
import os
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np

def main():
    cid = input("Cliend ID: ")
    os.system('cls' if os.name == 'nt' else 'clear')

    secret = input("Client Secret: ")
    os.system('cls' if os.name == 'nt' else 'clear')

    path_to_artists = input("Path to artist names: ")

    with open(path_to_artists, "r") as f:
      artists = f.readlines()
    artists_list = [i.replace("\n", "") for i in artists]

    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager, )

    artist_name = []
    track_name = []
    popularity = []
    track_id = []

    print("Searching for tracks")
    num_tracks = input("Number of tracks per artist: ")
    num_tracks = int([10 if num_tracks == "" else num_tracks][0])

    for artist in artists_list:
        # for i in range(0, len_tracks):
        track_results = sp.search(q=artist, type='track', limit=num_tracks,)
        for i, t in enumerate(track_results['tracks']['items']):
            artist_name.append(t['artists'][0]['name'])
            track_name.append(t['name'])
            track_id.append(t['id'])
            popularity.append(t['popularity'])

    count = len(track_id)
    # print(track_id)
    print(f"Number of tracks: {count}")
    uri = 'https://localhost'
    scope = "playlist-modify-private"
    username = input("username: ")
    token = util.prompt_for_user_token(username=username,
                                       scope=scope
                                       ,client_id= cid,
                                       client_secret=secret,
                                       redirect_uri=uri, show_dialog=True)

    print("Adding tracks to playlist")
    if "rap" in path_to_artists.lower() :
        playlist_id = "3YXnAB19mU5QDha8gPN5Jl" # audioToLyrics_rap playlist of makybl
    elif "country" in path_to_artists.lower():
        playlist_id = "7kgsLrjLRwimdBwhxuqS82" # audioToLyrics_country playlist of makybl

    if token:
        sp = spotipy.Spotify(auth=token)
        sp.trace = False
        for track in track_id:
            sp.user_playlist_add_tracks(username, playlist_id, [track], )
    print("Done!")

if __name__ == '__main__':
    main()