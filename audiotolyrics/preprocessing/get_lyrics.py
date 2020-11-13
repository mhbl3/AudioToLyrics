import lyricsgenius
import os 

# Get all the songs and artists used 
dir_ = input("Directory for music: ")

if dir_=="":
    dir_ = ".../mydata"
files = os.listdir(dir_)
files = [i for i in files if "txt" not in i]

client_access_token = input("Genius Client Access: ") 
genius = lyricsgenius.Genius(client_access_token)

for file in files:
    temp = file.split("-")
    artist, leftovers = temp[0], temp[1]
    if "official" in leftovers.lower() or "video" in leftovers.lower():
        temp = leftovers.split("(")
        leftovers = temp[0]
    else:
        leftovers = leftovers.split("-")[0]
        
    if "feat" in leftovers.lower():
        song = leftovers.lower().split("feat")[0]
    elif "ft" in leftovers.lower():
        song = leftovers.lower().split("ft")[0]
    else:
        song = leftovers
        
    song_genius = genius.search_song(song, artist)
    if song_genius is None:
        print(f"Skipping {song} by {artist}")
        continue
    filename = os.path.join(dir_, artist + " - " + song+".txt")
    with open(filename, "w+", encoding="utf-8") as fl:
        fl.write(song_genius.lyrics)
