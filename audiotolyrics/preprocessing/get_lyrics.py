import lyricsgenius
import os 

def main():
    # Get all the songs and artists used
    dir_ = input("Directory for music: ")

    if dir_=="":
        dir_ = ".../mydata"
    files = os.listdir(dir_)
    files = [i for i in files if "txt" not in i]

    client_access_token = input("Genius Client Access: ")
    genius = lyricsgenius.Genius(client_access_token)

    for file in files:
        if "–" in file[:15]:
            temp = file.split("–")
        else:
            temp = file.split("-")
        artist, song = temp[0], temp[1]
        if "(" in song.lower() :
            song = song.split("(")[0]
        elif "[" in song.lower():
            song = song.split("[")[0]
        if "feat" in song:
            song = song.split("feat")[0]
        elif "ft" in song:
            song = song.split("ft")[0]

        if ".mp3" in song:
            song = song.split(".mp3")[0]

        song_genius = genius.search_song(song, artist)
        if song_genius is None:
            print(f"Skipping {song} by {artist}")
            continue
        filename = os.path.join(dir_, artist + " - " + song+".txt")
        with open(filename, "w+", encoding="utf-8") as fl:
            fl.write(song_genius.lyrics)

if __name__ == '__main__':
    main()