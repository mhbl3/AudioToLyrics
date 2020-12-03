import torch
import librosa
import pandas as pd
import numpy as np
import pickle as pkl
import os
from keras.utils import to_categorical
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class audioContainer():
    def __init__(self, path_songs, sr, use_log_spectrogram=False, song_duration=15, offset=25, limit=None):
        assert type(path_songs) == list
        self.audio_with_text = []
        self.use_log_spectrogram = use_log_spectrogram
        self.paths = []
        flag = True
        for path_song in path_songs:
            all_data = os.listdir(path_song)
            print(f"Number of files in folder (lyrics+ tracks): {len(all_data)}")
            list_audio_only = [audio for audio in all_data if "txt" not in audio]
            list_audio_only_no_space = [audio.replace(" ", "") for audio in list_audio_only]
            text_no_space = [text.replace(" ", "") for text in all_data if "txt" in text]
            # temp_audio = [audio.replace("–", "-") for audio in list_audio_only_no_space]
            temp = []
            for lyrics in text_no_space:
                for idx, audio in enumerate(list_audio_only_no_space):
                    if lyrics.split(".txt")[0] in audio:
                        temp.append(list_audio_only[idx])
                        break
            # temp = [list_audio_only[i] for i, audio in enumerate(temp_audio) for text in text_no_space if text in audio.split("-")[1]]
            self.audio_with_text.extend(temp)
            print(f"Number of audio tracks with lyrics: {len(temp)}")
            tmp = [os.path.join(path_song, audio) for audio in temp]
            self.paths.extend(tmp)
            if flag:
                idx_last_first_genre = len(self.paths) -1
                flag = False
        if limit is not None:
            my_reduced_path = []
            my_reduced_audio_text = []
            idx = np.random.choice(np.arange(0, len(self.paths[:idx_last_first_genre+1])), size=limit//2, replace=False )
            my_reduced_path.extend(list(np.asarray(self.paths)[idx]))
            my_reduced_audio_text.extend(list(np.asarray(self.audio_with_text)[idx]))
            idx = np.random.choice(np.arange(idx_last_first_genre+1, len(self.paths)), size=limit // 2,
                                   replace=False)
            my_reduced_path.extend(list(np.asarray(self.paths)[idx]))
            my_reduced_audio_text.extend(list(np.asarray(self.audio_with_text)[idx]))
            self.paths = my_reduced_path
            self.audio_with_text = my_reduced_audio_text

        shuffled_idx = np.arange(len(self.paths))
        np.random.shuffle(shuffled_idx)
        self.paths = list(np.asarray(self.paths)[shuffled_idx])
        self.audio_with_text = list(np.asarray(self.audio_with_text)[shuffled_idx])
        self.sr = sr
        self.duration = song_duration
        self.offset = offset

    def load_songs(self, n_fft=512, hop_length=160):
        self.n_fft = n_fft
        self.hop_length = hop_length
        audiowaves = []
        for song in self.paths:
            audiowave, _ = librosa.load(song, sr=self.sr, duration=self.duration,
                         offset=self.offset, mono=True)
            if self.use_log_spectrogram:
                # stft = librosa.core.stft(audiowave, hop_length=hop_length, n_fft=n_fft)
                # spectrogram = np.abs(stft)
                mel_log = librosa.feature.melspectrogram(y=audiowave, sr= self.sr,
                                                         n_fft=n_fft, hop_length=hop_length,
                                                        n_mels=64,
                                                        fmax=8000, fmin =50)
                # log_spectrogram = librosa.amplitude_to_db(spectrogram)
                audiowaves.append(librosa.power_to_db(mel_log))
            else:
                audiowaves.append(audiowave)

        if self.use_log_spectrogram:
            self.audiowaves = np.asarray(audiowaves)
        else:
            self.audiowaves = np.asarray(audiowaves)[:, None, :] # expand to 3d for convnet

    def save(self, path="./audiocontainer.pkl"):
        with open(path, "w") as f:
            pkl.dump(self, f)

# Adapted from https://data-flair.training/blogs/python-based-project-image-caption-generator-cnn/
class textContainer():
    def __init__(self, path_texts, myaudioContainer,
                 text_limit_per_song=None):
        assert type(myaudioContainer) == audioContainer
        assert type(path_texts) == list
        self.song_lyrics_path = []
        self.path_texts = []
        for path_text in path_texts:
            all_data = os.listdir(path_text)
            self.song_lyrics_path.extend([path_text+"/---"+text for text in all_data if "txt" in text])
        self.audio_with_text = myaudioContainer.audio_with_text
        self.audiowaves = myaudioContainer.audiowaves
        self.text_limit_per_song = text_limit_per_song
        self.myaudioContainer = myaudioContainer

    def _load_doc(self, file):
        # Opening the file as read only
        file = open(file, 'r')
        text = file.read()
        file.close()
        return text

    def _all_lyrics_for_audio(self):
        lyrics_for_audio = {}
        audio_with_text = self.audio_with_text

        text_no_space = [text.replace(" ","") for text in self.song_lyrics_path]
        list_audio_only_no_space = [audio.replace(" ", "") for audio in audio_with_text]
        path_text_to_use = []
        for audio_nospace in list_audio_only_no_space:
            for i, text in enumerate(self.song_lyrics_path):
                if (text_no_space[i].split("---")[1].split(".tx")[0] in audio_nospace) and (text_no_space[i].split("---")[1].split(".tx")[0] not in path_text_to_use):
                    path_text_to_use.append(text)
        # path_text_to_use = [text for i, text in enumerate(self.song_lyrics_path) for audio_nospace in list_audio_only_no_space if text_no_space[i].split("---")[1].split(".tx")[0] in audio_nospace]
        path_text_to_use = [text.replace("---", "") for text in path_text_to_use]
        self.path_text_to_use = path_text_to_use
        for filename, audio_data_path in zip(path_text_to_use, self.audio_with_text):
            file = self._load_doc(filename)
            lyrics = file.split('\n')[1:]
            for i, sentence in enumerate(lyrics):
                if sentence != "":
                    if audio_data_path not in lyrics_for_audio:
                        lyrics_for_audio[audio_data_path] = [sentence]
                    else:
                        lyrics_for_audio[audio_data_path].append(sentence)
                if self.text_limit_per_song is not None:
                    if i == self.text_limit_per_song:
                        break
        return lyrics_for_audio

    def _cleaning_text(self, lyrics):
        table = str.maketrans('', '', string.punctuation)
        for song, caps in lyrics.items():
            for i, song_lyric in enumerate(caps):
                song_lyric.replace("-", " ")
                desc = song_lyric.split()
                # converts to lowercase
                desc = [word.lower() for word in desc]
                # remove punctuation from each token
                desc = [word.translate(table) for word in desc]
                # remove hanging 's and a
                desc = [word for word in desc if (len(word) > 1)]
                # remove tokens with numbers in them
                desc = [word for word in desc if (word.isalpha())]
                # convert back to string
                song_lyric = ' '.join(desc)
                lyrics[song][i] = song_lyric
        return lyrics

    def _insert_tokens(self, clean_lyrics):
        for key, sentences in clean_lyrics.items():
            temp = []
            for sentence in sentences:
                temp_sentence = 'startseq ' + sentence + ' endseq'
                temp.append(temp_sentence)
            clean_lyrics[key] = temp
        return clean_lyrics

    def _text_vocabulary(self, clean_lyrics_with_token):
        # build vocabulary of all unique words
        vocab = set()
        for key in clean_lyrics_with_token.keys():
            [vocab.update(d.split()) for d in clean_lyrics_with_token[key]]
        return vocab

    def _dict_to_list(self, lyrics):
        all_lyrics = []
        for key in lyrics.keys():
            [all_lyrics.append(d) for d in lyrics[key]]
        return all_lyrics

    def _create_tokenizer(self, lyrics):
        lyrics_list = self._dict_to_list(lyrics)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lyrics_list)
        return tokenizer

    def _max_length(self, lyrics):
        lyrics_list = self._dict_to_list(lyrics)
        return max(len(d.split()) for d in lyrics_list)

    def _create_sequences(self, lyrics, audio_feature):
        X1, X2, y = list(), list(), list()
        if self.text_limit_per_song is not None:
            lyrics = lyrics[:self.text_limit_per_song]
        # walk through each lyrics
        count = 1
        for lyric in lyrics:
            print(f"Lyric count: {count}")
            count += 1
            print(lyric)
            # encode the sequence
            seq = self.tokenizer.texts_to_sequences([lyric])[0]
            # split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=self.max_len)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=self.vocab_length)[0]
                # store
                X1.append(audio_feature)
                X2.append(in_seq)
                y.append(out_seq)
        return np.array(X1), np.array(X2), np.array(y)

    def _create_sequences2(self, lyrics, audio_feature):
        # walk through each lyrics
        overall_dict = {}
        for keys, _ in audio_feature.items():
            count = 1
            # X1, X2, y = list(), list(), list()
            mini_dict = {"input":[], "output":[]}
            # mini_dict[audio_feature[keys]] = []
            print(f"Handling file {keys}")
            for lyric in lyrics[keys]:
                if self.text_limit_per_song is not None:
                    lyric = lyric[:self.text_limit_per_song]
                print(f"Lyric count: {count}")
                count += 1
                print(lyric)
                # encode the sequence
                seq = self.tokenizer.texts_to_sequences([lyric])[0]
                # split one sequence into multiple X,y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=self.max_len)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=self.vocab_length)[0]
                    # store
                    # X1.append(audio_feature[keys])
                    mini_dict["input"].append(in_seq)
                    mini_dict["output"].append(out_seq)
                    # X2.append(in_seq)
                    # y.append(out_seq)
            overall_dict[keys] = mini_dict
            # with open(f"./datasets/in_out_{keys}.pkl", "wb") as f:
            #     pkl.dump([X1, X2, y], f)
            # del X1
            # del X2
            # del y
        return overall_dict, audio_feature
        # return np.array(X1), np.array(X2), np.array(y)

    def run_datapipeline(self):
        print("Associating lyrics and audio")
        self.original_lyrics = self._all_lyrics_for_audio()
        print("Cleaning lyrics")
        lyrics = self._cleaning_text(self.original_lyrics)
        print("Inserting start and end token")
        lyrics = self._insert_tokens(lyrics)
        self.cleaned_lyrics = lyrics
        print("Building vocabulary")
        self.vocab = self._text_vocabulary(lyrics)
        self.vocab_length = len(self.vocab) + 1
        print("Creating tokens")
        self.tokenizer = self._create_tokenizer(lyrics)
        self.max_len = self._max_length(lyrics)

        audio = {}
        print("Creation audio dictionary")
        for i, key in enumerate(lyrics.keys()):
            if self.myaudioContainer.use_log_spectrogram:
                audio[key] = self.audiowaves[i, :]
            else:
                audio[key] = self.audiowaves[i, :].reshape(1, -1)

        list_input_audio = []
        list_input_text = []
        list_output_text = []
        self.end_index_songs = []
        tmp = 0
        print("Format data for input-output")
        # for key, _ in audio.items():
        #     print(f"Handling file {key}")
        # self.input_audio, self.input_text, self.output_text = \
        return self._create_sequences2(lyrics, audio)
            # list_input_audio.append(input_audio)
            # list_input_text.append(input_text)
            # list_output_text.append(output_text)
            # tmp += len(input_text)
            # self.end_index_songs.append(tmp)
        # self.input_audio = np.concatenate(list_input_audio)
        # self.input_text = np.concatenate(list_input_text)
        # self.output_text = np.concatenate(list_output_text)
        # return self.input_audio, self.input_text, self.output_text



class myDataset(torch.utils.data.Dataset):
    def __init__(self, audio, text, path="./datasets"):
        'Initialization'
        self.audio_dict = audio
        self.text_dict = text

        # if path is None:
        #     self.labels = labels
        #     self.audio = audio
        #     self.text = text
        # else:
        #     self.audio, self.text, self.labels = self._load_data(path)
        # self.path = path

    def __len__(self):
        'Denotes the total number of samples'
        counter = 0
        for key, _ in self.text_dict.items():
            counter += len(self.text_dict[key]["output"])
        return counter #len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        keys = list(self.text_dict.keys())[index]
        text_in = np.concatenate(self.text_dict[keys]["input"])
        text_out = np.concatenate(self.text_dict[keys]["output"])
        audio = self.audio_dict[keys]

        return audio, text_in, text_out

    def _load_data(self, path):
        myfiles = os.listdir(path)
        input_audio = []
        input_text = []
        output_text = []
        for file in myfiles:
            if "in_out" in file:
                print(file)
                path_temp = os.path.join(path, file)
                with open(path_temp, "rb") as f:
                    mylist = pkl.load(f)
                input_audio.append(mylist[0])
                input_text.append(np.asarray(mylist[1]))
                output_text.append(np.asarray(mylist[2]))
        return input_audio, input_text, output_text