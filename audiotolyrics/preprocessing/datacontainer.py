import torch
import librosa
import pandas as pd
import numpy as np
import os
from keras.utils import to_categorical
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class audioContainer():
    def __init__(self, path_songs, sr, song_duration=15, offset=60):
        self.paths = path_songs
        self.sr = sr
        self.duration = song_duration
        self.offset = offset
    def load_songs(self):
        audiowaves = []
        for song in self.paths:
            audiowave, = librosa.load(song, sr=self.sr, duration=self.duration,
                         offset=self.offset, mono=True)
            audiowaves.append(audiowave)

        self.audiowaves = np.asarray(audiowaves)

# Adapted from https://data-flair.training/blogs/python-based-project-image-caption-generator-cnn/
class textContainer():
    def __init__(self, path_texts, audio_data_paths, audiowaves):
        all_data = os.listdir(path_texts)
        self.path = [text for text in all_data if "txt" in text]
        self.audio_data_paths = audio_data_paths
        self.audiowaves = audiowaves

    def _load_doc(self):
        # Opening the file as read only
        file = open(self.paths, 'r')
        text = file.read()
        file.close()
        return text

    def _all_lyrics_for_audio(self):
        lyrics_for_audio = {}
        audio_data_paths = self.audio_data_paths
        list_of_file = os.listdir(audio_data_paths)
        list_audio_only = [audio for audio in list_of_file if "txt" not in audio]
        list_audio_only_no_space = [audio.replace("") for audio in list_audio_only]
        text_no_space = [text.replace("") for text in self.paths]

        audio_with_text = [audio for audio in list_audio_only_no_space for text in text_no_space if audio in text]
        idx_audio_text = [i for i,audio in enumerate(list_audio_only) if audio in audio_with_text]

        audio_data_paths = list(np.asarray(audio_with_text)[idx_audio_text])
        for filename, audio_data_path in zip(self.paths, audio_data_paths):
            file = self._load_doc(filename)
            lyrics = file.split('\n')[1:]
            for sentence in lyrics:
                if sentence != "":
                    if audio_data_path not in lyrics_for_audio:
                        lyrics_for_audio[audio_data_path] = [sentence]
                    else:
                        lyrics_for_audio[audio_data_path].append(sentence)
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
                temp_sentence = '<start> ' + sentence + ' <end>'
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
        # walk through each description for the image
        for lyric in lyrics:
            # encode the sequence
            seq = self.tokenizer.texts_to_sequences([lyric])[0]
            # split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=self.max_len)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq - 1], num_classes=self.vocab_length)[0]
                # store
                X1.append(audio_feature)
                X2.append(in_seq)
                y.append(out_seq)
        return np.array(X1), np.array(X2), np.array(y)

    def run_datapipeline(self):
        lyrics = self._all_lyrics_for_audio()
        lyrics = self._cleaning_text(lyrics)
        lyrics = self._insert_tokens(lyrics)
        self.cleaned_lyrics = lyrics
        self.vocab = self._text_vocabulary(lyrics)
        self.vocab_length = len(self.vocab)
        self.tokenizer = self._create_tokenizer(lyrics)
        self.max_len = self._max_length(lyrics)

        audio = {}
        for i, key in enumerate(lyrics.keys()):
            audio[key] = self.audiowaves[i, :].reshape(1, -1)

        list_input_audio = []
        list_input_text = []
        list_output_text = []
        for key, _ in audio.items():
            input_audio, input_text, output_text = self._create_sequences(lyrics[key], audio[key])
            list_input_audio.append(input_audio)
            list_input_text.append(input_text)
            list_output_text.append(output_text)
        self.input_audio = np.concatenate(list_input_audio)
        self.input_text = np.concatenate(list_input_text)
        self.output_text = np.concatenate(list_output_text)
        return self.input_audio, self.input_text, self.output_text



class myDataset(torch.utils.data.Dataset):
  def __init__(self, audio, text, labels):
        'Initialization'
        self.labels = labels
        self.audio = audio
        self.text = text

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

  def __getitem__(self, index):
        'Generates one sample of data'
        return self.audio[index], self.text[index], self.labels[index]