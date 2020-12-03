import torch
from ..preprocessing.datacontainer import myDataset
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from torch.nn import TransformerDecoderLayer
from torch.nn import LayerNorm
from torch.nn import TransformerDecoder
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

class audioFeatureExtractor(torch.nn.Module):
    def __init__(self, vocab_size, filter_list, ks_list, stride_list, seq_len, max_len,
                 nhead=8, input_channel=1, use_spectrogram=True,
                 embedding_size=8,
                 latent_dim=256, n_operations=4):
        torch.nn.Module.__init__(self)
        self.max_len = max_len
        self.input_channel = input_channel
        self.word_embedding_dim = embedding_size
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.out_channels = filter_list
        self.ks_list = ks_list
        self.stride_list = stride_list
        self.sequence_length = seq_len
        self.n_operations = n_operations
        self.nhead = nhead
        self.max_pool_filter = 2
        self.lstm_dim = 50

        # Handling Audio
        # if use_spectrogram:
        #     self.layer1 = ConvBlock(in_channels=1, out_channels=64)
        #     self.layer2 = ConvBlock(in_channels=64, out_channels=128)
        #     self.layer3 = ConvBlock(in_channels=128, out_channels=256)
        #     self.layer4 = ConvBlock(in_channels=256, out_channels=512)
        #     self.fc1 = nn.Linear(512, latent_dim, bias=True)
        # else:
        self.compute_window_size(seq_len)
        self.layer1 = torch.nn.Sequential(torch.nn.Conv1d(in_channels=input_channel,
                                                          out_channels=filter_list[0],
                                                          kernel_size=ks_list[0],
                                                          stride=stride_list[0],
                                                          #  padding= 1,
                                                          ),

                                          torch.nn.BatchNorm1d(filter_list[0]),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool1d(self.max_pool_filter,
                                                             #  padding=1
                                                             )
                                          )

        self.layer2 = torch.nn.Sequential(torch.nn.Conv1d(in_channels=filter_list[0],
                                                          out_channels=filter_list[1],
                                                          kernel_size=ks_list[1],
                                                          stride=stride_list[1]),
                                          torch.nn.BatchNorm1d(filter_list[1]),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool1d(self.max_pool_filter)
                                          )

        self.layer3 = torch.nn.Sequential(torch.nn.Conv1d(in_channels=filter_list[1],
                                                          out_channels=filter_list[2],
                                                          kernel_size=ks_list[2],
                                                          stride=stride_list[2]),
                                          torch.nn.BatchNorm1d(filter_list[2]),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool1d(self.max_pool_filter)
                                          )
        self.layer4 = torch.nn.Sequential(torch.nn.Conv1d(in_channels=filter_list[2],
                                                          out_channels=filter_list[3],
                                                          kernel_size=ks_list[3],
                                                          stride=stride_list[3]),
                                          torch.nn.BatchNorm1d(filter_list[3]),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool1d(self.max_pool_filter)
                                          )

        self.dense = torch.nn.Linear(in_features=filter_list[3] * self.Wn,
                                     out_features=latent_dim)
        self.relu = torch.nn.ReLU()

        # Text
        self.embedding = torch.nn.Embedding(self.vocab_size, self.word_embedding_dim)
        self.pos_embedding = torch.nn.Embedding(max_len, self.word_embedding_dim)
        self.lstm1 = torch.nn.LSTM(input_size=self.word_embedding_dim,
                                   hidden_size=self.lstm_dim,
                                   batch_first=True)

        self.lstm2 = torch.nn.LSTM(input_size=self.lstm_dim,
                                   hidden_size=self.latent_dim,
                                   batch_first=True)
        # Transformer decoder
        # self.transformer_decoder_layer = TransformerDecoderLayer(self.latent_dim, nhead=self.nhead)
        # decoder_norm = LayerNorm(self.latent_dim)
        # self.transformer_decoder = TransformerDecoder(self.transformer_decoder_layer, num_layers=6, norm=decoder_norm)
        # Common
        self.common_dense = torch.nn.Linear(in_features=latent_dim,
                                            out_features=1028)
        self.final_dense = torch.nn.Linear(1028, self.vocab_size)

    def forward(self, text, audio,
                cuda=False,
                use_spectrogram=False):
        text, audio = torch.tensor(text).type(torch.LongTensor), torch.tensor(audio)
        if cuda:
            text = text.cuda()
            audio = audio.cuda()
        if not use_spectrogram:
            if len(audio.shape) < 3:
                audio = audio.view(audio.size(0), 1, -1)
        # if use_spectrogram:
        #   audio = audio[:,None, :,:] # Adding 1 channel
        #   audio_out1 = self.layer1(audio)
        #   audio_out1 = F.dropout(audio_out1, p=0.2)
        #   audio_out2 = self.layer2(audio_out1)
        #   audio_out2 = F.dropout(audio_out2, p=0.2)
        #   audio_out3 = self.layer3(audio_out2)
        #   audio_out3 = F.dropout(audio_out3, p=0.2)
        #   audio_out4 = self.layer4(audio_out3)
        #   audio_out4 = F.dropout(audio_out4, p=0.2)
        #   audio_out4 = torch.mean(audio_out4, dim=3)
        #   audio_out4 = torch.max(audio_out4, dim=2)[0]
        #   audio_out5 = self.fc1(audio_out4)
        # else:
        audio_out1 = self.layer1(audio)
        audio_out2 = self.layer2(audio_out1)
        audio_out3 = self.layer3(audio_out2)
        audio_out4 = self.layer4(audio_out3)
        # print(audio_out3.shape)
        # print((audio.shape[1], self.Wn*self.out_channels[2]))
        audio_out5 = self.dense(audio_out4.view(audio.size(0), -1))

        text_out1 = self.embedding(text)
        pos = torch.arange(text.shape[-1])
        if cuda:
            pos = pos.cuda()

        pos_out = self.pos_embedding(pos)
        text_out2 = pos_out + text_out1
        h_0 = torch.zeros(1, text.size(0), self.lstm_dim)
        c_0 = torch.zeros(1, text.size(0), self.lstm_dim)  # latent_dim)
        h_1 = torch.zeros(1, text.size(0), self.latent_dim)
        c_1 = torch.zeros(1, text.size(0), self.latent_dim)
        if len(text_out2.shape) == 2:
            text_out2 = text_out2.unsqueeze(0)
        if cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
            h_1 = h_1.cuda()
            c_1 = c_1.cuda()

        text_out3, (h_out, c_out) = self.lstm1(text_out2, (h_0, c_0))
        text_out4, (h_out, c_out) = self.lstm2(text_out3, (h_1, c_1))

        added_output = audio_out5 + h_out.view(-1, self.latent_dim)

        prefinal = self.relu(self.common_dense(added_output))
        prefinal = F.dropout(prefinal, p=0.2)
        final = self.final_dense(prefinal)

        return final

    def compute_window_size(self, Wn):
        # Assuming stride = 1 and 3 CNN
        for i in range(self.n_operations):
            for j in range(2):
                if j == 0:
                    Wn = (Wn - self.ks_list[i]) // self.stride_list[i] + 1
                else:
                    Wn = (Wn - self.max_pool_filter) // self.max_pool_filter + 1
        self.Wn = Wn

    def traintextgen(self, input_audio, input_output_text,
                     epochs=300, learning_rate=0.0001, l2=0, opt="adam", use_spectrogram=False,
                     bsize=32, momentum=0.999, shuffle=False, cuda=True, limit=None, limit_songs=None):
        chosen_epoch = 100
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = l2
        self.opt = opt
        self.batch_size = bsize

        # data = myDataset(input_audio, input_output_text)
        # dataloader = torch.utils.data.DataLoader(data, batch_size=bsize, shuffle=shuffle)
        self.hist = np.zeros(epochs)
        criterion = torch.nn.functional.cross_entropy

        if cuda:
            self.cuda()
        else:
            self.cpu()
        print(f"Starting model training for {epochs} epochs")
        try:
            for epoch in tqdm(range(epochs)):
                if epoch == 0:
                    if opt == "adam":
                        optimizer = torch.optim.Adam(self.parameters(),
                                                     lr=learning_rate, weight_decay=l2)
                    else:
                        optimizer = torch.optim.SGD(self.parameters(),
                                                    lr=learning_rate, weight_decay=l2, momentum=momentum)
                elif epoch == chosen_epoch and chosen_epoch is not None:
                    learning_rate = learning_rate / 100
                    if opt == "adam":
                        optimizer = torch.optim.Adam(self.parameters(),
                                                     lr=learning_rate, weight_decay=l2)
                    else:
                        optimizer = torch.optim.SGD(self.parameters(),
                                                    lr=learning_rate, weight_decay=l2, momentum=momentum)
                hist_batch = []
                # for batch_audio, batch_text, batch_text_out in dataloader:
                for key in tqdm(input_audio.keys()):
                    batch_text_in = torch.tensor(np.asarray(input_output_text[key]["input"]))
                    batch_text_out = torch.tensor(np.asarray(input_output_text[key]["output"]))
                    batch_audio = torch.tensor(input_audio_dict[key])
                    if (limit_songs is not None) and (limit_songs == count_songs):
                        break
                    for l in np.arange(0, batch_text_in.shape[0], bsize):
                        # print(( batch_text_in.shape[0], l))
                        optimizer.zero_grad()
                        mylen = batch_text_in[l:l + bsize, :].shape[0]
                        tmp = torch.zeros((mylen, batch_audio.shape[0], batch_audio.shape[1]))
                        tmp[:mylen, :, :] = batch_audio
                        output_hat = self.forward(batch_text_in[l:l + bsize, :], tmp, cuda=cuda,
                                                  use_spectrogram=use_spectrogram)
                        out = torch.argmax(batch_text_out[l:l + bsize], axis=1)
                        if cuda:
                            out = out.type(torch.LongTensor).cuda()
                        else:
                            out = out.type(torch.LongTensor)
                        loss = criterion(output_hat, out)
                        self.hist[epoch] = loss.item()
                        hist_batch.append(self.hist[epoch])
                        loss.backward()
                        optimizer.step()
                        if (limit is not None) and (limit == count):
                            break
                if epoch % 10 == 0:
                    print(f"Epoch #{epoch + 1}, loss={np.mean(hist_batch)}")
        except KeyboardInterrupt:
            print("Keyboard interrupt exception caught")

    def spitbars(self, token, input_audio,
                 in_text="startseq", stop="endseq", cuda=True, use_spectrogram=False):

        for i in range(self.max_len):
            sequence = token.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=self.max_len)
            pred = self.forward(sequence, input_audio, cuda=cuda, use_spectrogram=use_spectrogram)
            pred = np.argmax(pred.cpu().detach().numpy())
            word = self.word_for_id(pred, token)
            if word is None:
                break
            in_text += ' ' + word
            if stop is None:
                continue
            else:
                if word == stop:
                    break

        return in_text

    def word_for_id(self, integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def savemodel(self, path="./audiotolyrics_model.pt"):
        with open(path, "wb") as f:
            torch.save(self, f)
