import torch
from ..preprocessing.datacontainer import myDataset
import numpy as np
from keras.preprocessing.sequence import pad_sequences



class audioFeatureExtractor(torch.nn.Module):
    def __init__(self, vocab_size, filter_list, ks_list, stride_list, seq_len, max_len,
                 padding=1,
                 embedding_size=8, input_size=1,
                 latent_dim=256, n_operations=3):
        torch.nn.Module.__init__(self)
        self.max_len = max_len
        self.input_size = input_size
        self.word_embedding_dim = embedding_size
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.out_channels = filter_list
        self.ks_list = ks_list
        self.stride_list = stride_list
        self.sequence_length = seq_len
        self.n_operations = n_operations
        self.compute_window_size(seq_len)
        self.lstm_dim = 50

        # Handling Audio
        self.layer1 = torch.nn.Sequential(torch.nn.Conv1d(in_channels=input_size,
                                                          out_channels=filter_list[0],
                                                          kernel_size=ks_list[0],
                                                          stride=stride_list[0],
                                                          #  padding= 1,
                                                          ),

                                          torch.nn.BatchNorm1d(filter_list[0]),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool1d(ks_list[0],
                                                             #  padding=1
                                                             )
                                          )

        self.layer2 = torch.nn.Sequential(torch.nn.Conv1d(in_channels=filter_list[0],
                                                          out_channels=filter_list[1],
                                                          kernel_size=ks_list[1],
                                                          stride=stride_list[1]),
                                          torch.nn.BatchNorm1d(filter_list[1]),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool1d(ks_list[1])
                                          )

        self.layer3 = torch.nn.Sequential(torch.nn.Conv1d(in_channels=filter_list[1],
                                                          out_channels=filter_list[2],
                                                          kernel_size=ks_list[2],
                                                          stride=stride_list[2]),
                                          torch.nn.BatchNorm1d(filter_list[2]),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool1d(ks_list[2])
                                          )

        self.dense = torch.nn.Linear(in_features=filter_list[2] * self.Wn,
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

        # Common
        self.common_dense = torch.nn.Linear(in_features=latent_dim,
                                            out_features=1028)
        self.final_dense = torch.nn.Linear(1028, self.vocab_size)

    def forward(self, text, audio, cuda=False):
        text, audio = torch.tensor(text).type(torch.LongTensor), torch.tensor(audio)
        if cuda:
            text = text.cuda()
            audio = audio.cuda()
        if len(audio.shape) < 3:
            audio = audio.view(audio.size(0), 1, -1)
        audio_out1 = self.layer1(audio)
        audio_out2 = self.layer2(audio_out1)
        audio_out3 = self.layer3(audio_out2)
        audio_out4 = self.dense(audio_out3.view(audio.size(0), -1))

        text_out1 = self.embedding(text)
        pos = torch.arange(text.shape[-1])
        if cuda:
            pos = pos.cuda()
        pos_out = self.pos_embedding(pos)
        text_out2 = pos_out + text_out1
        h_0 = torch.zeros(1, audio.size(0), self.lstm_dim)
        c_0 = torch.zeros(1, audio.size(0), self.lstm_dim)  # latent_dim)
        h_1 = torch.zeros(1, audio.size(0), self.latent_dim)
        c_1 = torch.zeros(1, audio.size(0), self.latent_dim)
        if len(text_out2.shape) == 2:
            text_out2 = text_out2.unsqueeze(0)
        if cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
            h_1 = h_1.cuda()
            c_1 = c_1.cuda()

        text_out3, (h_out, c_out) = self.lstm1(text_out2, (h_0, c_0))
        text_out4, (h_out, c_out) = self.lstm2(text_out3, (h_1, c_1))

        added_output = audio_out4 + h_out.view(-1, self.latent_dim)

        prefinal = self.relu(self.common_dense(added_output))
        final = self.final_dense(prefinal)

        return final

    def compute_window_size(self, Wn):
        # Assuming stride = 1 and 3 CNN
        for i in range(self.n_operations):
            for j in range(2):
                if j == 0:
                    Wn = (Wn - self.ks_list[i]) // self.stride_list[i] + 1
                else:
                    Wn = (Wn - self.ks_list[i]) // self.ks_list[i] + 1
        self.Wn = Wn

    def traintextgen(self, input_audio, input_text, output_text,
              epochs=300, learning_rate=0.001, l2=0, opt="adam",
              bsize=32, momentum=0.999, shuffle=False, cuda=True):
        chosen_epoch = 100
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = l2
        self.opt = opt
        self.batch_size = bsize

        data = myDataset(input_audio, input_text, output_text)
        dataloader = torch.utils.data.DataLoader(data, batch_size=bsize, shuffle=shuffle)
        self.hist = np.zeros(epochs)
        criterion = torch.nn.functional.cross_entropy

        if cuda:
            self.cuda()
        print(f"Starting model training for {epochs} epochs")
        for epoch in range(epochs):
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
            for batch_audio, batch_text, batch_text_out in dataloader:
                optimizer.zero_grad()
                output_hat = self.forward(batch_text, batch_audio.reshape(len(batch_text), 1, -1), cuda=cuda)
                batch_text_out = torch.argmax(batch_text_out, axis=1)
                if cuda:
                    batch_text_out = batch_text_out.type(torch.LongTensor).cuda()
                else:
                    batch_text_out = batch_text_out.type(torch.LongTensor)
                loss = criterion(output_hat, batch_text_out)
                self.hist[epoch] = loss.item()
                hist_batch.append(self.hist[epoch])
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch #{epoch + 1}, loss={np.mean(hist_batch)}")

    def spitbars(self, token, input_audio, in_text="start", stop=None):

        for i in range(self.max_len):
            sequence = token.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=self.max_len)
            pred = self.forward(sequence, input_audio.reshape(1, 1, -1), cuda=True)
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
        with open(path, "w") as f:
            torch.save(self, f)
