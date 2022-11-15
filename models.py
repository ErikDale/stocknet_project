import torch.nn as nn
import torch
import torch.nn.functional as F

batch_size = 32

class LstmNet2(nn.Module):
    """
    Class that represents a model consisting of an embedding layer,
    two lstm layers and two linear layers
    """
    def __init__(self, vocab_length):
        super(LstmNet2, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_length, 10)

        # First lstm layer that the tweets are going to go through
        self.lstm1 = nn.LSTM(input_size=10,
                            hidden_size=20,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)
        # Second lstm layer that is going to classify the tweets and price input
        self.lstm2 = nn.LSTM(input_size=9,
                           hidden_size=1024,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=False)
        # Two linear layers
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 1)

    def forward(self, tweet_inp, price_inp):
        embed = self.embedding(tweet_inp)
        output, hidden = self.lstm1(embed)

        # Take the hidden layer from the first lstm layer as tweet input
        tweet_inp = hidden[0][0].cpu().detach()
        inp = torch.FloatTensor()
        price_inp = torch.squeeze(price_inp)
        if tweet_inp.shape[0] != batch_size:
            iter = batch_size - tweet_inp.shape[0]
            tensor = torch.LongTensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

            price_tensor = torch.FloatTensor([[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])
            for i in range(iter):
                tweet_inp = torch.cat((tweet_inp, tensor))
                price_inp = torch.cat((price_inp, price_tensor))

        tweet_inp = torch.reshape(tweet_inp, (batch_size, 5, 4))

        # Concatenate the tweet and price data
        torch.cat((torch.transpose(tweet_inp, 0, 2), torch.transpose(price_inp, 0, 2)), out=inp)

        # Transpose that input and feed it to the second lstm layer
        output, hidden = self.lstm2(torch.transpose(inp, 0, 2))
        x = self.fc1(output[:, -1, :])
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


class LstmNet(nn.Module):
    """
    Class that represents an lstm model
    """
    def __init__(self):
        super(LstmNet, self).__init__()
        self.lstm = nn.LSTM(input_size=25,
                           hidden_size=1024,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=False)
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 1)

    def forward(self, inp):
        output, hidden = self.lstm(inp)
        x = self.fc1(output[:, -1, :])
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 1)  # 2 classes

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


class TweetLstmNet(nn.Module):
    """
    Class that converts a tweet to a vector of numbers.
    """
    def __init__(self, vocab_length):
        super(TweetLstmNet, self).__init__()
        self.embedding = nn.Embedding(vocab_length, 10)
        self.lstm = nn.LSTM(input_size=10,
                           hidden_size=20,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=False)

    def forward(self, inp):
        embed = self.embedding(inp)
        output, hidden = self.lstm(embed)
        return hidden[0][0][0].cpu().detach().numpy()


class PriceLstmNet(nn.Module):
    """
    Class that converts the prices to a vector of numbers.
    """
    def __init__(self):
        super(PriceLstmNet, self).__init__()
        self.lstm = nn.LSTM(input_size=5,
                           hidden_size=20,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=False)

    def forward(self, inp):
        output, hidden = self.lstm(inp)
        return hidden[0][0][0].cpu().detach().numpy()