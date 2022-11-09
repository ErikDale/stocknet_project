import torch.nn as nn
import torch
import torch.nn.functional as F


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