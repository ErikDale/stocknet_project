import numpy as np
import os
import json
import re
import torch.nn as nn
import torch


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


def combine_tweets_prices():
    """
    Method that combines the tweets and prices that fall on the same day
    """
    price_values, targets = extract_prices()

    tweets, uniquewords = extract_tweets()

    keys = tweets.keys()

    # Combining the prices and tweets from the same dates
    # and putting them in one array
    combined = []

    for i in range(len(price_values)):
        for key in keys:
            if key == price_values[i][0][0]:
                keys_str = [str(j) for j in tweets[key]]
                values_str = [str(k) for k in price_values[i][0]]
                combined.append(keys_str + values_str)
                break

    # Removing date column
    [r.pop(20) for r in combined]

    target_combined = combined[5:]

    print(target_combined)

    targets = []

    for i in range(len(target_combined)):
        if float(target_combined[i][20]) > float(target_combined[i][23]):
            targets.append(0)
        else:
            targets.append(1)

    values = []

    for i in range(len(combined)):
        if i > len(combined) - 5:
            break
        value = []
        for j in range(i, i + 5):
            value.append(combined[j])

        values.append(value)

    values = values[:len(targets)]

    for i in range(len(values)):
        for j in range(len(values[i])):
            for k in range(len(values[i][j])):
                values[i][j][k] = float(values[i][j][k])
    return values, targets

def make_text_into_numbers(text, uniquewords):
    """
    Method that converts text to a list of numbers
    """
    iwords = text.lower().split(' ')
    numbers = []
    for n in iwords:
        try:
            numbers.append(uniquewords.index(n))
        except ValueError:
            numbers.append(0)
    numbers = numbers + [0,0,0,0,0]

    return numbers[:6]


def read_text_file(file_path, dict, file_name):
    """
    Method that reads the json object in the tweet file and puts them into a dictionary
    """
    array = []
    with open(file_path, 'r') as f:
        for jsonObj in f:
            if len(jsonObj) != 0:
                data = json.loads(jsonObj)
                # Reading only the text from the tweets excluding date and userid
                text = data["text"]
                text = ' '.join(text)
                array.append(text)
        dict[file_name] = ' '.join(array)


def extract_tweets():
    """
    Method that extracts the tweets
    :return: an array containing the different tweets
    """
    dict = {}
    # Folder Path
    path = "./stocknet-dataset-master/tweet/preprocessed/AAPL"
    # Read text File
    # iterate through all file
    for file in os.listdir(path):
        file_path = f"{path}/{file}"
        read_text_file(file_path, dict, file)
    # Removing non-alphanumeric character from the tweets
    for key in dict.keys():
        string = dict[key]
        string = re.sub(r'[^A-Za-z0-9 ]+', '', string)
        dict[key] = string

    allwords = ' '.join(dict.values()).lower().split(' ')
    uniquewords = list(set(allwords))
    for key in dict.keys():
        string = dict[key]
        vector = make_text_into_numbers(string, uniquewords)
        model = TweetLstmNet(len(uniquewords))
        vector = torch.unsqueeze(torch.LongTensor(vector), dim=0)
        hidden_vector = model(vector)
        dict[key] = hidden_vector
    return dict, uniquewords


def extract_prices():
    """Method that extracts the prices of the stocks"""
    cols = [0, 1, 2, 3, 4, 5]

    aapl = np.loadtxt("./stocknet-dataset-master/price/raw/AAPL.csv", delimiter=",", dtype=str, usecols=cols,
                      skiprows=1)

    target_prices = aapl[5:]

    targets = []

    for i in range(len(target_prices)):
        if float(target_prices[i][1]) > float(target_prices[i][4]):
            targets.append(0)
        else:
            targets.append(1)

    values = []

    for i in range(len(aapl)):
        if i > len(aapl) - 5:
            break
        value = []
        for j in range(i, i + 5):
            value.append(aapl[j])

        values.append(value)

    values = values[:len(targets)]
    return values, targets
