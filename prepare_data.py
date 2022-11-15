import numpy as np
import os
import json
import re
import torch
from models import TweetLstmNet
from torch.utils.data import Dataset


# Dataset class
class CustomDataset(Dataset):
    def __init__(self, tweets, prices, targets, transform, target_tranform):
        tweets = np.array(tweets)
        prices = np.array(prices)
        targets = np.array(targets)
        self.tweets = torch.from_numpy(tweets)
        self.prices = torch.from_numpy(prices)
        self.targets = torch.from_numpy(targets)  # n_samples, 1
        self.n_samples = len(tweets)
        self.transform = transform
        self.target_transform = target_tranform

    def __getitem__(self, index):
        tweet_item = self.tweets[index]
        price_item = self.prices[index]
        target_item = self.targets[index]
        return tweet_item, price_item, target_item

    def __len__(self):
        return self.n_samples


def combine_tweets_prices():
    """
    Method that combines the tweets and prices that fall on the same day
    """
    # Extract the data
    price_values, targets = extract_prices()

    tweets, uniquewords = extract_tweets()

    keys = tweets.keys()

    # Combining the prices and tweets from the same dates
    # and putting them in one array
    combined = []

    for i in range(len(price_values)):
        for key in keys:
            if key == price_values[i][0][0]:
                # Removing date column
                price_values[i][0] = np.delete(price_values[i][0], 0)
                keys_str = [float(j) for j in tweets[key]]
                values_str = [float(k) for k in price_values[i][0]]
                combined.append([keys_str, values_str])
                break

    # Finding the targets
    target_combined = combined[5:]

    targets = []

    for i in range(len(target_combined)):
        if target_combined[i][1][0] > target_combined[i][1][3]:
            targets.append(0)
        else:
            targets.append(1)

    #values = []

    # Creating two list of five day windows
    # One for the tweets and one for the prices
    tweets = []
    prices = []

    for i in range(len(combined)):
        if i > len(combined) - 5:
            break
        #value = []
        tweet = []
        price = []
        for j in range(i, i + 5):
            # Normalizing prices (between 0 and 1)
            combined[j][1][0:5] = normalize(combined[j][1][0:5], 0, 1)
            tweet += combined[j][0]
            price.append([combined[j][1]])
            #value.append(combined[j])
        tweets.append(tweet)
        prices.append(price)
        #values.append(value)

    #values = values[:len(targets)]
    #print(values)
    tweets = tweets[:len(targets)]
    prices = prices[:len(targets)]
    return tweets, prices, targets, uniquewords


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
    # Converting each tweet to numbers
    for key in dict.keys():
        string = dict[key]
        vector = make_text_into_numbers(string, uniquewords)
        #model = TweetLstmNet(len(uniquewords))
        #vector = torch.unsqueeze(torch.LongTensor(vector), dim=0)
        #hidden_vector = model(vector)
        dict[key] = vector
    return dict, uniquewords


def normalize(arr, t_min, t_max):
    """
    Method that normalizes an array
    """
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def extract_prices():
    """Method that extracts the prices of the stocks"""
    cols = [0, 1, 2, 3, 4, 5]

    aapl = np.loadtxt("./stocknet-dataset-master/price/raw/AAPL.csv", delimiter=",", dtype=str, usecols=cols,
                      skiprows=1)

    target_prices = aapl[5:]

    targets = []

    # Creating the targets
    for i in range(len(target_prices)):
        if float(target_prices[i][1]) > float(target_prices[i][4]):
            targets.append(0)
        else:
            targets.append(1)

    values = []

    # Creating the values in five day windows
    for i in range(len(aapl)):
        if i > len(aapl) - 5:
            break
        value = []
        for j in range(i, i + 5):
            value.append(aapl[j])

        values.append(value)

    values = values[:len(targets)]
    return values, targets