import os
import numpy as np
import re
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset

batch_size = 16


def combine_tweets_prices():
    """
    Method that combines the tweets and prices that fall on the same day
    """
    # Extract the data
    price_values = extract_prices()

    tweets, uniquewords = extract_tweets()

    # Combining the prices and tweets from the same dates
    # and putting them in one array
    combined = []

    for j in range(len(price_values)):
        keys = tweets[j].keys()

        dates = [row[0] for row in price_values[j]]

        new_prices = []

        for row in price_values[j]:
            new_prices.append(np.delete(row, 0))

        for i in range(len(price_values[j])):
            for key in keys:
                if key == dates[i]:
                    # Removing date column
                    #price_values[i] = np.delete(price_values[i], 0)
                    keys_float = [float(j) for j in tweets[j][key]]
                    values_float = [float(k) for k in new_prices[i]]
                    combined.append([keys_float, values_float])
                    break

    # Finding the targets
    target_combined = combined[5:]

    targets = []

    for i in range(1, len(target_combined)):
        if target_combined[i][1][4] > target_combined[i-1][1][4]:
            targets.append(0)
        else:
            targets.append(1)

    # values = []

    # Creating two list of five day windows
    # One for the tweets and one for the prices
    tweets = []
    prices = []

    for i in range(len(combined)):
        if i > len(combined) - 5:
            break
        # value = []
        tweet = []
        price = []
        for j in range(i, i + 5):
            # Normalizing prices (between 0 and 1)
            combined[j][1][0:5] = normalize(combined[j][1][0:5], 0, 1)
            tweet += combined[j][0]
            price.append([combined[j][1]])
            # value.append(combined[j])
        tweets.append(tweet)
        prices.append(price)
        # values.append(value)

    tweets = tweets[:len(targets)]
    prices = prices[:len(targets)]
    return tweets, prices, targets, uniquewords


def createDataset():
    """
    Method that creates the train and test dataloaders and returns them
    """
    tweets, prices, targets, unique_words = combine_tweets_prices()

    # Transform for values
    transform = transforms.Compose([
        transforms.ToTensor()
    ]
    )

    # Transform for targets
    target_transform = transforms.Compose([
        transforms.ToTensor()
    ]
    )

    # Initlaizes dataset
    dataset = CustomDataset(tweets, prices, targets, transform, target_transform)

    # Splits the dataset 80% train, 20% test
    train_ds, test_ds = random_split(dataset, [round(len(dataset) * 0.8), round(len(dataset) * 0.2)])

    # Creates dataloaders
    train_dl = DataLoader(dataset=train_ds, shuffle=True, batch_size=batch_size)
    test_dl = DataLoader(dataset=test_ds, shuffle=True, batch_size=batch_size)

    # Saves dataloaders to file
    torch.save(train_dl, "dataloaders/bigger_train_dl_16_batch.pt")
    torch.save(test_dl, "dataloaders/bigger_test_dl_16_batch.pt")

    # open file in write mode
   # with open(r'./dataloaders/more_unique_words.txt', 'w', encoding="utf-8") as fp:
         #for item in unique_words:
            # write each item on a new line
            #fp.write("%s\n" % item)

    return train_dl, test_dl, unique_words


# Dataset class
class CustomDataset(Dataset):
    """
    Dataset class
    """
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


def extract_prices():
    price_files = os.listdir("./stocknet-dataset-master/price/raw/")
    cols = [0, 1, 2, 3, 4, 5]

    prices = []
    for file in price_files:
        price_list = np.loadtxt("./stocknet-dataset-master/price/raw/" + file, delimiter=",", dtype=str, usecols=cols,
                          skiprows=1)
        prices.append(price_list)

    return prices


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
    numbers = numbers + [0, 0, 0, 0, 0]

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
    tweet_folders = os.listdir("./stocknet-dataset-master/tweet/preprocessed")

    all_unique_words = []


    dict_array = []

    for folder in tweet_folders:
        dict = {}
        path = "./stocknet-dataset-master/tweet/preprocessed/" + folder
        for file in os.listdir(path):
            file_path = f"{path}/{file}"
            read_text_file(file_path, dict, file)

        allwords = ' '.join(dict.values()).lower().split(' ')
        uniquewords = list(set(allwords))
        all_unique_words += uniquewords
        # Converting each tweet to numbers
        for key in dict.keys():
            string = dict[key]
            # Removing non-alphanumeric character from the tweets
            string = re.sub(r'[^A-Za-z0-9 ]+', '', string)
            dict[key] = string
            vector = make_text_into_numbers(string, uniquewords)
            # model = TweetLstmNet(len(uniquewords))
            # vector = torch.unsqueeze(torch.LongTensor(vector), dim=0)
            # hidden_vector = model(vector)
            dict[key] = vector
        dict_array.append(dict)
    return dict_array, all_unique_words


def normalize(arr, t_min, t_max):
    """
    Method that normalizes an array
    """
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr



