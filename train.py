import torch
import torch.nn as nn
from models import LstmNet
import matplotlib.pyplot as plt


# Define relevant variables for the ML task
batch_size = 32
learning_rate = 0.001
num_epochs = 200


# Getting unique words from file
unique_words = []
# open file and read the content in a list
with open(r'./dataloaders/more_unique_words.txt', 'r', encoding='utf-8') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        unique_words.append(x)

# Fetching dataloader from file
train_dl = torch.load("./dataloaders/bigger_train_dl_32_batch.pt")


# Initialize the model
model = LstmNet(len(unique_words))

# Initialize the loss and optimizer functions
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

losses = []


def train_model():
    """
    Training the model
    """
    loss = None
    for epoch in range(num_epochs):
        for batch_idx, (tweet, price, targets) in enumerate(train_dl):
            tweet = tweet.to(device=device)
            price = price.to(device=device)
            targets = targets.to(device=device)

            # Add padding to the targets that are not correct size
            if targets.shape[0] != batch_size:
                iter = batch_size - targets.shape[0]
                for i in range(iter):
                    tensor = torch.LongTensor([0])
                    targets = torch.cat((targets, tensor))

            # Getting predictions from model
            scores = model(torch.squeeze(tweet.type(torch.LongTensor)), price.type(torch.FloatTensor))

            # Calculating loss
            loss = criterion(torch.squeeze(scores).type(torch.FloatTensor), targets.type(torch.FloatTensor))

            # Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch % 25) == 0:
                print(loss.detach().numpy())
        # Adding loss to list so it can be plotted later
        losses.append(loss.detach().numpy())

    # Saving the model
    path = 'models/bigger_32_batch_mse.model'
    torch.save(model.state_dict(), path)
    print('Model saved as ' + path)


def plotLoss():
    """
    Method that plots the loss over epochs
    """
    x_values = list(range(len(losses)))

    # plotting the points
    plt.plot(x_values, losses)

    # naming the x axis
    plt.xlabel('Epochs')
    # naming the y axis
    plt.ylabel('Loss')

    plt.title('Train Loss graph')

    plt.show()

# Traning the model and plotting the loss
train_model()
plotLoss()

