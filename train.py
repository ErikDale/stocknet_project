import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from prepare_data import combine_tweets_prices, CustomDataset
from models import LstmNet, LstmNet2

# Define relevant variables for the ML task
batch_size = 32
learning_rate = 0.001
num_epochs = 200

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

# Initialize the model
model = LstmNet2(len(unique_words))

# Initialize the loss and optimizer functions
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model():
    """Training the model"""
    for epoch in range(num_epochs):
        for batch_idx, (tweet, price, targets) in enumerate(train_dl):
            tweet = tweet.to(device=device)
            price = price.to(device=device)
            targets = targets.to(device=device)
            # Add padding to the targets
            if targets.shape[0] != batch_size:
                iter = batch_size - targets.shape[0]
                for i in range(iter):
                    tensor = torch.LongTensor([0])
                    targets = torch.cat((targets, tensor))
            scores = model(torch.squeeze(tweet.type(torch.LongTensor)), price.type(torch.FloatTensor))
            loss = criterion(torch.squeeze(scores).type(torch.FloatTensor), targets.type(torch.FloatTensor))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch % 25) == 0:
                print(loss.detach().numpy())

    # Saving the model
    path = './models/test.model'
    torch.save(model.state_dict(), path)
    print('Model saved as ' + path)


train_model()

