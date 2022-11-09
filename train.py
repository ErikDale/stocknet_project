import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from prepare_data import combine_tweets_prices, CustomDataset
from models import LstmNet

# Define relevant variables for the ML task
batch_size = 16
learning_rate = 0.003
num_epochs = 100

# values, targets = combine_tweets_prices()

values, targets = combine_tweets_prices()


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
dataset = CustomDataset(values, targets, transform, target_transform)
# print(dataset.__len__())
# print(int(len(dataset) * 0.8))
# print(int(len(dataset) * 0.2))

# Splits the dataset 80% train, 20% test
train_ds, test_ds = random_split(dataset, [round(len(dataset) * 0.8), round(len(dataset) * 0.2)])

# Creates dataloaders
train_dl = DataLoader(dataset=train_ds, shuffle=True, batch_size=batch_size)
test_dl = DataLoader(dataset=test_ds, shuffle=True, batch_size=batch_size)

model = LstmNet()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model():
    """Training the model"""
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_dl):
            data = data.to(device=device)
            targets = targets.to(device=device)
            scores = model(data.type(torch.FloatTensor))
            loss = criterion(scores, targets.view(len(targets), 1).type(torch.FloatTensor))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch % 25) == 0:
                print(loss.detach().numpy())


# os.chdir("../../")
# path = './relu_sigmoid.model'
# torch.save(model.state_dict(), path)
# print('Model saved as ' + path)


def test_model():
    """Testing the model"""
    with torch.no_grad():
        num_correct = 0
        fp = 0
        counter = 0
        for batch_idx, (data, targets) in enumerate(test_dl):
            # Generate prediction
            prediction = model(data.type(torch.FloatTensor))

            for i in range(len(prediction)):
                counter += 1
                # Predicted class value round()
                predicted_class = round(float(prediction[i]))
                actual_target = targets[i]

                print(
                    f'Prediction: {prediction[i]}, Predicted class {predicted_class} - Actual target: {actual_target}')

                if predicted_class == actual_target:
                    num_correct += 1
                elif (predicted_class == 1 and actual_target == 0) or (
                        predicted_class == 0 and actual_target == 1):
                    fp += 1

        print("Accuracy: ", (num_correct / counter) * 100)


train_model()
test_model()
