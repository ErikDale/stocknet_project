import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from prepare_data import combine_tweets_prices, CustomDataset
from models import LstmNet2


# Define relevant variables for the ML task
batch_size = 16

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


def test_model():
    """Testing the model"""
    with torch.no_grad():
        num_correct = 0
        fp = 0
        counter = 0
        for batch_idx, (tweet, price, targets) in enumerate(test_dl):
            ## Add padding to the targets
            if targets.shape[0] != batch_size:
                iter = batch_size - targets.shape[0]
                for i in range(iter):
                    tensor = torch.LongTensor([0])
                    targets = torch.cat((targets, tensor))
            # Generate prediction
            prediction = model(torch.squeeze(tweet.type(torch.LongTensor)), price.type(torch.FloatTensor))

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


# Loading a model
model.load_state_dict(torch.load("./models/test.model"))

test_model()