import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from prepare_data import combine_tweets_prices, CustomDataset
from models import LstmNet
import matplotlib.pyplot as plt
import sklearn.metrics as skm


# Define relevant variables for the ML task
batch_size = 32

# train_dl, test_dl, unique_words = createDataset()
unique_words = []
# open file and read the content in a list
with open(r'./dataloaders/unique_words.txt', 'r') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        unique_words.append(x)


test_dl = torch.load("./dataloaders/test_dl_32_batch.pt")

# Initialize the model
model = LstmNet(len(unique_words))

criterion = nn.BCELoss()

losses = []

classes = [0, 1]


def test_model():
    """
    Testing the model
    """
    predictions = []
    ground_truth = []
    with torch.no_grad():
        for batch_idx, (tweet, price, targets) in enumerate(test_dl):
            ## Add padding to the targets
            if targets.shape[0] != batch_size:
                iter = batch_size - targets.shape[0]
                for i in range(iter):
                    tensor = torch.LongTensor([0])
                    targets = torch.cat((targets, tensor))
            # Generate prediction
            prediction = model(torch.squeeze(tweet.type(torch.LongTensor)), price.type(torch.FloatTensor))
            loss = criterion(torch.squeeze(prediction).type(torch.FloatTensor), targets.type(torch.FloatTensor))
            losses.append(loss.detach().numpy())

            for i in range(len(prediction)):
                # Predicted class value round()
                predicted_class = round(float(prediction[i]))
                predictions.append(predicted_class)

                actual_target = targets[i]
                ground_truth.append(int(actual_target))

                print(
                    f'Prediction: {prediction[i]}, Predicted class {predicted_class} - Actual target: {actual_target}')

        # Plotting confusion matrix and printing accuracy, f1, precision and recall
        cm = skm.confusion_matrix(y_true=ground_truth, y_pred=predictions, labels=classes)
        disp = skm.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot()
        plt.show()
        print("Sk learn Accuracy: ", skm.accuracy_score(y_true=ground_truth, y_pred=predictions))
        print("Sk learn F1: ", skm.f1_score(y_true=ground_truth, y_pred=predictions, labels=classes))
        print("Sk learn precision: ", skm.precision_score(y_true=ground_truth, y_pred=predictions, labels=classes))
        print("Sk learn recall: ", skm.recall_score(y_true=ground_truth, y_pred=predictions, labels=classes))


def plotLoss():
    x_values = list(range(len(losses)))

    # plotting the points
    plt.plot(x_values, losses)

    # naming the x axis
    plt.xlabel('Epochs')
    # naming the y axis
    plt.ylabel('Loss')

    plt.title('Test Loss graph')

    # function to show the plot
    plt.show()


# Loading a model
model.load_state_dict(torch.load("./models/16_batch_bce.model"))

test_model()
plotLoss()