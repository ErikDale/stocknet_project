import torch
import torch.nn as nn
from models import LstmNet
import matplotlib.pyplot as plt
import sklearn.metrics as skm


batch_size = 32

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

# Loading dataloader from file
test_dl = torch.load("./dataloaders/bigger_test_dl_32_batch.pt")

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
            # Add padding to the targets that are not correct size
            if targets.shape[0] != batch_size:
                iter = batch_size - targets.shape[0]
                for i in range(iter):
                    tensor = torch.LongTensor([0])
                    targets = torch.cat((targets, tensor))
            # Generate prediction
            prediction = model(torch.squeeze(tweet.type(torch.LongTensor)), price.type(torch.FloatTensor))

            # Calculating loss
            loss = criterion(torch.squeeze(prediction).type(torch.FloatTensor), targets.type(torch.FloatTensor))

            # Adding loss to list so it can be plotted later
            losses.append(loss.detach().numpy())

            for i in range(len(prediction)):
                # Predicted class value rounded
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
    """
    Method that plots the loss over epochs
    """
    x_values = list(range(len(losses)))

    # plotting the points
    plt.plot(x_values, losses)

    # x axis
    plt.xlabel('Epochs')
    # y axis
    plt.ylabel('Loss')

    plt.title('Test Loss graph')

    plt.show()


# Loading a model
model.load_state_dict(torch.load("./models/bigger_32_batch_mse.model"))

# Testing and plotting the loss
test_model()
plotLoss()