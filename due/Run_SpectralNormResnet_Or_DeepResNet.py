import os
import torch
import torch.nn as nn
from lib.datasets import pre_dataloader
from lib.helper_functions import accuracy_fn, plot_predictions, plot_decision_boundary
from torch.utils.data import DataLoader
from DeepResNet import DeepResNet
from SpectralNormResNet import SpectralNormResNet
from tqdm.auto import tqdm

from sngp_wrapper.covert_utils import convert_to_sn_my

from lib.utils import set_seed

# Turn data into tensors
# X = torch.from_numpy(X).type(torch.float)
# y = torch.from_numpy(y).type(torch.float)

# y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
# go from logits -> prediction probabilities -> prediction labels

def train(model, data_loader, loss_fn, optimizer, accuracy_fn, device):

    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()

    for batch, (X, y) in enumerate(data_loader):
        # send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_logits  = model(X).squeeze()

        y_pred = torch.round(torch.sigmoid(y_logits))  # turn logits -> pred probs -> pred labls(>0.5 1,  <0.5, 0 )

        # 2. Calculate loss
        loss = loss_fn(y_logits, y)
        train_loss += loss

        train_acc += accuracy_fn(y_true = y,
                                 y_pred = y_pred )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test(model, data_loader, loss_fn,  accuracy_fn, device ):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()

    #with torch.inference_mode():
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            test_logits = model(X).squeeze()

            test_pred = torch.round(torch.sigmoid(test_logits))

            test_loss += loss_fn(test_logits, y)

            # y_pred = test_logits.argmax(dim=1)
            # correct = torch.eq(y_true, y_pred).sum().item()
            # acc = (correct / len(y_pred)) * 100
            test_acc += accuracy_fn(y_true = y,
                                    y_pred = test_pred ) # # test_pred.argmax(dim=1) -> Go from logits -> pred labels

    # Adjust metrics and print out
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)

    print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")


def main(SN_flag = False):
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    batch_size = 128

    input_dim, num_classes, train_dataloader, test_dataloader = pre_dataloader( batch_size = batch_size )

    model = DeepResNet(input_dim = input_dim, num_classes= 1, num_layers= 3,
                        num_hidden= 128, activation = "relu", dropout_rate=0.1)

    if SN_flag:
        ##  From Tao Wang
        spec_norm_replace_list = ["Linear", "Conv2D"]
        spec_norm_bound = 10. # 9.
        # # Enforcing Spectral-Normalization on each layer
        model = convert_to_sn_my(model, spec_norm_replace_list, spec_norm_bound)

    # From Inducing points
    # sn_model = SpectralNormResNet(input_dim = input_dim, features = 128,
    #                               depth = 3, num_outputs = 1, spectral_normalization = True)

    #loss_fn = nn.CrossEntropyLoss()

    # nn.BCEWithLogitsLoss works with raw logits
    loss_fn = nn.BCEWithLogitsLoss()  # # BCEWithLogitsLoss = sigmoid built-in

    # loss = loss_fn(torch.sigmoid(y_logits), y_train)  # Using nn.BCELoss you need torch.sigmoid()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    epochs = 20

    for epoch in tqdm(range(epochs)):
        print(f"Epoch {epoch}\n-------------------------------")
        train(model, train_dataloader, loss_fn, optimizer, accuracy_fn, device)
        test(model, test_dataloader,loss_fn, accuracy_fn, device)
    print("Done!")

# Example usage
if __name__ == "__main__":
    seed = 23
    set_seed(seed)
    main(True)



