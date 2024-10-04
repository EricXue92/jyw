import os
import torch
import torch.nn as nn
from lib.datasets import get_spam_or_gamma_dataset, pre_dataset
from lib.helper_functions import accuracy_fn, plot_predictions, plot_decision_boundary
from torch.utils.data import DataLoader
from DeepResNet import DeepResNet
from SpectralNormResNet import SpectralNormResNet
from tqdm.auto import tqdm
# from sngp_wrapper.covert_utils import convert_to_sn_my
from lib.utils import set_seed

NUM_WORKERS = os.cpu_count()

# Count number of devices
# torch.cuda.device_count()

# y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
# go from logits -> prediction probabilities -> prediction labels

def train(model, data_loader, loss_fn, optimizer, accuracy_fn, device):
    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_logits  = model(X).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))  # turn logits -> pred probs -> pred labls(>0.5 1,  <0.5, 0 )
        loss = loss_fn(y_logits, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true = y,
                                 y_pred = y_pred )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test(model, data_loader, loss_fn,  accuracy_fn, device ):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
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
                                    y_pred = test_pred )

    test_loss /= len(data_loader)
    test_acc /= len(data_loader)

    print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": test_loss.item(),
            "model_acc": test_acc}

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

    #### spam or gamma
    #
    path = "../lib/"
    ds = get_spam_or_gamma_dataset(path + "gamma")

    #### input: "Twonorm.arff",X.shape: (7400, 20) "Ring.arff", (7400, 20) "Banana.arff" (5300,2)
    # ds =  pre_dataset("Twonorm.arff")

    # ds = get_banana_dataset(file_path = "../lib/Banana.arff")
    ds = get_twonorm_dataset(file_path = "../lib/Twonorm.arff")


    input_dim, num_classes, train_dataset, test_dataset = ds

    kwargs = {"num_workers": NUM_WORKERS, "pin_memory": True}
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True,
                                  drop_last = True, **kwargs )
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False,
                                 drop_last = True, **kwargs )

    ###  From Tao Wang
    # for testing the performance of normal model
    # model = DeepResNet(input_dim = input_dim, num_layers= 3, num_hidden= 128, activation = "relu",
    #                   num_outputs = 1, dropout_rate=0.1)
    # spec_norm_replace_list = ["Linear", "Conv2D"]
    # spec_norm_bound = 9. # 9.
    # # # Enforcing Spectral-Normalization on each layer
    # model = convert_to_sn_my(model, spec_norm_replace_list, spec_norm_bound)

    features = 128
    depth = 6
    coeff = 3. # 0.95
    dropout_rate = 0.01
    num_outputs = 1  # regression with 1D output and binary classification
    lr = 0.01 # 3e-3

    if SN_flag:
        # ResNet + SN
        model = SpectralNormResNet(input_dim = input_dim, features = features, depth = depth, spectral_normalization = True,
                                   coeff = coeff,  dropout_rate  = dropout_rate, num_outputs = num_outputs ) # 0.95
    else:
        # only ResNet
        model = DeepResNet(input_dim=input_dim, num_layers = depth, num_hidden = features, num_outputs = num_outputs,
                           dropout_rate  = dropout_rate )

    # nn.BCEWithLogitsLoss works with raw logits
    loss_fn = nn.BCEWithLogitsLoss()  # # BCEWithLogitsLoss = sigmoid built-in

    # loss_fn = nn.CrossEntropyLoss()
    # loss = loss_fn(torch.sigmoid(y_logits), y_train)  # Using nn.BCELoss you need torch.sigmoid()
    optimizer = torch.optim.AdamW(model.parameters(), lr= lr)
    epochs = 200

    for epoch in tqdm(range(epochs)):
        print(f"Epoch {epoch}\n-------------------------------")
        train(model, train_dataloader, loss_fn, optimizer, accuracy_fn, device)
        test(model, test_dataloader,loss_fn, accuracy_fn, device)
    print("Done!")

# Example usage
if __name__ == "__main__":
    seed = 23
    set_seed(seed)
    main()



