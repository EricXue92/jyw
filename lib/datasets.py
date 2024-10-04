import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from ucimlrepo import fetch_ucirepo


NUM_WORKERS = os.cpu_count()

def read_arrf(file_path):
    with open(file_path, encoding="utf-8") as f:
        header = []
        for line in f:
            if line.startswith("@attribute"):
                header.append(line.split()[1])
            elif line.startswith("@data"):
                break
        df = pd.read_csv(f, header=None)
        df.columns = header
    class_mapping = {2:0, 1:1}
    df['Class'] = df['Class'].map(class_mapping)

    # Convert dataframe to NumPy arrays
    if file_path.split('/')[1] == "Banana.arff":
        X = df.iloc[:, :2].to_numpy()
    elif file_path.split('/')[1] == "Twonorm.arff" or file_path.split('/')[1] == "Ring.arff":
        X = df.iloc[:, :20].to_numpy()
    else:
        print("Invalid dataname!")

    y = df['Class'].to_numpy()
    return X , y

# input: "Twonorm.arff",X.shape: (7400, 20) "Ring.arff", (7400, 20) "Banana.arff" (5300,2)
def pre_dataset(data):
    file_path = "lib/" + data
    X, y = read_arrf(file_path)

    input_dim = X.shape[1]
    num_classes = len(np.unique(y))

    # Assuming X and y are numpy arrays
    X_tensor = torch.tensor(X, dtype= torch.float32)  # Convert X to a tensor
    y_tensor = torch.tensor(y, dtype= torch.long)  # Convert y to a tensor (assuming classification labels)

    dataset = TensorDataset(X_tensor, y_tensor)

    dataset_size = len(dataset)

    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Extract training and test data for scaling
    X_train = torch.stack([x for x, _ in train_dataset]).numpy()
    X_test = torch.stack([x for x, _ in test_dataset]).numpy()

    # Fit scaler on training data and transform both train and test data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled data back to tensors
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                  torch.tensor([y for _, y in train_dataset], dtype=torch.long)
                                  )

    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32),
                                 torch.tensor([y for _, y in test_dataset], dtype=torch.long)
                                 )

    return input_dim, num_classes, train_dataset, test_dataset



# input: "spam" or "gamma"
def get_spam_or_gamma_dataset(data_name):
    # X.shape -> (4601, 58) and (19020, 10)
    if data_name == "spam":
        spambase = fetch_ucirepo(id=94)
        X, y = spambase.data.features.to_numpy(), spambase.data.targets.to_numpy()
    elif data_name == "gamma":
        magic_gamma_telescope = fetch_ucirepo(id=159)
        X, y = magic_gamma_telescope.data.features.to_numpy(), magic_gamma_telescope.data.targets
        y.loc[:, "class"] = y["class"].map({'g': 0, 'h': 1})  #y = y.apply(lambda x: 0 if x == 'g' else 1) #
        y = y.astype('float').to_numpy()
    else:
        print("Invalid data_name !")

    y = y.squeeze()  # arr_flat = arr.reshape(-1)
    num_classes = np.unique(y).size
    input_dim = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    return input_dim, num_classes, train_dataset, test_dataset


