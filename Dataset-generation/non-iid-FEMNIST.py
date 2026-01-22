import numpy as np
import json
import matplotlib.pyplot as plt
import os
import random
from collections import defaultdict
from numpy.random import dirichlet

def make_femnist_dirichlet(list_paths= [], n_clients=8, alpha = 0.5):

    users_list = []
    users_data = dict()
    list_data = []
    list_label = []

    for dir_path in list_paths:
        with open(dir_path) as f:
            data = json.load(f)
        users_list += data["users"]
        users_data.update(data["user_data"])

    for idx in users_list:
        list_data += users_data[idx]["x"]
        list_label += users_data[idx]["y"]

    x = np.array(list_data)
    y = np.array(list_label)
    x = np.reshape(x, (len(x), 28, 28))

    unique_classes = np.unique(y)
    client_data = defaultdict(list)

    for cls in unique_classes:
        # Indices of samples belonging to this class
        class_indices = np.where(y == cls)[0]
        
        # Dirichlet distribution for the class
        dirichlet_weights = dirichlet(alpha * np.ones(n_clients))
        
        # Shuffle class indices and compute split sizes
        np.random.shuffle(class_indices)
        split_sizes = (dirichlet_weights * len(class_indices)).astype(int)
        
        # Ensure all samples are allocated
        split_sizes[-1] += len(class_indices) - sum(split_sizes)
        
        # Assign samples to each client
        start = 0
        for client_id, size in enumerate(split_sizes):
            client_data[client_id].extend(class_indices[start:start + size])
            start += size

    # Shuffle each client's data
    for client_id in client_data:
        np.random.shuffle(client_data[client_id])

    #create final splits
    x_splits = [x[client_data[client_id]] for client_id in range(n_clients)]
    y_splits = [y[client_data[client_id]] for client_id in range(n_clients)]

    return x_splits, y_splits

if __name__ == "__main__":
    num_clients = 8
    BASE_FOLDER = "./Non-IID-FEMNIST/"
    split_data = 0.8
    alpha = 0.2
    list_paths = ["./data/femnist/data/all_data/all_data_0.json",
             "./data/femnist/data/all_data/all_data_1.json",
             "./data/femnist/data/all_data/all_data_2.json"]
    if not os.path.exists(BASE_FOLDER):
        os.makedirs(BASE_FOLDER)
    x_splits, y_splits = make_femnist_dirichlet( list_paths=list_paths, n_clients= num_clients, alpha=alpha)

    for i in range(num_clients):
        node_data_path = os.path.join(BASE_FOLDER, str(i+1))
        if not os.path.exists(node_data_path):
            os.makedirs(node_data_path)
        data = x_splits[i]
        labels = y_splits[i]
        split_point = int(len(labels)*split_data)
        X_train = data[:split_point]
        Y_train = labels[:split_point]
        X_test = data[split_point:]
        Y_test = labels[split_point:]
        np.save(os.path.join(node_data_path, "x_train.npy"), X_train)
        np.save(os.path.join(node_data_path, "x_test.npy"), X_test)
        np.save(os.path.join(node_data_path, "y_train.npy"), Y_train)
        np.save(os.path.join(node_data_path, "y_test.npy"), Y_test)