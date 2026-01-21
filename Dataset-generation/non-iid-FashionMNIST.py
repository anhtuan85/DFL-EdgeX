import numpy as np
import torch
from torchvision import datasets, transforms
import os
def _dirichlet_partition_indices(y, n_clients=8, alpha=0.5, rng=None):
    """
    Label-wise Dirichlet partition on the FULL dataset.
    Returns: list of np.ndarray indices (one per client).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    y = np.asarray(y, dtype=np.int64)
    n_classes = int(y.max() + 1)
    all_idx = np.arange(len(y))
    class_idx = [all_idx[y == k] for k in range(n_classes)]

    client_idx = [[] for _ in range(n_clients)]

    for k in range(n_classes):
        idx_k = class_idx[k].copy()
        rng.shuffle(idx_k)

        p = rng.dirichlet(alpha * np.ones(n_clients))
        counts = (p * len(idx_k)).astype(int)

        # fix rounding so sum(counts) == len(idx_k)
        diff = len(idx_k) - counts.sum()
        if diff > 0:
            counts[rng.choice(n_clients, size=diff, replace=True)] += 1
        elif diff < 0:
            for _ in range(-diff):
                j = int(rng.integers(0, n_clients))
                while counts[j] == 0:
                    j = int(rng.integers(0, n_clients))
                    counts[j] -= 1

        start = 0
        for j in range(n_clients):
            end = start + counts[j]
            if end > start:
                client_idx[j].extend(idx_k[start:end].tolist())
            start = end

    return [np.array(ci, dtype=np.int64) for ci in client_idx]

def make_fashionmnist_dirichlet(
    root="./data",
    n_clients=8,
    alpha=0.5,
    train_ratio=0.8,        # 8/2 split inside each node
    seed=42,
    min_train=200,
    min_test=50,
    download=True,
    normalize_0_1=True,
):
    """
    1) Load Fashion-MNIST train+test, combine into one pool
    2) Dirichlet non-IID split across n_clients
    3) Within EACH client, split into train/test by train_ratio (8/2)

    Returns (ragged per client, dtype=object):
      x_train[i]: (n_i, 784)
      y_train[i]: (n_i,)
      x_test[i]:  (m_i, 784)
      y_test[i]:  (m_i,)
    """
    tr = datasets.FashionMNIST(root=root, train=True, download=download)
    te = datasets.FashionMNIST(root=root, train=False, download=download)

    X_all = np.concatenate([tr.data.numpy(), te.data.numpy()], axis=0)  # (70000, 28, 28) uint8
    y_all = np.concatenate([tr.targets.numpy(), te.targets.numpy()], axis=0).astype(np.int64)

    if normalize_0_1:
        X_all = X_all.astype(np.float32) / 255.0
    else:
        X_all = X_all.astype(np.uint8)

    X_all = X_all.reshape(len(X_all), -1)  # -> (70000, 784)

    rng = np.random.default_rng(seed)

    # Retry until every client has enough samples AND enough for train/test split
    min_total = min_train + min_test
    while True:
        client_indices = _dirichlet_partition_indices(y_all, n_clients=n_clients, alpha=alpha, rng=rng)

        ok = True
        for idx in client_indices:
            n = len(idx)
            n_tr = int(np.floor(train_ratio * n))
            n_te = n - n_tr
            if n < min_total or n_tr < min_train or n_te < min_test:
                ok = False
                break

        if ok:
            break

    x_train_list, y_train_list, x_test_list, y_test_list = [], [], [], []

    for idx in client_indices:
        idx = idx.copy()
        rng.shuffle(idx)

        n = len(idx)
        n_tr = int(np.floor(train_ratio * n))
        tr_idx = idx[:n_tr]
        te_idx = idx[n_tr:]

        x_train_list.append(X_all[tr_idx])
        y_train_list.append(y_all[tr_idx])
        x_test_list.append(X_all[te_idx])
        y_test_list.append(y_all[te_idx])

    x_train = np.array(x_train_list, dtype=object)
    y_train = np.array(y_train_list, dtype=object)
    x_test  = np.array(x_test_list,  dtype=object)
    y_test  = np.array(y_test_list,  dtype=object)

    return x_train, y_train, x_test, y_test

        
if __name__ == "__main__":
    num_clients = 8
    BASE_FOLDER = "./Non-IID-FashionMNIST/"
    alpha = 0.2
    if not os.path.exists(BASE_FOLDER):
        os.makedirs(BASE_FOLDER)
    x_train, y_train, x_test, y_test = make_fashionmnist_dirichlet(n_clients=num_clients, alpha=alpha, seed=1, train_ratio=0.8,
                                                                   min_train=200, min_test=50, normalize_0_1=True)
    
    for i in range(num_clients):
        node_data_path = os.path.join(BASE_FOLDER, str(i+1))
        if not os.path.exists(node_data_path):
            os.makedirs(node_data_path)
        
        X_train, X_test, Y_train, Y_test = x_train[i], x_test[i], y_train[i], y_test[i]

        np.save(os.path.join(node_data_path, "x_train.npy"), X_train)
        np.save(os.path.join(node_data_path, "x_test.npy"), X_test)
        np.save(os.path.join(node_data_path, "y_train.npy"), Y_train)
        np.save(os.path.join(node_data_path, "y_test.npy"), Y_test)