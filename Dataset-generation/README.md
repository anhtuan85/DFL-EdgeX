# Generate non-IID data by Dirichlet distribution
This source code will generate non-IID data by Dirichlet distribution. After run this source code, you cand move each data part to each edge node

## Fashion-MNIST
Open file `non-iid-FashionMNIST.py` and change the hyperparameter for your experiment

    num_clients = 8
    BASE_FOLDER = "./Non-IID-FashionMNIST/"
    alpha = 0.2
Run the code `non-iid-FashionMNIST.py`
```
python non-iid-FashionMNIST.py
```

## CIFAR-10
Open file `non-iid-CIFAR10.py` and change the hyperparameter for your experiment

    num_clients = 8
    BASE_FOLDER = "./Non-IID-CIFAR/"
    alpha = 0.2

Run the code `non-iid-CIFAR10.py`
```
python non-iid-CIFAR10.py
```

## FEMNIST
Download the dataset from [this link](https://github.com/TalwalkarLab/leaf)

Open file `non-iid-FEMNIST.py` and choose some list user files from this dataset
```
    num_clients = 8
    BASE_FOLDER = "./Non-IID-FEMNIST/"
    split_data = 0.8
    alpha = 0.2
    list_paths = ["./data/femnist/data/all_data/all_data_0.json",
             "./data/femnist/data/all_data/all_data_1.json",
             "./data/femnist/data/all_data/all_data_2.json"]
```

Run the code:
```
python non-iid-FEMNIST.py
```
