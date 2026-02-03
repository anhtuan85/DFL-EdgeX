# DFL-EdgeX
Design and Implementation of DFL Framework based on EdgeX platform in Digital Twin Network

## Installation

* Clone this repository
* Install requirements:
```
pip install -r requirements.txt
```
* Install EdgeX platform for edge nodes (Raspberry Pi or Jetson devices) from [link](https://docs.edgexfoundry.org/1.2/getting-started/quick-start) 
## Datasets

Check and run the source code in folder `Dataset-generation`

## Run experiment

1. Run the non-IID dataset generation in folder `Dataset-generation` and move each data part to each node

2. Move the DFLNode source code to edge node

2. Register the EdgeX profiles ([this link](./docs/register-edgex.md))

3. Run folder DFLManagment in your PC
```
python ./manage.py runserver 0.0.0.0:8000
```

<!-- 4. Run folder DFLNode in each edge node
```
python App.py
``` -->

## Register new device to Digital Twin
[Please refer to the documents.](./docs/)

