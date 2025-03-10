## Installation
Use the following command to install pyKT:

Create conda environment.

```
conda create --name=pykt python=3.7.5
source activate pykt
```


```
pip install -U pykt-toolkit -i  https://pypi.python.org/simple 

```

## Prepare a Dataset
Place the dataset in the correct directory:

```
data/{dataset_name}/
```
Ensure your dataset is correctly named.

Data preprocessing:

```
cd examples
python data_preprocess.py --dataset_name='dataset name'
```

## Train a Model
Train your model using:

```
python wandb_dkt_train.py --dataset_name='dataset name'
```

## Evaluate your Model
Locate your trained model under:
```
examples/saved_model/
```
Move the following files up one directory level:
```
config.json
qid_model.ckpt
```
To evaluate model performance on the test set, run:
```
python wandb_predict.py
```

## Prepare a Subset
Extract the prerequisite-dependent exercise relations using either method: 
```
dkt_method.py
new_method.py
```
Refer to the following files to help obtain a causal or random subset of the dataset.
```
2009_subset.ipynb
2012_subset.ipynb
2017_subset.ipynb
```
Use the same steps outlined above to preprocess your data, train the model, and evaluate it on your subset.

## pyKT

[![Downloads](https://pepy.tech/badge/pykt-toolkit)](https://pepy.tech/project/pykt-toolkit)
[![GitHub Issues](https://img.shields.io/github/issues/pykt-team/pykt-toolkit.svg)](https://github.com/pykt-team/pykt-toolkit/issues)
[![Documentation](https://img.shields.io/website/http/pykt-team.github.io/index.html?down_color=red&down_message=offline&up_message=online)](https://pykt.org/)

pyKT is a python library build upon PyTorch to train deep learning based knowledge tracing models. The library consists of a standardized set of integrated data preprocessing procedures on more than 7 popular datasets across different domains, 5 detailed prediction scenarios, more than 10 frequently compared DLKT approaches for transparent and extensive experiments. For more details about pyKT, please see [website](https://pykt.org/) and [docs](https://pykt-toolkit.readthedocs.io/en/latest/quick_start.html).
