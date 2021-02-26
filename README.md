<div align="center">    

# Equivariant-GNNs

<!--
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
-->

<!--
ARXIV
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->

![CI testing](https://github.com/amorehead/Equivariant-GNNs/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--
Conference
-->
</div>

## Description

An environment for running experiments with equivariant GNN architectures

## How to run

First, install and configure Conda environment:

```bash
# Clone project:
git clone https://github.com/amorehead/Equivariant-GNNs

# Change to project directory:
cd Equivariant-GNNs

# (If on HPC cluster) Load 'open-ce' module
module load open-ce/0.1-0

# (If on HPC cluster) Clone Conda environment into this directory using provided 'open-ce' environment:
conda create --prefix ./venv --clone open-ce-0.1-0

# (If on HPC cluster - Optional) Create Conda environment in a particular directory using provided 'open-ce' environment:
conda create --prefix MY_VENV_DIR --clone open-ce-0.1-0

# (Else, if on local machine) Set up Conda environment locally
conda env create --prefix ./venv -f environment.yml

# (Else, if on local machine - Optional) Create Conda environment in a particular directory using local 'environment.yml' file:
conda env create --prefix MY-VENV-DIR -f environment.yml

# Activate Conda environment located in the current directory:
conda activate ./venv

# (Optional) Activate Conda environment located in another directory:
conda activate MY-VENV-DIR

# (Optional) Deactivate the currently-activated Conda environment:
conda deactivate

# (If on local machine - Optional) Perform a full update on the Conda environment described in 'environment.yml':
conda env update -f environment.yml --prune

# (Optional) To remove this long prefix in your shell prompt, modify the env_prompt setting in your .condarc file with:
conda config --set env_prompt '({name})'
 ```

(If on HPC cluster) Install all project dependencies:

```bash
# Install project as a pip dependency in the Conda environment currently activated:
pip3 install -e .

# Install external pip dependencies in the Conda environment currently activated:
pip3 install -r requirements.txt

# Install pip dependencies used for unit testing in the Conda environment currently activated:
pip3 install -r tests/requirements.txt
 ```

Configure Weights and Biases (Wandb) to point to project directory for config details:

```bash
export WANDB_CONFIG_DIR=.
 ```   

Then, navigate to any file and run it:

 ```bash
# Run a particular module (example: Equivariant-GNNs architecture as main contribution):
python3 project/lit_set.py
```

## Imports

This project is setup as a package which means you can now easily import any file into any other file like so:

```python
from project.datasets.RG.rg_dgl_data_module import RGDGLDataModule
from project.lit_set import LitSET
from pytorch_lightning import Trainer

# Model
model = LitSET()  # Provide model parameters here

# Data
data_dir = 'final'  # Specify data directory here
data_module = RGDGLDataModule(data_dir)
data_module.setup()

# Train
trainer = Trainer()
trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())

# Test using the best model!
trainer.test(test_dataloaders=data_module.test_dataloader())
```

### Citation

```
@article{Equivariant-GNNs,
  title={Equivariant-GNNs: An environment for running experiments with equivariant GNN architectures},
  author={Morehead, Alex, Chen, Chen, and Cheng, Jianlin},
  journal={N/A},
  year={2021}
}
```
