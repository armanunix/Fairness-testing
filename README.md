# The Implementation of DICE
### An information-theoritic tool to testing and debugging fairness defects in deep Neural Network

This repository provides the tool and the evaluation Experimrnts for the paper "Information-Theoretic Testing and Debugging of
Fairness Defects in Deep Neural Networks" accepted for the technical track at ICSE'2023.

The repository includes:

a Dockerfile to build the Docker script,
a set of required libraries for running the tool on local machine,
the source code of DICE,
the pre-built evaluation all results: Dataset, and
the scripts to rerun all search experiments: scripts.
# Docker File
```
```
# Tool
Our tool consists of two steps: 1) the search phase uses clustering and gradients to maximize the
amounts of discrimination and generate as many discrimination instance as possible 2) the debugging
phase that uses a layer localizer and causal analysis to pinpoint the root cause of discrimination
in the internal of deep neural network.
# Requirements
Python 3.8
numpy==1.22.0.
pandas==1.5.1
tensorflow==2.7.0
scipy==1.4.1
argparse==1.1
protobuf==3.9.2
scikit-learn==1.1.3
aif360==0.4.0
IPython==7.13.0
regex
# How to setup DICE
If you use the pre-built Docker image, the tool is already built to use in Ubuntu base. Otherwise, the installation of required packlages and libraries should be sufficient to run Parfait-ML. Note: the tool is substantially tested in Linux system.
# Getting Started with an example
After succesfully setup DICE, you can try a simple example to check the basic functionality of our tool. We prepared a simple run script for census dataset with sex, race, and age as sensitive attributes

To run the search algorithm over Census dataset:
```
cd DAIKE/DAIKE_tutorial
python daike_census.py
```
The other models for a different dataset can be run in the same fashion.

To compue the amounts of discrimination and characterize entropy, run
```
python entropy_census.py
```

To run the debugging algorithm over Census dataset:
```
python RQ3_census.py
```

The baseline tools can be found inside DAIKE baseline:
```
cd DAIKE/DAIKE_baseline
```

The data of analysis can be found in:
```
cd DAIKE/results
```
There is one folder for each experiments that include DAIKE and the state-of-the-art outcomes.
