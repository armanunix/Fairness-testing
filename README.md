# The Implementation of DICE
### An information-theoritic tool to testing and debugging fairness defects in deep Neural Network

This repository contains the tools and evaluation experiments for the paper "Information-Theoretic Testing and Debugging of Fairness Defects in Deep Neural Networks," which was accepted for the technical track at ICSE'2023. It includes a Dockerfile for building the Docker script, a collection of necessary libraries for running the tool on a local machine, the source code of DICE, pre-built evaluation datasets, and scripts for re-running all search experiments.
# Docker File
```
```
# Tool
Our tool is composed of two phases: 1) The search phase employs clustering and gradients to identify and generate as many instances of discrimination as possible. 2) The debugging phase utilizes a layer localizer and causal analysis to pinpoint the source of discrimination within the deep neural network.
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
cd DICE/DICE_tutorial
python DICE_census.py
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
cd DICEDICE_baseline
```

The data of analysis can be found in:
```
cd DICE/results
```
There is one folder for each experiments that include DAIKE and the state-of-the-art outcomes.
