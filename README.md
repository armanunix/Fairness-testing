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
numpy==1.22.0 </br>
pandas==1.5.1 </br>
tensorflow==2.7.0 </br>
scipy==1.4.1 </br>
argparse==1.1 </br>
protobuf==3.9.2 </br>
scikit-learn==1.1.3 </br>
aif360==0.4.0 </br>
IPython==7.13.0 </br>
regex </br>
# How to setup DICE
Our tool, Parfait-ML, is pre-built in a Debian-based Docker image for easy use. However, if you choose to run it without the pre-built image, you will need to install the required packages and libraries. Please note that the tool has been primarily tested on Linux systems.
# Getting Started with an example
After successfully setting up DICE, you can test its basic functionality by running a simple example. We have provided a run script for the census dataset, where sex, race, and age are used as sensitive attributes.

To use the search algorithm on the Census dataset, run the following command:

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
