# The Implementation of DICE
### An information-theoritic tool to testing and debugging fairness defects in deep Neural Network

Our tool consists of two steps: 1) the search phase uses clustering and gradients to maximize the
amounts of discrimination and generate as many discrimination instance as possible 2) the debugging
phase that uses a layer localizer and causal analysis to pinpoint the root cause of discrimination
in the internal of deep neural network.

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
