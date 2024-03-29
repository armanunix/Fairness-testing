# The Implementation of DICE
### An information-theoritic tool for testing and debugging fairness defects in deep Neural Network

This repository provides the tool and the evaluation Experiments for the paper ["Information-Theoretic Testing and Debugging of
Fairness Defects in Deep Neural Networks"](https://www.computer.org/csdl/proceedings-article/icse/2023/570100b571/1OM4zHEXjfa), ICSE'2023.

# The structure of the repository 

The repository includes:

- [Dockerfile](https://github.com/armanunix/Fairness-testing/blob/main/DICE/Dockerfile) to build the Docker script,
- [DICE_baseline](https://github.com/armanunix/Fairness-testing/tree/main/DICE/DICE_baseline) includes the implementation of our baselines,    
- [DICE_data](https://github.com/armanunix/Fairness-testing/tree/main/DICE/DICE_data) includes functions to load datasets,     
- [DICE_model](https://github.com/armanunix/Fairness-testing/tree/main/DICE/DICE_model) includes the implementation of DNN model,    
- [DICE_tutorial](https://github.com/armanunix/Fairness-testing/tree/main/DICE/DICE_tutorial) includes the source code of DICE, 
- [DICE_utils](https://github.com/armanunix/Fairness-testing/tree/main/DICE/DICE_utils) includes utility functions of DICE,    
- [Raw_data](https://github.com/armanunix/Fairness-testing/tree/main/DICE/Raw_data/raw) includes raw dataset files(meps15/16),    
- [Clusters](https://github.com/armanunix/Fairness-testing/tree/main/DICE/clusters) includes pickles from KMeans clustering,   
- [Datasets](https://github.com/armanunix/Fairness-testing/tree/main/DICE/datasets) includes pre-processed datasets,   
- [Models](https://github.com/armanunix/Fairness-testing/tree/main/DICE/models) includes trained models checkpoints,  
- [Requirements](https://github.com/armanunix/Fairness-testing/blob/main/DICE/Requirements.txt) set of required libraries for running the tool on local machine, 
- [Results](https://minersutep-my.sharepoint.com/:f:/g/personal/vmonjezi_miners_utep_edu/EqN3oXLgnppGuxsgdMqBH54BuDSfFgUUX0xS5E5O-aMBQw?e=pMY2Eg) includes the results of all experimrnts,  
- [Script](https://github.com/armanunix/Fairness-testing/blob/main/DICE/DICE_tutorial/run_script_final.sh) file to rerun all experiments,  
- [License](https://github.com/armanunix/Fairness-testing/blob/main/LICENSE) file.  

# Docker repository
```
docker pull armanunix/dice:1.0.0
docker run --rm -it armanunix/dice:1.0.0
```
We recommend to use Docker's [volume](https://docs.docker.com/engine/reference/run/#volume-shared-filesystems) feature to connect the docker container to the own file system so that DICE's results can be easily accessed. Below commands create a docker volume and Run a container using the DICE:1.0.0 image and mount the volume to it.
```
sudo docker volume create DICE_VOLUME
sudo docker run -it --name=DICE_CONT --mount source=DICE_VOL,destination=/root/results armanunix/dice:1.0.0
```
Now, any results generated in the container will be stored in the host directory "/var/lib/docker/volumes/DICE_VOLUME/_data"

# Tool
Our tool consists of two steps: 1) the search phase uses clustering and gradients to maximize the
amounts of discrimination and generate as many discrimination instance as possible 2) the debugging
phase that uses a layer localizer and causal analysis to pinpoint the root cause of discrimination
in the internal of deep neural network.

# Requirements
Python 3.8  
numpy==1.22.0  
pandas==1.5.1  
tensorflow==2.7.0  
scipy==1.4.1  
argparse==1.1  
protobuf==3.9.2  
scikit-learn==0.22.2.post1   
aif360==0.4.0  
IPython==7.13.0  
regex==2022.10.31
# How to setup DICE
If you use the pre-built Docker image, the tool is already built to use in Ubuntu base. Otherwise, the installation of required packlages and libraries should be sufficient to run DICE. Note: the tool is substantially tested in Linux system.
# Getting Started with an example
After succesfully setup DICE, you can try a simple example to check the basic functionality of our tool. We prepared a simple run script for census dataset with sex, race, and age as sensitive attributes
DICE first generates indicvidual discriminatory instances using a two-phase gradient-guided search. DICE is able to analyze the input dataset using a set of its protected attributes. Then, DICE debugging algorithm uses generated discriminatory instances of DICE search algorithm to localize the biased layer and neurons through causal analysis. Throughout the paper for the RQ1 table, we run DICE search algorithm 10 times each 1 hour. Here you can try a simple example to check the basic functionality of DICE searching/debugging in 10 minutes.


To run the search algorithm over Census dataset considering Sex, race, and age as the protected attributes - sensitive_index=9,8,1 for 10 minutes. Note that for RQ2 results, the argument -RQ must be set to -RQ=2. 
```
python3 DICE_Search.py -dataset=census -sensitive_index=9,8,1 -timeout=600 -RQ=1
```
The result of the search will be saved to /results/census/DICE/RQ1/981_10runs/
To run the debugging algorithm on the 200 instances of generated discriminatory instances from the above command:
```
python3 DICE_Debugging.py -dataset=census -sensitive_index=9,8,1 -num_samples=200
```
The results of the above code will be stored in /results/census/DICE/RQ3

The other models for a different dataset can be run in the same fashion. For example, to run the search and debugging on german credit dataset with sex and age as protected attributes:
```
python3 DICE_Search.py -dataset=credit -sensitive_index=13,9 -timeout=600 -RQ=1
python3 DICE_Debugging.py -dataset=credit -sensitive_index=13,9 -num_samples=200
```
Here are the sensitive indices for 10 datasets we used to evaluate our tool:   
census 9,8,1   
compas 3,2,1   
bank 1   
credit 13,9   
default 5,2   
diabetes 8  
heart 2,1   
meps15 and meps16 10,2,1   
student 3,2



To run the RQ2 experiments, sensitive_index argument should be one index per each run. DICE is able to handle more than one protected attribute, but to be consistent with our baselines we set one sensitive feature per each experiment. You can try a simple example on census dataset and sex feature - sensitive_index=9. Note that for RQ2 results, the argument -RQ must be set to -RQ=2.
```
python3 DICE_Search.py -dataset=census -sensitive_index=9 -timeout=300 -RQ=2
```
The results of the above command will be saved in /results/census/DICE/RQ2/. Results for other datasets and other sensitive features can be done in the same fashion.

For RQ1 we run DICE 10 times each 1 hour for all datasets, For RQ2 we run 10 times each 15 minutes, and for RQ3 we set the number of samples(-num_samples) to 1000. The corresponding results can be found in [Results](https://minersutep-my.sharepoint.com/:f:/g/personal/vmonjezi_miners_utep_edu/EqN3oXLgnppGuxsgdMqBH54BuDSfFgUUX0xS5E5O-aMBQw?e=ZAWhbJ).

# Complete Evaluation Reproduction
We include the script to run the search and debugging algorithms for the entire dataset (warning: it will run all experiments):
```
sh run_script_final.sh
```
we include this command to reproduce the RQ1 experiment on census dataset and sensitive_index=9,8,1. Note that this command uses the discriminatory instances generated in our RQ1 experiment and can be found in [RQ1](https://minersutep-my.sharepoint.com/:f:/g/personal/vmonjezi_miners_utep_edu/EmUeDc0IaFxCpFflp0C-8AMBN_vmV2guny4JMZhBtAYOXQ?e=rMz2Mm)
This folder should be copied in /results/census/DICE/ to be used by the below command:
```
python3 DICE_RQ1.py -dataset=census -sensitive_index=9,8,1
```

For reproduction of RQ3 results on census dataset with our experiment results in [RQ1](https://minersutep-my.sharepoint.com/:f:/g/personal/vmonjezi_miners_utep_edu/EmUeDc0IaFxCpFflp0C-8AMBN_vmV2guny4JMZhBtAYOXQ?e=rMz2Mm):
```
python3 DICE_RQ3.py -dataset=census -sensitive_index=9,8,1
```
There is one folder for each experiment that includes DICE and the state-of-the-art outcomes.
