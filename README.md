# Towards Autonomous Operation of Two-Stage Reverse Osmosis Water Treatment System with Multi-Agent Reinforcement Learning

## Overview

This repository contains the code for implementing and evaluating multi-agent reinforcement learning (MARL) algorithms for optimizing the operation of a two-stage reverse osmosis (RO) process. The goal is to develop control policies that can effectively manage the RO process under various operational conditions and disturbances, ensuring stable and efficient performance. Under review for submission to Desalination journal (https://www.sciencedirect.com/journal/desalination).

## Graphical Abstract
![GraphicalAbstract](https://github.com/user-attachments/assets/608ddaf7-ff39-4e18-aa3f-c72878d24ddf)



## Repository Structure

The repository is organized as follows:

-   `evaluate_intended_failure.py`: Script for evaluating the performance of trained RL agents under specific failure scenarios.
-   `optimize_pressure_RO_centralized.py`: Script for training RL agents with a centralized control architecture for RO process optimization.
-   `optimize_pressure_RO.py`: Script for training RL agents with a decentralized control architecture for RO process optimization.
-   `requirements.txt`: List of Python dependencies required to run the code.
-   `algorithms/`: Contains implementations of different MARL algorithms.
    -   `mixer/`: Implementation of mixing networks for value function decomposition.
-   `config/`: Configuration files for running experiments.
    -   `run_exp_episodic_centralized.sh`: Shell script for running experiments with a centralized control architecture.
    -   `run_exp_episodic.sh`: Shell script for running experiments with a decentralized control architecture.
    -   `episodic_conf/`: Configuration files for training.
-   `evaluation/`: Scripts and notebooks for evaluating trained RL agents.
    -   `hinderonegivenperiod.sh`: Shell script for evaluating agent performance when hindering one agent for a given period.
    -   `inspect_evaluation.ipynb`: Jupyter notebook for inspecting and analyzing evaluation results.
-   `interpretability/`: Notebooks and scripts for interpreting the learned policies.
    -   `interpret_evaluation_result.ipynb`: Jupyter notebook for interpreting evaluation results and visualizing agent behavior.
    -   `visualize_shap.ipynb`: Jupyter notebook for visualizing SHAP values to understand feature importance.
-   `parameters/`: Directory to contain trained model parameters.
    -   `VDN/`: Directory storing trained parameters for the VDN algorithm.
-   `TwoStageROProcessEnvironment/`: Contains the implementation of the two-stage RO process environment.
-   `utils/`: Utility functions and classes.
-   `visualization/`: Scripts for visualizing the RO process and agent behavior.

## Important Considerations

### About the codes:
-   There exists a number of **hard-coded paths**, **algorithm selection**, **parameter selection**, _etc_.
-   Please customize the code to your need.
-   Although there is only QMIX.py in algorithms/mixer directory, it contains more functionality than mere 'QMIX' implementation.
    -   ReplayBuffer, PrioritizedReplayBuffer are implemented. It may seem odd, but PrioritizedReplayBuffer was used as main replay buffer, only in UNIFORM mode (same function as vanilla ReplayBuffer). In addition, methods such as loss calculation are implemented in the class.

## Details about implementation

## Algorithms Used for MARL
-   Value Decomposition Networks (VDN)
    -   https://arxiv.org/abs/1706.05296
-   QMIX
    -   https://arxiv.org/abs/1803.11485
-   Deep Recurrent Q-Network (DRQN)
    -   https://arxiv.org/abs/1507.06527

## Libraries Used for MARL
-   PyTorch
    -   https://pytorch.org/
-   PettingZoo
    -   https://pettingzoo.farama.org/index.html
    -   https://arxiv.org/abs/2009.14471


## Algorithm Used for Two-Stage RO Simulation
-   The development of membrane fouling in full-scale RO processes (Chen et al., 2004)
    -   https://doi.org/10.1016/j.memsci.2003.11.028
    -   Implemented with Julia by the authors.

-   Modeling and evaluating performance of full-scale reverse osmosis system in industrial water treatment plant (Jeong et al., 2021)
    -   https://doi.org/10.1016/j.desal.2021.115289
    -   Implemented with Julia, without foulant and cake layer formation modeling. 

## Libraries and Language(s) Used for Two-Stage RO Simulation
-   Julia
    -   https://julialang.org/
    -   Adopted for its high-speed and implementational advantage. Since tremendous evaluation of the RO model is necessary for reinforcement learning, a little speed-up of the RO model was a big advantage for us.
-   JuliaCall
    -   https://juliapy.github.io/PythonCall.jl/stable/juliacall/




To install the necessary dependencies, run:

```bash
conda env create -f ROMARL.yml
