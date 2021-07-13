<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://pps-lab.com/research/ml-sec/">
    <img src="https://github.com/pps-lab/fl-analysis/blob/master/documentation/ml-sec-square.png?raw=true" alt="Logo" width="80" height="80">  
  </a>

  <h2 align="center">Federated Learning with Adversaries</h2>

[comment]: <> (  <h3 align="center">Framework for to analyse FL with ad</h3> )
</p>

<!-- TABLE OF CONTENTS -->
<details open="open"> 
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#backdoor-attacks">Backdoor attacks</a></li>
        <li><a href="#robustness">Robustness</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Requirements</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## About the Project
This framework can be used to simulate and analyse a federated learning setting in which some clients are compromised by an adversary.
The adversary attempts to compromise the integrity of the shared global model by sending malicious updates to the server.

The framework was used to perform the analysis of federated learning robustness under a norm bound defense as part
of [RoFL: Attestable Robustness for Secure Federated Learning](https://arxiv.org/abs/2107.03311).
A research implementation of the secure federated learning with constraints framework can be found [here](https://github.com/pps-lab/rofl-project-code).

### Backdoor attacks
In federated learning, adversaries can perform backdoor attacks to poison the global model.
This framework implements existing attack strategies such as [model replacement](https://arxiv.org/abs/1807.00459), 
on a wide variety of tasks and backdoor attack targets proposed in previous work, such as
attacks on [prototypical targets](https://research.google/pubs/pub48698/) or [edge cases](https://arxiv.org/abs/2007.05084).

### Robustness
The framework provides several tools to analyse client updates, measure backdoor performance and deploy defenses to
gain insight on model robustness in federated learning. 

<!-- GETTING STARTED -->
## Getting Started

We now describe how to set up this framework.

### Requirements
The dependencies can be automatically installed through `pipenv`.
The high-level requirements are as follows.
- Python 3 (tested on version 3.7)
- [TensorFlow](https://www.tensorflow.org/) (version 2.0)

Before starting, ensure that you have `pipenv` installed:

```sh
pip install pipenv
```

### Installation

1. Clone the repo
```sh
git clone https://github.com/pps-lab/fl-analysis.git
```

2. Install the Python packages
```sh
pipenv install
```

## Usage
The configuration of the framework is specified in a config file in YAML format.
A minimal example of a config is shown below.
```yaml
environment:
  num_clients: 3383
  num_selected_clients: 30
  num_malicious_clients: 0
  experiment_name: "Sample run without attackers"

server:
  num_rounds: 80
  num_test_batches: 5
  aggregator:
    name: FedAvg
  global_learning_rate: -1

client:
  clip:
    type: l2
    value: 10
  model_name: resnet18
  benign_training:
    num_epochs: 2
    batch_size: 24
    optimizer: Adam
    learning_rate: 0.001

dataset:
  dataset: femnist
  data_distribution: nonIID
```
The full specification of the supported config options can be found [here](https://pps-lab.com/fl-analysis/)
Some example config files can be find in `train_configs`.

## Sample usage:
With a config file `config.yml` ready, the framework can be started by invoking:
```commandline
python -m src.main -c config.yml
```

## Available models
Some pre-trained models are available in the `models` for experiments and can be included in training using the `environment.load_model` config key.
- `lenet5_emnist_088.h5` LeNet5 for federated-MNIST at 0.88 accuracy.
- `lenet5_emnist_097.h5` LeNet5 for federated-MNIST at 0.97 accuracy.
- `lenet5_emnist_098.h5` LeNet5 for federated-MNIST at 0.98 accuracy.
- `resnet18.h5` ResNet18 for CIFAR-10 at 0.88 accuracy.
- `resnet18_080.h5` ResNet18 for CIFAR-10 at 0.80 accuracy.
- `resnet18_082.h5` ResNet18 for CIFAR-10 at 0.82 accuracy.
- `resnet156_082.h5` ResNet56 for CIFAR-10 at 0.86 accuracy.


## Output 
Basic training progress is sent to standard output.
More elaborate information is stored in an output folder.
The directory location can be specified through the `XXX` option.
By default, its ... .
The framework stores progress in tfevents, which can be viewed using Tensorboard, e.g.,
```bash
tensorboard --logdir ./experiments/{experiment_name}
```

<!-- LICENSE -->
## License

This project's code is distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

* Hidde Lycklama - [hiddely](https://github.com/hiddely)
* Lukas Burkhalter - [lubux](https://github.com/lubux)

## Project Links: 
* [https://github.com/pps-lab/fl-analysis](https://github.com/pps-lab/fl-analysis)
* [https://pps-lab.com/research/ml-sec/](https://pps-lab.com/research/ml-sec/)
