.. Adversarial FL documentation master file, created by
   sphinx-quickstart on Mon Jun 14 16:37:35 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Federated Learning with Adversaries
==========================================
This page is documentation for the `FL with Adversaries <https://github.com/pps-lab/fl-analysis>`_ analysis framework.
For now, it provides a reference documentation of all the configuration options that are available in the framework.

.. toctree::
   :maxdepth: 2

   config


Config
======
This is an example of a basic configuration for the framework.
This experiment consists of an FL setup with 3383 clients for the Federated-MNIST task.
The experiment runs for 80 rounds and in each round, 30 clients are selected
The clients train for two epochs and the updates are constrained to have an L2-norm bound of 10.

.. code-block:: yaml

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

Other configuration examples can be found in `./train_configs` in the project.
For all configuration options, see :doc:`config`




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`