.. Adversarial FL documentation configuration specification file.

.. _full_configuration:

Framework Configuration
==========================================

To use the framework, a config must be specified in YAML format.
A full specification of all the options is defined below.
The configuration file consists of several config `sections`, some of which must be specified.

.. autoclass:: src.config.definitions.Config
   :members:

.. autoclass:: src.config.definitions.ClientConfig
   :members:

.. autoclass:: src.config.definitions.TrainingConfig
   :members:
.. autoclass:: src.config.definitions.LearningDecay
   :members:
.. autoclass:: src.config.definitions.MaliciousConfig
   :members:
.. autoclass:: src.config.definitions.NormBound
   :members:
.. autoclass:: src.config.definitions.Quantization
   :members:
.. autoclass:: src.config.definitions.Server
   :members:
.. autoclass:: src.config.definitions.FederatedDropout
   :members:
.. autoclass:: src.config.definitions.Dataset
   :members:
.. autoclass:: src.config.definitions.Environment
   :members:
.. autoclass:: src.config.definitions.HyperparameterConfig
   :members:
