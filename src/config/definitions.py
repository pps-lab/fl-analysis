
from dataclasses import dataclass, MISSING, field
from typing import Optional, Dict, Any, List

from mashumaro.mixins.yaml import DataClassYAMLMixin

"""
This class defines the configuration schema of the framework.
"""

@dataclass
class Quantization(DataClassYAMLMixin):
    """
    Apply quantization to the client updates, to simulate conversion to fixed-point representation in the crypto domain
     (secure aggregation).

    """
    type: str = MISSING # "probabilistic" or "deterministic"
    """Type of quantization to be used. Either `probabilistic` or `deterministic`."""
    bits: int = MISSING
    """Number of total bits per parameter to quantize to."""
    frac: int = MISSING
    """Number of bits to use for the fractional part of the number."""


@dataclass
class HyperparameterConfig(DataClassYAMLMixin):
    """
    Config for hyperparameters being tuned in this run.
    """
    args: Dict[str, Any]
    """Free-format dictionary of key-value pairs, where `key` must be string and `value` can be any type that tfevents
    can handle."""

    # in the future.. config of values to log and when.


@dataclass
class Environment(DataClassYAMLMixin):
    """Defines the environment of the experiment.
    In addition, this config section has additional settings for logging different statistics by the framework."""
    num_clients: int = MISSING
    """Number of clients"""
    num_selected_clients: int = MISSING
    """Number of clients selected in each round"""

    num_malicious_clients: int = MISSING
    """Number of malicious clients in total"""
    experiment_name: str = MISSING
    """Name of the experiment (used to create the logging directory)"""

    num_selected_malicious_clients: int = None
    """Number of malicious clients selected in each round"""

    malicious_client_indices: Optional[List[int]] = None
    """Select a specific set of clients to be malicious, based on client index"""

    attack_frequency: Optional[float] = None #
    """Frequency of malicious parties being selected. Default is None, for random selection"""

    # these should be removed in the future
    attacker_full_dataset: bool = False #
    """ Whether the attacker has the full dataset."""
    attacker_full_knowledge: bool = False
    """Whether the attacker has full knowledge of the other client updates in a given round."""

    load_model: Optional[str] = None
    """Load a model to be used in training instead of a randomly initialized model.
    Can be used to load pre-trained models."""
    ignore_malicious_update: bool = False
    """Compute a malicious update, but ignore the update in aggregation.
    Can be used to compute malicious client update statistics over time without compromising the model."""

    print_every: int = 1
    """Defines after how many rounds should the framework evaluate the accuracy of the model."""

    print_batch_text: bool = False
    """For character-prediction tasks, print batch of test set including prediction"""

    save_updates: bool = False
    """Save all client updates in each round (Note: Quickly takes up a significant amount of storage space)."""
    save_norms: bool = False
    """Log the norms of the client updates to a file."""
    save_weight_distributions: bool = False
    """Save the weight update distributions to tfevents files."""
    save_history: bool = False
    """Save the history of the global models on the server."""
    save_model_at: List[int] = field(default_factory=lambda: [])
    """Framework saves the global model to a file at these rounds."""
    save_weight_outside_bound: Optional[float] = None  # to keep track of weights outside the l_inf bound
    """Keep track of how many weights are outside the given l_inf bound."""

    print_backdoor_eval: bool = False
    """Whether to log the raw backdoor predictions of the model (at evaluation time)"""

    seed: int = 0 # randomness seed
    """Randomness seed"""

    use_config_dir: bool = False # whether to use the config parent dir as the target experiment dir
    """Whether to create a separate experiments output directory. 
    If `False` (default), the directory of the config YAML file is used as output directory."""

    limit_tf_gpu_mem_mb: Optional[int] = None
    """Provide memory limit (MB) for tensorflow to allocate on the GPU, to leave space for other operations."""

@dataclass
class Dataset(DataClassYAMLMixin):
    """Defines the dataset to use."""
    dataset: str = MISSING
    """Dataset type. Supported: `mnist`, `femnist`, `cifar10`"""
    data_distribution: str = MISSING
    """Data distribution over the clients. specify `IID` for I.I.D. distribution, `NONIID` otherwise."""
    number_of_samples: int = -1
    """Number of samples to use in the dataset. -1 means all samples"""
    augment_data: bool = False
    """Whether to augment the data with random horizontal flips and horizontal/vertical shifts. Used for CIFAR-10."""
    normalize_mnist_data: bool = False  # Legacy flag, CIFAR is normalized
    """Legacy flag, whether to normalize the MNIST dataset"""


@dataclass
class FederatedDropout(DataClassYAMLMixin):
    """Defines federated dropout behavior."""
    rate: float = 1.0
    """Dropout rate. Keep `rate` percentage of parameters"""
    all_parameters: bool = True
    """If set to True, applies dropout on all parameters randomly according to the dropout rate."""
    nonoverlap: bool = False
    """Each client receives a unique mask that is not overlapped with other clients' masks."""
    randommask: bool = False
    """Enable low rank mode instead of federated dropout, i.e. only mask the uplink."""


@dataclass
class Server(DataClassYAMLMixin):
    """Defines server behavior."""
    num_rounds: int = MISSING # Number of training rounds.
    """Number of training rounds"""
    num_test_batches: int = MISSING # Number of client epochs.
    """Number of batches to evaluate on in each evaluation"""

    federated_dropout: Optional[FederatedDropout] = None
    """Federated dropout configuration"""
    aggregator: Optional[Dict[str, Any]] = None
    """Aggregator to use. Default is FedAvg."""
    global_learning_rate: Optional[float] = None
    """Global learning rate"""

    intrinsic_dimension: int = 1000
    """For subspace learning, the size of the intrinsic dimension."""
    gaussian_noise: float = 0.0
    """Amount (sigma) of centered around 0 Gaussian noise to add to the model update each round."""


@dataclass
class LearningDecay(DataClassYAMLMixin):
    """Defines a learning rate decay schedule"""
    type: str = MISSING # exponential or boundaries
    """Type of learning rate decay. Supported: `exponential` for exponential decay and `boundaries` for 
    decay at arbitrary step boundaries."""

    # exponential
    decay_steps: Optional[int] = None
    """The number of steps after which the learning rate decays each time (requires `exponential` type)."""
    decay_rate: Optional[float] = None
    """The rate at which the learning rate decays (requires `exponential` type)."""

    # boundaries
    decay_boundaries: Optional[List[int]] = None
    """The list of decay boundaries (requires `boundaries` type)."""
    decay_values: Optional[List[float]] = None
    """The list of decay multiples corresponding to the areas defined by the boundaries (requires `boundaries` type)."""

    # epochs
    step_epochs: bool = False
    """Boundaries and steps are expressed as epoch"""

@dataclass
class TrainingConfig(DataClassYAMLMixin):
    """Client training configuration"""
    num_epochs: int = MISSING
    """Number of training epochs"""
    batch_size: int = MISSING # Client batch size
    """ """
    learning_rate: float = MISSING
    """ """
    decay: Optional[LearningDecay] = None
    """Optional learning rate decay schedule"""
    optimizer: str = "Adam" # Optimizer
    """Optimizer to use"""
    regularization_rate: Optional[float] = None
    """Use L2 regularization to limit the size of the model `update` (not the model itself)."""


@dataclass
class NormBound(DataClassYAMLMixin):
    """Enforce a norm bound"""
    type: str = MISSING # l2, linf, median_l2
    """Type of norm bound. Supported: `l2`, `linf`, `median_l2`"""
    value: float = MISSING
    """Norm bound value"""
    probability: Optional[float] = None # in case of linf, random clip
    """(`linf`) Legacy option to support clipping of randomly selected parameters"""

@dataclass
class MaliciousConfig(DataClassYAMLMixin):
    """Malicious training configuration. A malicious training configuration is largely defined by an attack objective
    (targeted or untargeted), an evasion method (to evade a potential defense such as a norm bound) and a backdoor
    that defines the specific set of samples to poison."""
    objective: Dict[str, Any] = MISSING
    """Attack objective. Corresponds with attack classes. Supported: `TargetedAttack`, `UntargetedAttack`."""
    evasion: Optional[Dict[str, Any]] = None
    """Evasion method. Supported: `NormBoundPGDEvasion` or `TrimmedMeanEvasion`"""
    backdoor: Optional[Dict[str, Any]] = None
    """Backdoor type. Supports several kinds of backdoors, see the examples for more details. Support types: `tasks`
    (select a set handwriters to poison), `edge_case` (select a set of external, edge-case samples), 
    `semantic` (define specific samples by id to poison)
    """
    attack_type: Optional[str] = None
    """Legacy option, `targeted` or `untargeted`"""
    estimate_other_updates: bool = False
    """Estimate the update that the clients will send based on the difference of the global model with 
    the previous round, based on an idea proposed by Bhagoji et al."""

    attack_start: Optional[int] = 0
    """Round after which to start attacking"""
    attack_stop: Optional[int] = 10000000
    """Round after which to stop attacking"""

    # In a scaling attack with multiple attackers, whether attackers should divide
    # a single malicious update amongst themselves.
    multi_attacker_scale_divide: Optional[bool] = None
    """When multiple attackers are selected in a single round, one attacker computes an update that all 
    selected attackers then submit to the server. This can be useful to improve norm bound evasion.
    The default behavior (when this flag is `false` or unspecified) is that all
    attackers compute an update independently."""


@dataclass
class ClientConfig(DataClassYAMLMixin):
    """Defines the behavior of the clients."""
    model_name: str = MISSING
    """What model to use. Note: Not all model / dataset combinations are supported. Supported models:
    `dev`, `mnist_cnn`, `bhagoji`, `resnet18`, `resnet32`, `resnet44`, `resnet56`, `resnet110`, `resnet18_v2`,
    `resnet56_v2`, `dev_intrinsic`, `dev_fc_intrinsic`, `bhagoji_intrinsic`, `mnistcnn_intrinsic`, `lenet5_mnist`,
    `lenet5_cifar`, `lenet5_intrinsic`, `allcnn`, `allcnn_intrinsic`"""
    benign_training: TrainingConfig = MISSING
    """Training config of the clients"""
    quantization: Optional[Quantization] = None
    """Optional probabilistic quantization configuration"""
    malicious: Optional[MaliciousConfig] = None
    """Optional configuration for the malicious clients"""
    optimized_training: bool = True # whether to create the TF training loop
    """Whether to use optimized Tensorflow training. This should work by default, but when a large amount of clients
    (> 5000) is used with limited GPU memory, the training process may run out of memory after training for some time."""
    gaussian_noise: Optional[float] = None
    """Sigma of 0-centered gaussian noise to apply to the training samples"""
    clip: Optional[NormBound] = None
    """Apply a norm bound to the client updates"""
    model_weight_regularization: Optional[float] = None # weight reg for model (l2)
    """Weight regularization of the model (l2)"""
    debug_client_training: bool = False
    """Debug client training process. Logs per-epoch training loss."""
    disable_bn: bool = False
    """Disable batchnorm in resnet models"""

@dataclass
class Config(DataClassYAMLMixin):
    """Config root that defines the sections that can/must be specified."""
    environment: Environment = MISSING
    """Details about the FL environment of the experiment"""
    server: Server = MISSING
    """Server configuration."""
    client: ClientConfig = MISSING
    """Client configuration."""
    dataset: Dataset = MISSING
    """Dataset configuration."""
    hyperparameters: Optional[HyperparameterConfig] = None
    """Hyperparameters that are used this run. This is a free-format section and can be used to log hyperparameters 
    in the tfevents format to later analyze using Tensorboard."""