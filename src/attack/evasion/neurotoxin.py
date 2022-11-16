from . import NormBoundPGDEvasion
from .evasion_method import EvasionMethod
import logging
import numpy as np
import tensorflow as tf

class NeurotoxinEvasion(NormBoundPGDEvasion):
    """
    Adaptive attack for probabilistic checking
    """

    def __init__(self, old_weights, norm_type, scale_factor, topk, last_round_weights, clipping_bound=None, benign_updates=None):
        self.topk = topk
        self.last_round_weights = last_round_weights  # K, top-K
        self.indices_to_reset = None
        self.pgd_factor = np.nan # so apply_pgd_weights gets called
        super().__init__(old_weights, norm_type, scale_factor, clipping_bound, np.nan, benign_updates)
        assert self.last_round_weights is not None, "Last round's weights cannot be None, we need them to estimate S!"

        self.indices_to_reset = self.compute_smallest_set_indices(old_weights, last_round_weights, topk)

    def compute_smallest_set_indices(self, old_weights, last_round_weights, topk):
        """
        Set self.indices_to_reset
        @return:
        """
        delta_weights = [old_weights[i] - last_round_weights[i]
                         for i in range(len(old_weights))]
        delta_weights = self.flatten_update(delta_weights)

        # find topk
        keep_number_of_weights = int(len(delta_weights) * (topk / 100.0))
        indices_to_reset = np.argpartition(np.abs(delta_weights), -keep_number_of_weights)[-keep_number_of_weights:]
        return indices_to_reset

    def apply_pgd_weights(self, old_weights, new_weights):
        """
        Does not apply PGD but projects onto S
        @param old_weights:
        @param new_weights:
        @return:
        """
        delta_weights = [new_weights[i] - old_weights[i]
                         for i in range(len(old_weights))]
        delta_weights = self.flatten_update(delta_weights)

        delta_weights[self.indices_to_reset] = 0.0

        delta_weights = self.unflatten(delta_weights, self.weights)
        new_weights = [old_weights[i] + delta_weights[i] for i in range(len(old_weights))]
        return new_weights

    def unflatten(self, w, weights):
        sizes = [x.size for x in weights]
        split_idx = np.cumsum(sizes)
        update_ravelled = np.split(w, split_idx)[:-1]
        shapes = [x.shape for x in weights]
        update_list = [np.reshape(u, s) for s, u in zip(shapes, update_ravelled)]
        return update_list

    def flatten_update(self, update):
        return np.concatenate([x.ravel() for x in update])
