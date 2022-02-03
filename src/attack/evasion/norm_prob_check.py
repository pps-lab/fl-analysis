from . import NormBoundPGDEvasion
from .evasion_method import EvasionMethod
import logging
import numpy as np
import tensorflow as tf

class NormBoundProbabilisticCheckingEvasion(NormBoundPGDEvasion):
    """
    Adaptive attack for probabilistic checking
    """

    def __init__(self, old_weights, norm_type, scale_factor, keep_number_of_weights, clipping_bound=None, pgd_factor=None, benign_updates=None):
        self.keep_number_of_weights = keep_number_of_weights  # K, top-K
        super().__init__(old_weights, norm_type, scale_factor, clipping_bound, pgd_factor, benign_updates)

        assert self.clipping_bound is not None, "Must provide an explicit clipping bound to clip other values to!"
        assert self.norm_type == "linf", "Norm type must be linf in order to use probabilistic checking!"

    def update_after_training(self, model):
        # scale top-K!

        new_weights = model.get_weights()
        delta_weights = [(new_weights[i] - self.weights[i])
                         for i in range(len(self.weights))]

        delta_weights = self.flatten_update(delta_weights)

        # select top-K
        indices_to_scale = np.argpartition(np.abs(delta_weights), -self.keep_number_of_weights)[-self.keep_number_of_weights:]
        indices_to_not_scale = list(set(range(len(delta_weights))) - set(indices_to_scale.tolist()))

        for i in indices_to_scale:
            delta_weights[i] = delta_weights[i] * self.scale_factor

        for i in indices_to_not_scale:
            delta_weights[i] = np.clip(delta_weights[i], -self.clipping_bound, self.clipping_bound)

        delta_weights = self.unflatten(delta_weights, self.weights)

        update = [self.weights[i] + delta_weights[i] for i in range(len(self.weights))]
        model.set_weights(update)

    def unflatten(self, w, weights):
        sizes = [x.size for x in weights]
        split_idx = np.cumsum(sizes)
        update_ravelled = np.split(w, split_idx)[:-1]
        shapes = [x.shape for x in weights]
        update_list = [np.reshape(u, s) for s, u in zip(shapes, update_ravelled)]
        return update_list

    def flatten_update(self, update):
        return np.concatenate([x.ravel() for x in update])

    def apply_defense(self, old_weights, new_weights, clip=None, clip_l2=None):
        """
        Applies defenses based on configuration
        :param clip:
        :param old_weights:
        :param new_weights:
        :return: new weights
        """
        assert old_weights is not None, "Old weights can't be none"
        assert new_weights is not None, "New weights can't be none"
        delta_weights = [new_weights[i] - old_weights[i]
                         for i in range(len(old_weights))]
        # clip_layers = self.config['clip_layers'] if self.config['clip_layers'] != [] else range(len(old_weights))
        clip_layers = range(len(old_weights))

        if clip is not None:
            delta_weights = [np.clip(delta_weights[i], -clip, clip) if i in clip_layers else delta_weights[i]
                             for i in range(len(delta_weights))]

        if clip_l2 and clip_l2 > 0:
            delta_weights = self.clip_l2(delta_weights, clip_l2, clip_layers)

        new_weights = [old_weights[i] + delta_weights[i] for i in range(len(old_weights))]

        return new_weights

    def apply_pgd_weights(self, old_weights, new_weights):
        pgd = self.norm_type
        if self.pgd_factor is not None:

            pgd_constraint = self.pgd_factor

            self.debug(f"Applying constraint {pgd} with value {pgd_constraint}")

            if pgd == 'linf':
                new_weights = self.apply_defense(old_weights, new_weights, pgd_constraint, None)
            elif pgd == 'l2':
                new_weights = self.apply_defense(old_weights, new_weights, None, pgd_constraint)
            elif pgd == 'median_l2':
                assert self.benign_updates is not None, "We must have knowledge of the benign updates in order to" \
                                                        "compute the required bound (median)!"
                # compute median
                median_pgd = self.benign_median * self.pgd_factor / self.scale_factor
                new_weights = self.apply_defense(old_weights, new_weights, None, median_pgd)
            else:
                raise Exception('PGD type not supported')
        return new_weights

    def clip_l2(self, delta_weights, l2, clip_layers):
        """
        Calculates the norm per layer.
        :param delta_weights: current weight update
        :param l2: l2 bound
        :param clip_layers: what layers to apply clipping to
        :return:
        """

        l2_norm_tensor = tf.constant(l2, dtype=tf.float32)
        layers_to_clip = [tf.reshape(delta_weights[i], [-1]) for i in range(len(delta_weights)) if
                          i in clip_layers]  # for norm calculation
        norm = tf.norm(tf.concat(layers_to_clip, axis=0))

        multiply = min((l2_norm_tensor / norm).numpy(), 1.0)

        return [delta_weights[i] * multiply if i in clip_layers else delta_weights[i] for i in
                range(len(delta_weights))]

    def compute_median(self, client_updates):
        """
        Compute the client median values based on client updates.

        @type client_updates: list[list[np.array]] client layer weights
        """
        l2_norms_per_client = [tf.norm(tf.concat([tf.reshape(delta_weights[i], [-1]) \
                                                  for i in range(len(delta_weights))], axis=0)) \
                               for delta_weights in client_updates]  # for norm calculation
        median_real = np.median(l2_norms_per_client)

        # Given this median, we can be smart and calcluate if, including us we are even/uneven, and depending on
        # that, select the value 1 above the median (influenced by us being added to the list), or average between the two.
        # Assumes attacker will _always_ give an update that is larger.
        # TODO: This should also depend on the number of malicious attackers.
        l2_norms_per_client.append(np.max(l2_norms_per_client))  # Pretend to add self as max
        median = np.median(l2_norms_per_client)

        return median

    def debug(self, v):
        logging.debug(f"Norm bound stealth: {v}")