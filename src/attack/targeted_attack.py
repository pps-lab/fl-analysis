from src.attack.attack import LossBasedAttack

import tensorflow as tf
import logging
import numpy as np

from src.data import image_augmentation

logger = logging.getLogger(__name__)


class TargetedAttack(LossBasedAttack):

    def generate(self, dataset, model, **kwargs):

        self.parse_params(**kwargs)

        self.weights = model.get_weights()

        loss_object_with_reg = self._combine_losses(
            self.stealth_method.loss_term(model) if self.stealth_method is not None else None,
            self.stealth_method.alpha if self.stealth_method is not None else None)

        for epoch in range(self.num_epochs):
            batch_counter = 0
            for batch_x, batch_y in dataset.get_data_with_aux(self.poison_samples, self.num_batch, self.noise_level):  # 10 is ICML
                # print(f"LR: {mal_optimizer._decayed_lr(var_dtype=tf.float32)}")

                with tf.GradientTape() as tape:
                    loss_value = loss_object_with_reg(y_true=batch_y, y_pred=model(batch_x, training=True))


                    # print(loss_value)
                    # print(batch_y)
                    # image_augmentation.debug(batch_x[0:1], batch_y[0:1])
                    grads = self._compute_gradients(tape, loss_value, model)
                    self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

                if self.step_decay is not None:
                    self.step_decay.apply_step()

                if self.stealth_method is not None:
                    self.stealth_method.update_after_batch(model)

                batch_counter += 1

            # test_success, adv_success = self.eval_aux_test(dataset, model, self.loss_object)
            # print(test_success, adv_success)

            logger.info(f"Epoch {epoch}: {batch_counter}")

        if self.stealth_method is not None:
            self.stealth_method.update_after_training(model)

        return model.get_weights()

    def parse_params(self, num_epochs, num_batch, poison_samples, optimizer, loss_object, step_decay=None,
                     noise_level=None):
        self.num_epochs = num_epochs
        self.num_batch = num_batch
        self.poison_samples = poison_samples
        self.optimizer = optimizer
        self.loss_object = loss_object
        self.step_decay = step_decay
        self.noise_level = noise_level

    def eval_aux_test(self, dataset, model, loss_object):
        def calc_acc(ds):
            counter = 10
            adv_ss = []
            for batch_x, batch_y in ds: # aux samples
                preds = model(batch_x, training=False)
                loss_value = loss_object(y_true=batch_y, y_pred=preds)

                pred_inds = preds.numpy().argmax(axis=1) == batch_y
                # print(pred_inds, batch_y)
                adv_success = np.mean(pred_inds)
                adv_ss.append(adv_success)

                counter -= 1
                if counter == 0:
                    break


            return np.mean(adv_ss)

        return calc_acc(dataset.get_data()), calc_acc(dataset.get_data_with_aux(self.poison_samples, self.num_batch, self.noise_level))