from src.attack.attack import LossBasedAttack

import logging
import numpy as np

import tensorflow as tf
from copy import copy
logger = logging.getLogger(__name__)

# Move this into generate later
# from src.torch_compat.anticipate import train_anticipate

class AnticipateTfAttack(LossBasedAttack):


    def generate(self, dataset, model, **kwargs):

        self.parse_params(**kwargs)

        self.weights = model.get_weights()

        loss_object_with_reg = self._combine_losses(
            self.stealth_method.loss_term(model) if self.stealth_method is not None else None,
            self.stealth_method.alpha if self.stealth_method is not None else None)

        attack_model = model
        current_model = copy(model)
        current_model.set_weights(attack_model.get_weights())

        fl_no_models = 10

        # from datetime import datetime
        # stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # logdir = 'logs/func/%s' % stamp  # <- Name of this `run`
        # writer = tf.summary.create_file_writer(logdir)
        # tf.summary.trace_on(graph=True, profiler=False)

        for epoch in range(self.num_epochs):
            batch_counter = 0

            for batch_x, batch_y in dataset.get_data_with_aux(self.poison_samples, self.num_batch, self.noise_level):
                # print(f"LR: {mal_optimizer._decayed_lr(var_dtype=tf.float32)}")
                loss = None

                with tf.GradientTape(persistent=True) as tape:

                    for anticipate_i in range(self.anticipate_steps):
                        if anticipate_i == 0:
                            current_model = self.honest_training(tape, dataset, current_model)

                            for att_weight, cur_weight in zip(attack_model.trainable_variables, current_model.trainable_variables):
                                after_avg = (att_weight + (cur_weight * (fl_no_models - 1))) / fl_no_models
                                cur_weight.assign(after_avg)
                        else:
                            current_model = self.honest_training(tape, dataset, current_model)

                        if self.optimization_method == 'A':
                            if anticipate_i == self.anticipate_steps - 1:
                                loss_value = loss_object_with_reg(y_true=batch_y,
                                                                  y_pred=current_model(batch_x, training=True))
                                loss = loss_value
                        else:
                            loss_value = loss_object_with_reg(y_true=batch_y,
                                                              y_pred=current_model(batch_x, training=True))
                            if loss is None:
                                loss = loss_value
                            else:
                                loss = loss + loss_value


                    # print(loss_value)
                    # print(batch_y)
                    # image_augmentation.debug(batch_x[0:1], batch_y[0:1])
                grads = self._compute_gradients(tape, loss_value, model)
                self.optimizer.apply_gradients(zip(grads, attack_model.trainable_variables))

                # if self.step_decay is not None:
                #     self.step_decay.apply_step()
                #
                # if self.stealth_method is not None:
                #     self.stealth_method.update_after_batch(model)

                batch_counter += 1

            # test_success, adv_success = self.eval_aux_test(dataset, model, self.loss_object)
            # print(test_success, adv_success)

            logger.info(f"Epoch {epoch}: {batch_counter}")

        if self.stealth_method is not None:
            self.stealth_method.update_after_training(attack_model)

        # with writer.as_default():
        #     tf.summary.trace_export("attack_graph", step=1)

        return attack_model.get_weights()

    def honest_training(self, tape, dataset, model):

        honest_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

        loss_object_with_reg = self._combine_losses(
            self.stealth_method.loss_term(model) if self.stealth_method is not None else None,
            self.stealth_method.alpha if self.stealth_method is not None else None)

        for epoch in range(self.num_epochs):

            for batch_x, batch_y in dataset.get_data():
                # print(f"LR: {mal_optimizer._decayed_lr(var_dtype=tf.float32)}")

                predictions = model(batch_x, training=True)
                total_loss = loss_object_with_reg(y_true=batch_y, y_pred=predictions)

                grads = tape.gradient(total_loss, model.trainable_variables)
                honest_optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # logger.info(f"Epoch {epoch}: {batch_counter}")
                # batch_counter = batch_counter + 1
                break

        return model

    # def local_train_honest(self, tape, dataset, model, num_clients=1):
    #     # TODO: Local fl training steps?
    #     for (batch_x, batch_y) in dataset.get_data():
    #         predictions = model(batch_x, training=True)
    #         loss_value = self.loss_object(y_true=batch_y, y_pred=predictions)
    #
    #         # reg = tf.reduce_sum(model.losses)
    #         # total_loss = loss_value + reg
    #
    #         grads = tape.gradient(loss_value, model.trainable_variables)
    #         # honest optimizer?
    #         self.optimizer.apply_gradients(zip(grads, model.trainable_weights))


    def parse_params(self, num_epochs, num_batch, poison_samples, optimizer, loss_object, step_decay=None,
                     noise_level=None, anticipate_steps=7, model_type="lenet5_mnist", optimization_method=None, fl_no_models=10, regular_train=False):
        self.num_epochs = num_epochs
        self.num_batch = num_batch
        self.poison_samples = poison_samples
        self.optimizer = optimizer
        self.loss_object = loss_object
        self.step_decay = step_decay
        self.noise_level = noise_level
        self.anticipate_steps = anticipate_steps
        self.model_type = model_type
        self.optimization_method = optimization_method
        self.fl_no_models = fl_no_models
        self.regular_train = regular_train

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
