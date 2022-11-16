from src.attack_dataset_config import AttackDatasetConfig
from src.backdoor.edge_case_attack import EdgeCaseAttack
from src.client_attacks import Attack
from src.data.tf_data import Dataset
from src.data.tf_data_global import GlobalDataset, IIDGlobalDataset, NonIIDGlobalDataset, DirichletDistributionDivider
from src.config.definitions import Config
from src.data.leaf_loader import load_leaf_dataset, process_text_input_indices, process_char_output_indices
import numpy as np


def load_global_dataset(config, malicious_clients, attack_dataset) -> GlobalDataset:
    """Loads dataset according to config parameter, returns GlobalData
    :type config: Config
    :type malicious_clients: np.array boolean list of clients malicious state
    """

    attack_type = Attack(config.client.malicious.attack_type) \
        if config.client.malicious is not None else None
    dataset: GlobalDataset

    if attack_type == Attack.BACKDOOR and attack_dataset.type == 'edge':
        pass # We are reloading in edge
    else:
        (dataset, (x_train, y_train)) = get_dataset(config, attack_dataset)

    if attack_type == Attack.BACKDOOR:
        attack_ds_config: AttackDatasetConfig = attack_dataset

        if attack_ds_config.type == 'semantic':
            assert attack_ds_config.train != [] and attack_ds_config.test, \
                "Must set train and test for a semantic backdoor!"
            # Based on pre-chosen images
            build_attack_selected_aux(dataset, x_train, y_train,
                                      attack_ds_config.train,
                                      attack_ds_config.test,
                                      attack_ds_config.target_label,
                                      [], #config['backdoor_feature_benign_regular'],
                                      attack_ds_config.remove_from_benign_dataset, None)
        elif attack_ds_config.type == 'semantic_pixel_pattern':
            assert attack_ds_config.train != [] and attack_ds_config.test, \
                "Must set train and test for a semantic backdoor!"
            # Based on pre-chosen images
            build_attack_selected_aux(dataset, x_train, y_train,
                                      attack_ds_config.train,
                                      attack_ds_config.test,
                                      attack_ds_config.target_label,
                                      [], #config['backdoor_feature_benign_regular'],
                                      attack_ds_config.remove_from_benign_dataset,
                                      attack_ds_config.trigger_position)
        elif attack_ds_config.type == 'tasks':
            # Construct 'backdoor tasks'
            build_attack_backdoor_tasks(dataset, malicious_clients,
                                        attack_ds_config.tasks,
                                        [attack_ds_config.source_label, attack_ds_config.target_label],
                                        attack_ds_config.aux_samples,
                                        attack_ds_config.augment_times)
        elif attack_ds_config.type == 'tasks_pixel_pattern':
            build_attack_backdoor_tasks_pixel_pattern(dataset, malicious_clients,
                                        attack_ds_config.tasks,
                                        [attack_ds_config.source_label, attack_ds_config.target_label],
                                        attack_ds_config.aux_samples,
                                        attack_ds_config.augment_times,
                                        attack_ds_config.trigger_position)
        elif attack_ds_config.type == 'edge':
            assert attack_ds_config.edge_case_type is not None, "Please specify an edge case type"

            # We have to reload the dataset adding the benign samples.
            (x_aux_train, mal_aux_labels_train), (x_aux_test, mal_aux_labels_test), (benign_x, benign_y) =\
                build_edge_case_attack(attack_ds_config.edge_case_type, attack_ds_config.edge_case_p,
                                       config.dataset.normalize_mnist_data)
            (dataset, (x_t_tst, _)) = get_dataset(config, attack_dataset, benign_x, benign_y)

            (dataset.x_aux_train, dataset.mal_aux_labels_train), (dataset.x_aux_test, dataset.mal_aux_labels_test) = \
                (x_aux_train, mal_aux_labels_train), (x_aux_test, mal_aux_labels_test)

        elif attack_ds_config.type == 'pixel_pattern':
            # do nothing
            build_pixel_pattern(dataset, attack_ds_config.target_label)
        else:
            raise NotImplementedError(f"Backdoor type {attack_ds_config.type} not supported!")
    elif attack_type == 'untargeted':
        pass
    else:
        pass # silent fail for now

    return dataset


def build_attack_backdoor_tasks(dataset, malicious_clients,
                                backdoor_tasks, malicious_objective, aux_samples, augment_times):
    dataset.build_global_aux(malicious_clients,
                             backdoor_tasks,
                             malicious_objective,
                             aux_samples,
                             augment_times)


def build_attack_backdoor_tasks_pixel_pattern(dataset, malicious_clients,
                                backdoor_tasks, malicious_objective, aux_samples, augment_times, trigger_position):
    dataset.build_global_aux(malicious_clients,
                             backdoor_tasks,
                             malicious_objective,
                             aux_samples,
                             augment_times)

    def pixel_pattern(images, tp):
        triggersize = 4
        trigger = np.ones((images.shape[0], triggersize, triggersize, images.shape[-1]))
        images[:, tp:(triggersize+tp), tp:(triggersize+tp), :] = trigger
        return images

    dataset.x_aux_train = pixel_pattern(dataset.x_aux_train, trigger_position)
    dataset.x_aux_test = pixel_pattern(dataset.x_aux_test, trigger_position)


def build_attack_selected_aux(ds, x_train, y_train,
                              backdoor_train_set, backdoor_test_set, backdoor_target,
                              benign_train_set_extra, remove_malicious_samples, trigger_position):
    """Builds attack based on selected backdoor images"""
    (ds.x_aux_train, ds.y_aux_train), (ds.x_aux_test, ds.y_aux_test) = \
        (x_train[np.array(backdoor_train_set)],
         y_train[np.array(backdoor_train_set)]), \
        (x_train[np.array(backdoor_test_set)],
         y_train[np.array(backdoor_test_set)])
    ds.mal_aux_labels_train = np.repeat(backdoor_target,
                                        ds.y_aux_train.shape).astype(np.uint8)
    ds.mal_aux_labels_test = np.repeat(backdoor_target, ds.y_aux_test.shape).astype(np.uint8)

    if benign_train_set_extra:
        extra_train_x, extra_train_y = x_train[np.array(benign_train_set_extra)], \
                                       y_train[np.array(benign_train_set_extra)]
        ds.x_aux_train = np.concatenate([ds.x_aux_train, extra_train_x])
        ds.y_aux_train = np.concatenate([ds.y_aux_train, extra_train_y])
        ds.mal_aux_labels_train = np.concatenate([ds.mal_aux_labels_train, extra_train_y])

    if trigger_position is not None:
        def pixel_pattern(images, tp):
            triggersize = 4
            # 0.6 because normalization
            trigger = np.full((images.shape[0], triggersize, triggersize, images.shape[-1]), 0.6)
            trigger[:, :, :, 2] = 0
            images[:, tp:(triggersize + tp), tp:(triggersize + tp), :] = trigger
            return images

        ds.x_aux_train = pixel_pattern(ds.x_aux_train, trigger_position)
        ds.x_aux_test = pixel_pattern(ds.x_aux_test, trigger_position)

    if remove_malicious_samples:
        np.delete(x_train, backdoor_train_set, axis=0)
        np.delete(y_train, backdoor_train_set, axis=0)
        np.delete(x_train, backdoor_test_set, axis=0)
        np.delete(y_train, backdoor_test_set, axis=0)



def shuffle(x, y):
    perms = np.random.permutation(x.shape[0])
    return x[perms, :], y[perms]

def build_edge_case_attack(edge_case, adv_edge_case_p, normalize_mnist_data):
    attack: EdgeCaseAttack = factory(edge_case)
    (x_aux_train, mal_aux_labels_train), (x_aux_test, mal_aux_labels_test), (orig_y_train, _) =\
        attack.load()

    if normalize_mnist_data:
        emnist_mean, emnist_std = 0.036910772, 0.16115953
        x_aux_train = (x_aux_train - emnist_mean) / emnist_std
        x_aux_test = (x_aux_test - emnist_mean) / emnist_std

    # If necessary, distribute edge_case samples by p

    # TODO: Fix shuffle for orig_y_train, only not working when labels differ!!!
    x_aux_train, mal_aux_labels_train = shuffle(x_aux_train, mal_aux_labels_train)
    x_aux_test, mal_aux_labels_test = shuffle(x_aux_test, mal_aux_labels_test)

    x_benign, y_benign = None, None
    if adv_edge_case_p < 1.0:
        # Some edge case values must be incorporated into the benign training set.
        index = int(adv_edge_case_p * x_aux_train.shape[0])
        x_benign, y_benign = x_aux_train[index:, :], orig_y_train[index:]
        x_aux_train, mal_aux_labels_train = x_aux_train[:index, :], mal_aux_labels_train[:index]

    return (x_aux_train, mal_aux_labels_train), (x_aux_test, mal_aux_labels_test), (x_benign, y_benign)
    # Note: ds.y_aux_train, ds.y_aux_test not set


def build_pixel_pattern(ds, backdoor_target):
    # (ds.x_aux_train, ds.y_aux_train), (ds.x_aux_test, ds.y_aux_test) = \
    #     (ds.x_train, ds.y_train), \
    #     (ds.x_test, ds. y_test)
    # ds.mal_aux_labels_train = np.repeat(backdoor_target,
    #                                     ds.y_aux_train.shape).astype(np.uint8)
    # ds.mal_aux_labels_test = np.repeat(backdoor_target, ds.y_aux_test.shape).astype(np.uint8)

    # Assign test set
    (ds.x_aux_test, ds.y_aux_test) = \
        (ds.x_test, ds. y_test)
    ds.mal_aux_labels_test = np.repeat(backdoor_target, ds.y_aux_test.shape).astype(np.uint8)


def factory(classname):
    from src.backdoor import edge_case_attack
    cls = getattr(edge_case_attack, classname)
    return cls()


def get_dataset(config, attack_ds_config, add_x_train=None, add_y_train=None):
    """

    @param config:
    @param attack_ds_config:
    @param add_x_train: x_train samples to add to training set
    @param add_y_train: y_train samples to add to training set
    @return:
    """
    dataset = config.dataset.dataset
    number_of_samples = config.dataset.number_of_samples
    data_distribution = config.dataset.data_distribution
    normalize_mnist_data = config.dataset.normalize_mnist_data  # Legacy
    num_clients = config.environment.num_clients


    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = Dataset.get_mnist_dataset(number_of_samples)
        if add_x_train is not None:
            x_train = np.concatenate([x_train, add_x_train])
            y_train = np.concatenate([y_train, add_y_train])
        if data_distribution == 'IID':
            ds = IIDGlobalDataset(x_train, y_train, num_clients=num_clients, x_test=x_test, y_test=y_test)
        else:
            (x_train_dist, y_train_dist) = \
                DirichletDistributionDivider(x_train, y_train, attack_ds_config.train,
                                             attack_ds_config.test,
                                             attack_ds_config.remove_from_benign_dataset,
                                             num_clients).build()
            ds = NonIIDGlobalDataset(x_train_dist, y_train_dist, x_test, y_test, num_clients=num_clients)

    elif dataset == 'fmnist':
        if data_distribution == 'IID':
            (x_train, y_train), (x_test, y_test) = Dataset.get_fmnist_dataset(number_of_samples)
            if add_x_train is not None:
                x_train = np.concatenate([x_train, add_x_train])
                y_train = np.concatenate([y_train, add_y_train])
            ds = IIDGlobalDataset(x_train, y_train, num_clients=num_clients, x_test=x_test, y_test=y_test)
        else:
            raise Exception('Distribution not supported')

    elif dataset == 'femnist':
        if data_distribution == 'IID':
            (x_train, y_train), (x_test, y_test) = Dataset.get_emnist_dataset(number_of_samples,
                                                                              num_clients,
                                                                              normalize_mnist_data)
            (x_train, y_train), (x_test, y_test) = (
                Dataset.keep_samples(np.concatenate(x_train), np.concatenate(y_train), number_of_samples),
                Dataset.keep_samples(np.concatenate(x_test), np.concatenate(y_test), number_of_samples))

            if add_x_train is not None:
                x_train = np.concatenate([x_train, add_x_train])
                y_train = np.concatenate([y_train, add_y_train])

            ds = IIDGlobalDataset(x_train, y_train, num_clients, x_test, y_test)
        else:
            (x_train, y_train), (x_test, y_test) = Dataset.get_emnist_dataset(number_of_samples,
                                                                              num_clients,
                                                                              normalize_mnist_data)

            if add_x_train is not None:
                # Here, x_train and y_train are already separated by handwriter.. Add to random handwriters
                handwriter_indices = np.random.choice(len(x_train), add_x_train.shape[0], replace=True)
                for index in handwriter_indices:
                    x_train[index] = np.concatenate([x_train[index], add_x_train[index:(index+1), :]])
                    y_train[index] = np.concatenate([y_train[index], add_y_train[index:(index + 1)]])

                for index in range(len(x_train)):
                    x_train[index], y_train[index] = shuffle(x_train[index], y_train[index])

            ds = NonIIDGlobalDataset(x_train, y_train, np.concatenate(x_test), np.concatenate(y_test),
                                     num_clients)
            x_train, y_train = np.concatenate(x_train), np.concatenate(y_train) # For aux

    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = Dataset.get_cifar10_dataset(number_of_samples)

        if add_x_train is not None:
            x_train = np.concatenate([x_train, add_x_train])
            y_train = np.concatenate([y_train, add_y_train])

        if data_distribution == 'IID':
            ds = IIDGlobalDataset(x_train, y_train, num_clients=num_clients, x_test=x_test, y_test=y_test)

        else:
            if attack_ds_config is not None:
                (x_train_dist, y_train_dist) = \
                    DirichletDistributionDivider(x_train, y_train, attack_ds_config.train,
                                                 attack_ds_config.test,
                                                 attack_ds_config.remove_from_benign_dataset,
                                                 num_clients).build()
            else:
                (x_train_dist, y_train_dist) = \
                    DirichletDistributionDivider(x_train, y_train, [],
                                                 [],
                                                 False,
                                                 num_clients).build()
            ds = NonIIDGlobalDataset(x_train_dist, y_train_dist, x_test, y_test, num_clients=num_clients)

    elif dataset == 'shakespeare':

        users, train_data, test_data = load_leaf_dataset("shakespeare")

        if data_distribution == "IID":

            x_train = [process_text_input_indices(train_data[user]['x']) for user in users]
            y_train = [process_char_output_indices(train_data[user]['y']) for user in users]

            x_test = np.concatenate([process_text_input_indices(test_data[user]['x']) for user in users])
            y_test = np.concatenate([process_char_output_indices(test_data[user]['y']) for user in users])

            if num_clients == 1:
                x_train = [np.concatenate(x_train)]
                y_train = [np.concatenate(y_train)]
                ds = NonIIDGlobalDataset(x_train, y_train, x_test, y_test, num_clients=num_clients)
            else:
                x_train = np.concatenate(x_train)
                y_train = np.concatenate(y_train)
                ds = IIDGlobalDataset(x_train, y_train, num_clients, x_test, y_test)

        else:
            if num_clients == len(users):

                # selected = np.random.choice(users, num_clients, replace=False)
                selected = users
                x_train = [process_text_input_indices(train_data[user]['x']) for user in selected]
                y_train = [process_char_output_indices(train_data[user]['y']) for user in selected]

                x_test = np.concatenate([process_text_input_indices(test_data[user]['x']) for user in selected])
                y_test = np.concatenate([process_char_output_indices(test_data[user]['y']) for user in selected])

                ds = NonIIDGlobalDataset(x_train, y_train, x_test, y_test, num_clients=num_clients)


            else:
                raise Exception("Smaller number of users in non-iid not supported!")


    else:
        raise Exception('Selected dataset with distribution not supported')

    return ds, (x_train, y_train)

