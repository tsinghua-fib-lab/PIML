# -*- coding: utf-8 -*-
"""
@author: zgz

"""
import argparse
import os
import numpy as np
import warnings

from pysr import pysr, best

import data.dataset as DATASET
import models.simulators as SIMULATOR
from utils import utils as UTILS

os.environ["PATH"] += ";C:/Users/111/AppData/Local/Programs/Julia-1.7.1/bin"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description='AI pedestrian simulation')
    parser.add_argument('-s', '--model_name_suffix', type=str, default='ptyheafr', help='model_name_suffix')
    parser.add_argument('-mnsd', '--model_name_suffix_default', type=str, default='ptyheafr', help='')
    parser.add_argument('-d', '--fit_dim', type=int, default=0, help='')
    parser.add_argument('-ppf', '--post_processing_flag', type=int, default=1, help='')
    parser.add_argument('-dp', '--data_path', type=str, default='./configs/data_configs/data_symbolic.yaml')
    parser.add_argument('-u', '--unary_ops', type=str, default="['exp', 'cos']")
    parser.add_argument('-sbs', '--sym_batch_size', type=int, default=500)
    parser.add_argument('-per', '--percentile', type=int, default=75)
    args = parser.parse_args()

    return args


def symbolic_regression(X, y, args):
    equations = pysr(
        X,
        y,
        procs=4,
        populations=8,
        niterations=10,
        # batching=True,
        # batchSize=args.sym_batch_size,
        multithreading=True,
        binary_operators=["+", "*"],
        unary_operators=args.unary_ops,
        # "inv(x) = 1/x",  # Define your own operator! (Julia syntax)
    )
    return equations


def post_filter(features, labels, seed):
    """

    Args:
        seed:
        features: n, 6
        labels: n

    Returns:

    """
    # labels的值分布

    print('max labels: ', np.max(labels))
    print('min labels: ', np.min(labels))
    print('max features: ', np.max(features, axis=0))
    print('min features: ', np.min(features, axis=0))

    n_bin = 200
    min_sampling_points = 40
    max_label = np.max(labels)
    min_label = np.min(labels)
    interval = np.floor((labels - min_label) * n_bin / (max_label - min_label))
    interval[interval == n_bin] -= 1
    interval = interval.astype(int)

    hist, bins = np.histogram(labels, bins=n_bin)
    threshold = (min_sampling_points / hist) * (np.log10(hist) + 1) ** 2
    threshold[threshold > 1] = 1
    prob_thre = np.array([threshold[i] for i in interval])

    np.random.seed(seed)
    p = np.random.uniform(0, 1, labels.shape)
    filter_idx = np.zeros(labels.shape)
    filter_idx[p < prob_thre] = 1
    features = features[filter_idx > 0]
    labels = labels[filter_idx > 0]

    print(hist)
    print(bins)
    hist, bins = np.histogram(features[:, 1], bins=n_bin)
    print(hist)
    print(bins)
    #
    # from matplotlib import pyplot as plt
    # plt.hist(labels, bins=bins)
    # plt.show()

    return features, labels


def direction_filter(features, labels, percentile):
    # 把力的绝对值比较小的点去掉
    magnitude = labels[:, 0]
    print(np.max(magnitude), np.min(magnitude))
    direction = labels[:, 1]
    percentile_threshold = np.percentile(magnitude, percentile)
    print('percentile_threshold: ', percentile_threshold)
    direction = direction[magnitude > percentile_threshold]
    features = features[magnitude > percentile_threshold, :]
    return features, direction


if __name__ == '__main__':
    args_ = get_args()
    args = UTILS.load_exp_configs_default(args_.model_name_suffix_default)
    args.fit_dim = args_.fit_dim
    args.device = 'cpu'
    args.finetune_data_path = args_.data_path
    args.model_name_suffix = args_.model_name_suffix
    args.post_processing_flag = args_.post_processing_flag
    print(args_.unary_ops)
    args.unary_ops = eval(args_.unary_ops)
    args.sym_batch_size = args_.sym_batch_size
    args.obs_feature_dim = 6

    # 后加入的属性：
    args.dataset_name = 'ucy'
    args.pinnsf_interaction = 'sim'
    args.collision_loss_version = 'v0'

    UTILS.args_printer(args)

    if args.training_mode == 'polar':
        real_dataset = DATASET.PointwisePedDatasetPolar()
    else:
        real_dataset = DATASET.PointwisePedDataset()
    real_dataset.load_data(args.finetune_data_path)
    real_dataset.build_dataset(args)

    simulator = SIMULATOR.BaseSimulator(args)
    if args.training_mode == 'polar':
        features, labels = simulator.prepare_symbolic_regression_data_polar(real_dataset.train_data)
    else:
        features, labels = simulator.prepare_symbolic_regression_data(real_dataset.train_data)

    if args.fit_dim == 0:
        labels = labels[:, 0]
        # features = np.concatenate((features[:, 0:1], features[:, 2:3], np.cos(features[:, 1:2] - features[:, 3:4])), axis=-1)
        # features = np.concatenate((features[:, 0:1], features[:, 2:3], np.cos(features[:, 1:2] - features[:, 3:4]), features[:, 5:6]), axis=-1)
        features = np.concatenate((features[:, 0:1], np.cos(features[:, 1:2] - features[:, 3:4]), features[:, 5:6]), axis=-1)
        # features = features[:, 0:4]
    else:
        features, labels = direction_filter(features, labels, percentile=args_.percentile)
        features = np.concatenate((features[:, 1:2], features[:, 3:4], features[:, 5:6]), axis=-1)

    print('before processing: ', labels.shape)
    if args.post_processing_flag:
        features, labels = post_filter(features, labels, args.seed)
        print('after processing: ', labels.shape)
    # features = features[:, :3]

    equations = symbolic_regression(features, labels, args)
    print(equations)
