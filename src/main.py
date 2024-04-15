# -*- coding: utf-8 -*-
"""
@author: zgz

"""
import os
import time
import torch
import random
import string
import argparse
import warnings
import setproctitle
import numpy as np

import data.data as DATA
import data.dataset as DATASET
import models.simulators as SIMULATOR
import utils.utils as UTILS
import utils.data_loader as LOADER

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description='AI pedestrian simulation')
    parser.add_argument('--exp_name', type=str, default='pedsim_debug', help='experiment name')
    parser.add_argument('--user_name', type=str, default='guozhen', help='user name')
    parser.add_argument('--seed', type=int, default=666, help='seed')
    parser.add_argument('-f', '--finetune_flag', action='store_true', help='finetune')
    parser.add_argument('--data_config', type=str, default='./configs/data_configs/toy.yaml', help='config file of prepared synthetic dataset')
    parser.add_argument('--ft_data_config', type=str, default='./configs/data_configs/toy_f.yaml', help='config file of prepared real dataset')

    parser.add_argument('--model', type=str, default='pinnsf_m', help='model')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--gpus', type=str, default="3", help='available gpus')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=3, help='batch size')
    parser.add_argument('--ft_batch_size', type=int, default=4, help='finetune batch size')
    parser.add_argument('--shuffle', action='store_true', help='shuffle')
    parser.add_argument('--num_workers', type=int, default=0, help='workers')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--epochs', type=int, default=2, help='number of maximum epochs')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--n_embedding', type=int, default=10, help='embedding size')
    parser.add_argument('--hidden_size', type=int, default=32, help='number of hidden nodes in each layer')
    parser.add_argument('--activation', type=str, default='relu', help='activation')
    parser.add_argument('--patience', type=int, default=1, help='patience during training')
    parser.add_argument('--ft_patience', type=int, default=5, help='patience during finetune')

    parser.add_argument('--topk_ped', type=int, default=6, help='top k pedestrians in sight')
    parser.add_argument('--topk_obs', type=int, default=10, help='top k obstacles in sight')
    parser.add_argument('--sight_angle_ped', type=int, default=90, help='pedestrians field of view on other pedestrians')
    parser.add_argument('--sight_angle_obs', type=int, default=90, help='pedestrians field of view on obstacles')
    parser.add_argument('--dist_threshold_ped', type=int, default=4, help='pedestrians attention distance on other pedestrians')
    parser.add_argument('--dist_threshold_obs', type=int, default=4, help='pedestrians attention distance on obstacles')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='train ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='val ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test ratio')

    parser.add_argument('--encoder_hidden_size', type=int, default=128, help='encoder_hidden_size')
    parser.add_argument('--processor_hidden_size', type=int, default=128, help='processor_hidden_size')
    parser.add_argument('--decoder_hidden_size', type=int, default=64, help='decoder_hidden_size')
    parser.add_argument('--encoder_hidden_layers', type=int, default=3, help='encoder_hidden_layers')
    parser.add_argument('--processor_hidden_layers', type=int, default=16, help='processor_hidden_layers')
    parser.add_argument('--decoder_hidden_layers', type=int, default=2, help='decoder_hidden_layers')

    parser.add_argument('--add_noise_flag', action='store_true', help='add noise in training or not')
    parser.add_argument('--add_noise_std', type=float, default=0.05, help='noise_std')

    parser.add_argument('--correction_hidden_layers', type=int, default=1, help='correction_hidden_layers')
    parser.add_argument('--finetune_lr_decay', type=float, default=1, help='finetune_lr_decay')
    parser.add_argument('--finetune_wd_aug', type=int, default=1, help='')

    parser.add_argument('--num_history_velocity', type=int, default=1, help='num_history_velocity')

    parser.add_argument('--skip_frames', type=int, default=25, help='skip frames for desired speed calculation')
    parser.add_argument('--valid_steps', type=int, default=5, help='valid steps')
    parser.add_argument('--time_decay', type=float, default=1, help='loss time decay')

    parser.add_argument('--training_mode', type=str, default='normal', help='normal, mttrain, polar, ft_pointwise')
    parser.add_argument('--res_hidden_layers', type=int, default=3, help='')
    parser.add_argument('--ft_lr_decay2', type=float, default=0., help='lr decay for pinnsf_res')
    parser.add_argument('--save_configs', action='store_true', help='save configs')

    parser.add_argument('--reg_weight', type=float, default=0., help='regularization weights')
    parser.add_argument('--collision_threshold', type=float, default=0.5, help='')
    parser.add_argument('--collision_loss_weight', type=float, default=10, help='collision penalty loss, range 200+')
    parser.add_argument('--val_coll_weight', type=float, default=30, help='validation collision, range 20-30')
    parser.add_argument('--hard_collision_penalty', type=float, default=10, help='range around 2')

    parser.add_argument('--teacher_weight', type=float, default=0, help='')
    parser.add_argument('--collision_pred_weight', type=float, default=10, help='')
    parser.add_argument('--collision_focus_weight', type=float, default=10, help='')
    parser.add_argument('--new_collision_loss_flag', type=int, default=0, help='')

    parser.add_argument('--tags', type=str, default='', help='experiment tags')
    parser.add_argument('--iter_flag', type=int, default=0, help='')
    parser.add_argument('--iter_model_name_suffix', type=str, default='', help='')

    parser.add_argument('--pinnsf_interaction', type=str, default='sim', help='sim or loss')
    parser.add_argument('--dataset_name', type=str, default='ucy', help='gc1560, gc2344, ucy')
    parser.add_argument('--true_label_weight', type=float, default=0, help='')

    parser.add_argument('--collision_loss_version', type=str, default='v0', help='')

    args = parser.parse_args()
    rand_str = list(string.ascii_lowercase) + list(string.digits)
    args.model_name_suffix = ''.join(random.sample(rand_str, 8))

    return args


def set_exp_configs(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    setproctitle.setproctitle(args.exp_name + '@' + args.user_name)


if __name__ == '__main__':
    args = get_args()
    set_exp_configs(args)
    UTILS.args_printer(args)
    if args.save_configs:
        UTILS.save_exp_configs_default(args)
    start_time = time.time()

    # Prepare synthetic dataset and train simulator
    synthetic_dataset = DATASET.PointwisePedDataset()
    synthetic_dataset.load_data(args.data_config)
    print('number of training dataset: ', len(synthetic_dataset.raw_data['train']))
    synthetic_dataset.build_dataset(args)
    train_loaders = LOADER.data_loader(synthetic_dataset.train_data, args.batch_size,
                                       args.seed, shuffle=args.shuffle, drop_last=True)
    simulator = SIMULATOR.BaseSimulator(args)

    # Train simulator
    simulator.train(train_loaders, synthetic_dataset.valid_data)
    if 'test_data' in dir(synthetic_dataset):
        simulator.test_multiple_rollouts(synthetic_dataset.test_data)

    # Finetune simulator
    if args.finetune_flag:
        real_dataset = DATASET.PointwisePedDataset()
        real_dataset.load_data(args.ft_data_config)
        real_dataset.build_dataset(args)
        train_loaders = LOADER.data_loader(real_dataset.train_data, args.f_batch_size,
                                           args.seed, shuffle=args.shuffle, drop_last=True)
        simulator.finetune(train_loaders, real_dataset.valid_data, real_dataset.test_data)

    print('Total train time: {}'.format(time.time() - start_time))

    # Test #collision metric
    vis_dataset = DATASET.TimeIndexedPedDatasetforVis()
    vis_dataset.load_data('./configs/data_configs/data_vis.yaml')
    vis_dataset.build_dataset(args)
    simulator = SIMULATOR.BaseSimulator(args)
    simulator.load_model(args, set_model=True, finetune_flag=args.finetune_flag)
    with torch.no_grad():
        simulator.model.eval()
        for i, data in enumerate(vis_dataset.dataset['vis']):
            data_for_visual = simulator.get_multiple_rollouts(data, load_model=False)
            collision_count = DATA.Pedestrians.collision_detection(data_for_visual.position, 0.5)
            hard_collision_count = DATA.Pedestrians.collision_detection(data_for_visual.position, 0.5 / 2)
            print('#collisions soft/hard: {} / {}'.format(
                torch.sum(collision_count).item(), torch.sum(hard_collision_count).item()))
            raw = vis_dataset.raw_data['vis'][i]
