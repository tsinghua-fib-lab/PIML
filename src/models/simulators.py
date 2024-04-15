# -*- coding: utf-8 -*-
"""
@author: zgz

"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append('..')

import models.model as MODEL
import data.data as DATA
import functions.metrics as METRIC
import utils.utils as UTILS


class BaseSimulator(DATA.Pedestrians):
    """docstring for BaseSimulator"""

    def __init__(self, args):
        super(BaseSimulator, self).__init__()
        self.args = args
        self.set_model(args)
        self.set_optimizer(args)
        self.set_scheduler(args)
        self.finetune_flag = False
        self.test_flag = False

        train_params = list(filter(lambda x: x.requires_grad, self.model.parameters()))
        print('#Trainable Parameters:', np.sum([p.numel() for p in train_params]))

    def set_model(self, args):
        print('\n------------- Initialize Model -------------')
        if args.model == 'base':
            self.model = MODEL.BaseSimModel(args)
        elif args.model in {'pinnsf', 'pinnsf_res'}:
            self.model = MODEL.PINNSF(args)
        elif args.model == 'pinnsf2':
            self.model = MODEL.PINNSF2(args)
        elif args.model == 'base_test':
            self.model = MODEL.Base_test(args)
        elif args.model == 'pinnsf_polar':
            self.model = MODEL.PINNSF_polar(args)
        elif args.model == 'pinnsf_bottleneck':
            self.model = MODEL.PINNSF_bottleneck(args)
        elif args.model == 'pinnsf_pb':
            self.model = MODEL.PINNSF_polar_bottleneck(args)
        elif args.model == 'pinnsf_pbc':
            self.model = MODEL.PINNSF_polar_bottleneck_collision(args)
        elif args.model == 'pinnsf_bm':
            self.model = MODEL.PINNSF_bottleneck_multitask(args)
        elif args.model == 'pinnsf_m':
            self.model = MODEL.PINNSF_multitask(args)
        else:
            raise NotImplementedError(args.model)
        if len(args.gpus) > 2:
            self.model = nn.DataParallel(self.model.to(args.device))
        else:
            self.model = self.model.to(args.device)

    def set_optimizer(self, args):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def set_scheduler(self, args):
        # scheduler = StepLR(optimizer, step_size=args.scheduler_step_size,
        #                    gamma=args.scheduler_gamma)
        self.scheduler = None

    def set_ft_model(self, args):
        if args.model == 'base':
            self.model = MODEL.BaseNDSimModel(args)
        elif args.model == 'base_test':
            self.model = MODEL.Base_test(args)
        elif args.model == 'pinnsf':
            self.model = MODEL.PINNSF(args)
        elif args.model == 'pinnsf2':
            self.model = MODEL.PINNSF2(args)
        elif args.model == 'pinnsf_polar':
            self.model = MODEL.PINNSF_polar(args)
        elif args.model == 'pinnsf_bottleneck':
            self.model = MODEL.PINNSF_bottleneck(args)
        elif args.model == 'pinnsf_pb':
            self.model = MODEL.PINNSF_polar_bottleneck(args)
        elif args.model == 'pinnsf_res':
            self.model = MODEL.PINNSF_residual(args)
        elif args.model == 'pinnsf_pbc':
            self.model = MODEL.PINNSF_polar_bottleneck_collision(args)
        elif args.model == 'pinnsf_bm':
            self.model = MODEL.PINNSF_bottleneck_multitask(args)
        elif args.model == 'pinnsf_m':
            self.model = MODEL.PINNSF_multitask(args)
        else:
            raise NotImplementedError(args.model)
        if len(args.gpus) > 2:
            self.model = nn.DataParallel(self.model.to(args.device))
        else:
            self.model = self.model.to(args.device)

    def set_ft_optimizer(self, args):
        if args.model in {'base', 'pinnsf_res'}:
            if isinstance(self.model, nn.DataParallel):
                corrector_params = list(map(id, self.model.module.corrector.parameters()))
                pretrained_params = filter(lambda p: id(p) not in corrector_params, self.model.parameters())
                params = [
                    {'params': self.model.module.corrector.parameters(), 'lr': args.learning_rate * args.ft_lr_decay2},
                    {'params': pretrained_params, 'lr': args.learning_rate * args.finetune_lr_decay}]
                self.optimizer = torch.optim.Adam(
                    params, lr=args.learning_rate, weight_decay=args.weight_decay)
            else:
                corrector_params = list(map(id, self.model.corrector.parameters()))
                pretrained_params = filter(lambda p: id(p) not in corrector_params, self.model.parameters())
                params = [{'params': self.model.corrector.parameters(), 'lr': args.learning_rate * args.ft_lr_decay2},
                          {'params': pretrained_params, 'lr': args.learning_rate * args.finetune_lr_decay}]
                self.optimizer = torch.optim.Adam(
                    params, lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.model in {'pinnsf', 'pinnsf2', 'pinnsf_polar', 'pinnsf_bottleneck', 'pinnsf_pbc', 'pinnsf_pb',
                            'base_test', 'pinnsf_bm', 'pinnsf_m'}:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=args.learning_rate * args.finetune_lr_decay,
                weight_decay=args.weight_decay * args.finetune_wd_aug)
        else:
            raise NotImplementedError('optimizer for {} has not been implemented'.format(args.model))

    def set_ft_scheduler(self, args):
        # scheduler = StepLR(optimizer, step_size=args.scheduler_step_size,
        #                    gamma=args.scheduler_gamma)
        self.scheduler = None

    def set_loss_func(self, args):
        pass

    @staticmethod
    def get_dist(pred, label):
        """
        pred: (N, 2)
        label: (N, 2)
        """
        return torch.norm(pred - label, p=2, dim=-1)

    @staticmethod
    def get_features_from_data(data, idx):
        state = torch.cat((data.position[..., idx], data.velocity[..., idx],
                           data.acceleration[..., idx], data.destination), dim=1)
        return state

    @staticmethod
    def reduction(values, mode):
        if mode == 'sum':
            return torch.sum(values)
        elif mode == 'mean':
            return torch.mean(values)
        elif mode == 'none':
            return values
        else:
            raise NotImplementedError

    def loss_func(self, pred, labels, reduction='none'):
        return F.mse_loss(pred, labels, reduction=reduction)

    def l1_reg_loss(self, embeddings, weight=1e-3, reduction='none'):
        return self.reduction(weight * torch.abs(embeddings), reduction)

    def multiple_rollout_mse_loss(self, pred, labels, time_decay, reduction='none', reverse=False):
        """
        multiple rollout training loss with time decay
        Args:
            reverse:
            time_decay:
            pred: c, t, n, 2
            labels:
            reduction:

        Returns:

        """
        loss = (pred - labels) * (pred - labels)
        if not reverse:
            decay = torch.tensor([time_decay ** (pred.shape[1] - t - 1) for t in range(pred.shape[1])],
                                 device=pred.device)
        else:
            decay = torch.tensor([time_decay ** t for t in range(pred.shape[1])], device=pred.device)
        decay = decay.reshape(1, int(pred.shape[1]), 1, 1)
        loss = loss * decay
        return self.reduction(loss, reduction)

    def multiple_rollout_collision_loss(self, pred, labels, time_decay, coll_focus_weight, collisions,
                                        reduction='none', abnormal_mask=None):
        """
        multiple rollout training loss with time decay
        Args:
            collisions: c, t, n
            time_decay:
            pred: c, t, n, 2
            labels: c, t, n, 2
            reduction:

        Returns:

        """
        collisions = torch.sum(collisions, dim=1)  # c, n
        collisions[collisions > 0] = 1.  # 只看有没有
        collision_w = collisions
        collision_w = collision_w.unsqueeze(1).repeat(1, pred.shape[1], 1)
        collision_w = collision_w.unsqueeze(-1)  # c, t, n, 1

        mse_loss = self.multiple_rollout_mse_loss(pred, labels, time_decay, reduction='none')
        collision_focus_loss = self.multiple_rollout_collision_avoidance_loss(pred, labels, time_decay,
                                                                              reduction='none')
        # loss = collision_w * (mse_loss + collision_focus_loss * coll_focus_weight)
        loss = collision_w * collision_focus_loss  # c,t,n,2

        if abnormal_mask is not None:
            abnormal_mask = abnormal_mask.reshape(1, 1, -1, 1)
            loss = loss * abnormal_mask

        return self.reduction(loss, reduction)

    def multiple_rollout_collision_avoidance_loss(self, pred, labels, time_decay, reduction='none'):
        """
        multiple rollout training loss with time decay
        Args:
            weight:
            time_decay:
            pred: c, t, n, 2
            labels: c, t, n, 2
            reduction:

        Returns:

        """
        ni = labels[:, -1:, :, :] - labels[:, 0:1, :, :]
        ni_norm = torch.norm(ni, p=2, dim=-1, keepdim=True)
        ni_norm = ni_norm + 1e-6
        ni = ni / ni_norm  # c,1,n,2

        pred_ = pred - torch.sum(pred * ni, dim=-1, keepdim=True) * ni
        labels_ = labels - torch.sum(labels * ni, dim=-1, keepdim=True) * ni

        loss = self.multiple_rollout_mse_loss(pred_, labels_, time_decay, reduction='none')
        return self.reduction(loss, reduction)

    def save_simulator(self):
        pass

    def load_model(self, args, set_model=True, finetune_flag=True, load_path=''):
        if set_model:
            if finetune_flag:
                self.set_ft_model(args)
            else:
                self.set_model(args)

        if not load_path:
            load_path = '../saved_model/{}_{}'.format(args.exp_name, args.model_name_suffix)
            if finetune_flag:
                load_path += '_finetuned'
        loaded_dict = torch.load(load_path)
        print('-------- load model from {} ----------'.format(load_path))

        if not (isinstance(self.model, nn.DataParallel)) and list(loaded_dict.keys())[0][:6] == 'module':  # 多卡模型，单卡加载
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in loaded_dict.items():
                name = k[7:]  # module字段在最前面，从第7个字符开始就可以去掉module
                new_state_dict[name] = v  # 新字典的key值对应的value一一对应
            loaded_dict = new_state_dict

        self.model.load_state_dict(loaded_dict)

    def save_model(self, args, finetune_flag=True, cpu_version=False):
        os.makedirs('../saved_model', exist_ok=True)
        save_path = f'../saved_model/{args.exp_name}_{args.model_name_suffix}'
        if finetune_flag:
            save_path += '_finetuned'
        if cpu_version:
            save_path += '_cpu'
            self.model = self.model.cpu()
        torch.save(self.model.state_dict(), save_path)

        if cpu_version:
            self.model = self.model.to(args.device)

    def train(self, train_loaders, val_data, test_data=None):
        print('\n------------- Training -------------')
        args = self.args
        start = time.time()
        min_loss = 1e5
        patience = 0

        if self.finetune_flag:
            self.epoch = 0
            self.save_model(args, self.finetune_flag)
            val_loss, val_mse = self.validate(val_data)
            min_loss = val_loss
            if test_data:
                self.test_multiple_rollouts(test_data, load_model=False, test_flag=True)

        for epoch in range(args.epochs):
            self.epoch = epoch
            self.collision_count = 0
            self.hard_collision_count = 0

            self.model.train()
            loss_log, collision_pred_loss_log, collision_loss_log, hard_collision_loss_log, mse_loss_log, acc_pred_log, reg_loss_log = [0.] * 7
            n_train = 0
            for batch_idx, batch_data in enumerate(train_loaders):
                self.optimizer.zero_grad()
                self.batch_idx = batch_idx
                if type(batch_data) == DATA.ChanneledTimeIndexedPedData:
                    loss, mse_loss, collision_loss, hard_collision_loss, collision_pred_loss, acc_pred, reg_loss = self.test_multiple_rollouts_for_training(batch_data)
                    mse_loss_log += mse_loss.item()
                    collision_pred_loss_log += collision_pred_loss.item()
                    collision_loss_log += collision_loss.item()
                    hard_collision_loss_log += hard_collision_loss.item()
                    acc_pred_log += acc_pred.item() * args.batch_size
                    reg_loss_log += reg_loss.item()
                    loss_log += loss.item()
                    n_train += torch.sum(batch_data.mask_p_pred == 1).item()
                else:
                    ped_features, obs_features, self_features, labels = batch_data
                    n_train = n_train + labels.shape[0]
                    predictions = self.model(ped_features, obs_features, self_features)
                    pred, p_msg = predictions[0], predictions[1]

                    if args.pinnsf_interaction == 'sim':
                        mse_loss = F.mse_loss(pred, labels[:, 4:6], reduction='sum')
                    elif args.pinnsf_interaction == 'loss':
                        if args.iter_flag:
                            sf_version = 'v2'
                        else:
                            sf_version = 'v0'
                        labels_ = UTILS.calc_acceleration(ped_features, sf_version, args.dataset_name)
                        mse_loss = F.mse_loss(p_msg, labels_, reduction='sum') + args.true_label_weight * F.mse_loss(pred, labels[:, 4:6], reduction='sum')

                    loss = mse_loss
                    if args.reg_weight > 0:
                        reg_loss = self.l1_reg_loss(p_msg, args.reg_weight, 'sum')
                        loss = loss + reg_loss
                        reg_loss_log += reg_loss.item()
                    if args.collision_pred_weight > 0 and args.model == 'pinnsf_bm':
                        assert ((1 >= predictions[-1]) & (predictions[-1] >= 0)).all(), 'Error: value error in collision predictions'
                        collision_pred_loss = F.binary_cross_entropy(predictions[-1], labels[:, 6:], reduction='sum')
                        acc_pred = torch.sum(torch.round(predictions[-1]) == labels[:, 6:]) / predictions[
                            -1].numel() * args.batch_size
                        acc_pred_log += acc_pred
                        loss = loss + collision_pred_loss
                        collision_pred_loss_log = collision_pred_loss_log + collision_pred_loss.item()
                    loss_log += loss.item()
                    mse_loss_log += mse_loss.item()

                loss.backward()
                self.optimizer.step()
                self.time_iter = time.time() - start

            if self.scheduler:
                self.scheduler.step()

            loss_log = loss_log / (n_train)
            collision_pred_loss_log = collision_pred_loss_log / (n_train)
            collision_loss_log = collision_loss_log / (n_train)
            hard_collision_loss_log = hard_collision_loss_log / (n_train)
            acc_pred_log = acc_pred_log / ((self.batch_idx + 1) * args.batch_size)
            train_mse = mse_loss_log / (n_train)

            print('Epoch {}:'.format(epoch))
            print("Time {:.4f} -- Training loss:{}, mse:{}, coll_pred:{}, acc_pred:{}, coll:{}, hard_coll:{}".format(
                self.time_iter, loss_log, train_mse, collision_pred_loss_log, acc_pred_log, collision_loss_log,
                hard_collision_loss_log))

            if self.finetune_flag:
                print('training collision count hard/soft: {} & {}'.format(self.hard_collision_count,
                                                                           self.collision_count))

            val_loss, val_mse = self.validate(val_data)
            if test_data:
                self.test_multiple_rollouts(test_data, load_model=False, test_flag=True)

            if val_loss < min_loss:
                print("!!!!!!!!!! Model Saved at epoch {} !!!!!!!!!!".format(epoch))
                self.save_model(args, self.finetune_flag)
                min_loss = val_loss
                patience = 0
            else:
                patience += 1
                if patience > (args.patience if self.finetune_flag else args.ft_patience): break

    def validate(self, val_data):
        if type(val_data) == DATA.PointwisePedData:
            val_loss, val_mse = self.test_pointwise(val_data)
        elif type(val_data) == list:
            val_loss, val_mse, _, _, _ = self.test_multiple_rollouts(val_data, load_model=False, test_flag=False)
        else:
            raise NotImplementedError

        print("Time {:.4f} -- Validation loss:{}, val_mse:{}".format(
            self.time_iter, val_loss, val_mse))


        return val_loss, val_mse

    def finetune(self, train_loaders, val_data, test_data):
        print('\n------------- Finetune -------------')
        args = self.args
        self.set_ft_model(args)
        self.set_ft_optimizer(args)
        self.set_ft_scheduler(args)

        # load model
        pretrained_dict = torch.load('../saved_model/{}_{}'.format(
            args.exp_name, args.model_name_suffix))
        nd_model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in nd_model_dict}
        nd_model_dict.update(pretrained_dict)
        self.model.load_state_dict(nd_model_dict)

        # finetune training & testing
        self.finetune_flag = True
        self.train(train_loaders, val_data, test_data)
        self.test_multiple_rollouts(test_data)
        self.finetune_flag = False

    def test_pointwise(self, data):
        self.model.eval()
        with torch.no_grad():
            ped_features, obs_features, self_features, labels = data[:]
            labels = labels[:, 4:6]
            pred = self.model(ped_features, obs_features, self_features)[0]
            loss = self.loss_func(pred, labels)
            loss = torch.mean(loss).item()
            mse = loss

        return loss, mse

    @staticmethod
    def post_process(data, pred_data, pred_mask_p, mask_p):
        """
        把到达终点以后的人置为终点
        t, n, 2
        """
        waypoints = data.waypoints  # *c, d, n, 2
        if waypoints.dim() > 3:
            dest_num = data.dest_num.unsqueeze(0).repeat(waypoints.shape[0], 1)
        else:
            dest_num = data.dest_num
        dest_idx = dest_num - 1  # *c, n

        dest_idx_ = dest_idx.unsqueeze(-2).unsqueeze(-1)  # *c, 1, n, 1
        dest_idx_ = dest_idx_.repeat(*([1] * (dest_idx_.dim() - 1) + [2]))
        dest = torch.gather(waypoints, -3, dest_idx_)  # *c, 1, n, 2
        tmp_arg = [1] * dest.dim()
        tmp_arg[-3] = pred_data.shape[-3]
        dest = dest.repeat(*tmp_arg)  # *c, t, n, 2

        pred_data[(mask_p == 1) & (pred_mask_p == 0)] = dest[(mask_p == 1) & (pred_mask_p == 0)]
        return pred_data

    def test_multiple_rollouts(self, data: DATA.TimeIndexedPedData, load_model=True, test_flag=True, reduction='sum'):
        """
        todo: get_multiple_rollouts的输入应该是 起始状态、destination和对应要run多少steps
        """
        args = self.args

        if type(data) == DATA.TimeIndexedPedData:
            self.model.eval()
            with torch.no_grad():
                p_pred = self.get_multiple_rollouts(data, t_start=args.skip_frames)
                p_pred = p_pred.position
                p_pred_ = p_pred.clone()
                p_pred_[:-1, :, :] = p_pred_[1:, :, :].clone()
                p_pred = p_pred_
                mask_p_pred = data.mask_p_pred.long()  # (*c) t, n
                labels = data.labels[..., :2]  # (*c) t * n * 2

                loss = self.loss_func(p_pred[mask_p_pred == 1], labels[mask_p_pred == 1], reduction='mean')
                loss = loss.item()
                mse = loss
                mae = METRIC.mae_with_time_mask(p_pred, labels, mask_p_pred, reduction='mean')

            # the computation of ot needs gradients
            ot = METRIC.ot_with_time_mask(p_pred, labels, mask_p_pred, reduction='mean', dvs=args.device)
            mmd = METRIC.mmd_with_time_mask(p_pred, labels, mask_p_pred, reduction='mean')
            collision = METRIC.collision_count(p_pred, 0.6, reduction='sum')

        elif type(data) == list:
            self.model.eval()
            mae_sum = 0.
            mse_sum = 0.
            ot_sum = 0.
            mmd_sum = 0.
            collision_sum = 0.
            hard_collision_sum = 0.
            loss_sum = 0.
            N = 0
            T = 0
            for data_ in data:
                with torch.no_grad():
                    pred = self.get_multiple_rollouts(data_, t_start=args.skip_frames, load_model=load_model)
                    p_pred = pred.position
                    mask_p_pred = data_.mask_p_pred.long()  # *c, t, n
                    collision_count = METRIC.collision_count(p_pred[args.skip_frames:], args.collision_threshold,
                                                             reduction='sum')
                    collision_sum += collision_count
                    hard_collision_count = METRIC.collision_count(p_pred[args.skip_frames:],
                                                                  args.collision_threshold / 2, reduction='sum')
                    hard_collision_sum += hard_collision_count

                    p_pred = self.post_process(data_, p_pred, pred.mask_p, mask_p_pred)
                    labels = data_.labels[..., :2]  # *c, t, n, 2

                    mse = F.mse_loss(p_pred[mask_p_pred == 1], labels[mask_p_pred == 1], reduction='sum')
                    loss = mse = mse.item()
                    if not test_flag:  # during validation
                        loss = loss + args.val_coll_weight * (collision_count + hard_collision_count)
                        # old version: loss = loss + args.val_coll_weight * (collision_count + hard_collision_count * 2)

                # the computation of ot needs gradients
                if test_flag:
                    mae = METRIC.mae_with_time_mask(p_pred, labels, mask_p_pred, reduction='sum')
                    ot = METRIC.ot_with_time_mask(p_pred, labels, mask_p_pred, reduction='sum', dvs=args.device)
                    mmd = METRIC.mmd_with_time_mask(p_pred, labels, mask_p_pred, reduction='sum')
                    mae_sum += mae
                    ot_sum += ot
                    mmd_sum += mmd

                N += torch.sum(mask_p_pred == 1).item()
                T += torch.sum(torch.sum(mask_p_pred, dim=1) > 0).item()
                loss_sum += loss
                mse_sum += mse

            loss = loss_sum / N
            mse = mse_sum / N
            mae = mae_sum / N
            ot = ot_sum / T
            mmd = mmd_sum / T

        else:
            raise NotImplementedError

        # todo: logging部分应该剥离出去
        if test_flag:
            print('---------------------------------------')
            print("Test loss:{}, test_mse:{}, test_mae:{}, test ot:{}, test mmd:{}".format(loss, mse, mae, ot, mmd))

        print('test/val collision count hard/soft: {} & {}'.format(hard_collision_sum, collision_sum))

        return loss, mse, mae, ot, mmd

    def get_multiple_rollouts(self, data: DATA.TimeIndexedPedData, t_start=0, load_model=True):
        """
        Args:
            data:
            t_start: rollout starts from frame #t
        """
        args = self.args
        if load_model:
            self.load_model(args, set_model=False, finetune_flag=self.finetune_flag)

        destination = data.destination
        waypoints = data.waypoints
        obstacles = data.obstacles
        mask_p_ = data.mask_p_pred.clone().long()  # *c, t, n

        state_features = [data.ped_features[..., t_start, :, :, :], data.obs_features[..., t_start, :, :, :],
                          data.self_features[..., t_start, :, :]]
        desired_speed = state_features[-1][..., -1].unsqueeze(-1)  # *c, n, 1
        a_cur = data.acceleration[..., t_start, :, :]  # *c, N, 2
        v_cur = data.velocity[..., t_start, :, :]  # *c, N, 2
        p_cur = data.position[..., t_start, :, :]  # *c, N, 2
        dest_cur = data.destination[..., t_start, :, :]  # *c, N, 2
        dest_idx_cur = data.dest_idx[..., t_start, :]  # *c, N
        dest_num = data.dest_num

        p_res = torch.zeros(data.position.shape, device=args.device)  # *c, t, n, 2
        v_res = torch.zeros(data.velocity.shape, device=args.device)  # *c, t, n, 2
        a_res = torch.zeros(data.acceleration.shape, device=args.device)  # *c, t, n, 2

        p_res[..., :t_start + 1, :, :] = data.position[..., :t_start + 1, :, :]
        v_res[..., :t_start + 1, :, :] = data.velocity[..., :t_start + 1, :, :]
        a_res[..., :t_start + 1, :, :] = data.acceleration[..., :t_start + 1, :, :]

        # 维护一个mask_p
        mask_p_new = torch.zeros(mask_p_.shape, device=mask_p_.device)
        mask_p_new[..., :t_start + 1, :] = data.mask_p[..., :t_start + 1, :].long()

        new_peds_flag = (data.mask_p - data.mask_p_pred).long()  # c, t, n

        for t in tqdm(range(t_start, data.num_frames)):
            p_res[..., t, :, :] = p_cur
            v_res[..., t, :, :] = v_cur
            a_res[..., t, :, :] = a_cur
            # mask_p_new[..., t, ~p_cur[:, 0].isnan()] = 1
            mask_p_new[..., t, :][~p_cur[..., 0].isnan()] = 1

            a_next = self.model(*state_features)[0]
            v_next = v_cur + a_cur * data.time_unit
            p_next = p_cur + v_cur * data.time_unit  # *c, n, 2

            # update destination & mask_p
            out_of_bound = torch.tensor(float('nan'), device=args.device)
            dis_to_dest = torch.norm(p_cur - dest_cur, p=2, dim=-1)
            dest_idx_cur[dis_to_dest < 0.5] += 1  # *c, n

            p_next[dest_idx_cur > dest_num - 1, :] = out_of_bound  # destination arrived

            dest_idx_cur[dest_idx_cur > dest_num - 1] -= 1
            dest_idx_cur_ = dest_idx_cur.unsqueeze(-2).unsqueeze(-1)  # *c, 1, n, 1
            dest_idx_cur_ = dest_idx_cur_.repeat(*([1] * (dest_idx_cur_.dim() - 1) + [2]))
            dest_cur = torch.gather(waypoints, -3, dest_idx_cur_).squeeze()  # *c, n, 2

            # update everyone's state
            p_cur = p_next  # *c, n, 2
            v_cur = v_next
            a_cur = a_next

            # update hist_v
            hist_v = state_features[2][..., :, 2:-3]  # *c, n, 2*x
            hist_v[..., :, :-2] = hist_v[..., :, 2:]
            hist_v[..., :, -2:] = v_cur

            # add newly joined pedestrians
            if t < data.num_frames - 1:
                new_idx = new_peds_flag[..., t + 1, :]  # c, n
                if torch.sum(new_idx) > 0:
                    p_cur[new_idx == 1] = data.position[..., t + 1, :, :][new_idx == 1, :]
                    v_cur[new_idx == 1] = data.velocity[..., t + 1, :, :][new_idx == 1, :]
                    a_cur[new_idx == 1] = data.acceleration[..., t + 1, :, :][new_idx == 1, :]
                    dest_cur[new_idx == 1] = data.destination[..., t + 1, :, :][new_idx == 1, :]
                    dest_idx_cur[new_idx == 1] = data.dest_idx[..., t + 1, :][new_idx == 1]

                    # update hist_v
                    hist_v[new_idx == 1] = data.self_features[..., t + 1, :, 2:-3][new_idx == 1]

            # calculate features
            ped_features, obs_features, dest_features = self.get_relative_features(
                p_cur.unsqueeze(-3), v_cur.unsqueeze(-3), a_cur.unsqueeze(-3),
                dest_cur.unsqueeze(-3), obstacles, args.topk_ped, args.sight_angle_ped,
                args.dist_threshold_ped, args.topk_obs,
                args.sight_angle_obs, args.dist_threshold_obs)
            ped_features = ped_features.squeeze()
            obs_features = obs_features.squeeze()
            dest_features = dest_features.squeeze()

            self_features = torch.cat((dest_features, hist_v, a_cur, desired_speed), dim=-1)
            state_features = [ped_features, obs_features, self_features]

        # todo: 处理一下mask_v 和 mask_a; 现在只处理了mask_p
        output = DATA.RawData(p_res, v_res, a_res, destination, destination, obstacles,
                              mask_p_new, meta_data=data.meta_data)
        return output

    def test_multiple_rollouts_for_training(self, data: DATA.TimeIndexedPedData, t_start=0):
        """
        加入对于碰撞的惩罚，dynamic weighting;
        加入对于加速度的预测
        Args:
            data:
            t_start:

        Returns:

        """
        args = self.args

        destination = data.destination
        waypoints = data.waypoints
        obstacles = data.obstacles
        mask_p_ = data.mask_p_pred.clone().long()  # *c, t, n

        state_features = [data.ped_features[..., t_start, :, :, :], data.obs_features[..., t_start, :, :, :],
                          data.self_features[..., t_start, :, :]]
        desired_speed = state_features[-1][..., -1].unsqueeze(-1)  # *c, n, 1
        a_cur = data.acceleration[..., t_start, :, :]  # *c, N, 2
        v_cur = data.velocity[..., t_start, :, :]  # *c, N, 2
        p_cur = data.position[..., t_start, :, :]  # *c, N, 2
        dest_cur = data.destination[..., t_start, :, :]  # *c, N, 2
        dest_idx_cur = data.dest_idx[..., t_start, :]  # *c, N
        dest_num = data.dest_num

        new_peds_flag = (data.mask_p - data.mask_p_pred).long()  # c, t, n

        loss = torch.tensor(0., requires_grad=True, device=p_cur.device)
        p_res = torch.zeros(data.position.shape, device=p_cur.device)
        collisions = torch.zeros(mask_p_.shape, device=p_cur.device)  # c, t, n
        hard_collisions = torch.zeros(mask_p_.shape, device=p_cur.device)  # c, t, n
        label_collisions = torch.zeros(mask_p_.shape, device=p_cur.device)  # c, t, n
        label_hard_collisions = torch.zeros(mask_p_.shape, device=p_cur.device)  # c, t, n
        a_res = torch.zeros(data.acceleration.shape, device=p_cur.device)
        pred_collisions = torch.zeros(data.ped_features[..., 0].shape, device=p_cur.device)
        true_collision = torch.zeros(data.ped_features[..., 0].shape, device=p_cur.device)
        reg_loss = torch.tensor(0., device=p_cur.device)
        for t in range(t_start, data.num_frames):

            predictions = self.model(*state_features)
            p_msg = predictions[1]  # c,n,k,6
            # if len(predictions) > 2:
            #     o_msg = predictions[2]
            mask = mask_p_[:, t, :]  # c,n

            if torch.sum(mask) > 0:
                collision = self.collision_detection(p_cur.clone().detach(), args.collision_threshold)  # c, n, n
                collision = torch.sum(collision, dim=-1)  # c, n
                collisions[:, t, :] = collision

                hard_collision = self.collision_detection(p_cur.clone().detach(),
                                                          args.collision_threshold / 2)  # c, n, n
                hard_collision = torch.sum(hard_collision, dim=-1)  # c, n
                hard_collisions[:, t, :] = hard_collision

                label_collision = self.collision_detection(data.labels[:, t, :, :2],
                                                           args.collision_threshold)  # c, n, n
                label_collision = torch.sum(label_collision, dim=-1)  # c, n
                label_collisions[:, t, :] = label_collision
                label_hard_collision = self.collision_detection(data.labels[:, t, :, :2],
                                                                args.collision_threshold / 2)  # c, n, n
                label_hard_collision = torch.sum(label_hard_collision, dim=-1)  # c, n
                label_hard_collisions[:, t, :] = label_hard_collision
                # collision = torch.clamp(collision - label_collision, 0, 1)
                # hard_collision = torch.clamp(hard_collision - label_hard_collision, 0, 1)

                p_res[:, t, ...] = p_cur
                a_res[:, t, ...] = a_cur

                if args.collision_pred_weight > 0 and args.model == 'pinnsf_bm':
                    pred_collisions[:, t, ...] = predictions[-1]
                    true_collision[:, t, ...] = DATA.Pedestrians.calculate_collision_label(state_features[0])

                if args.reg_weight > 0:
                    reg_loss += self.l1_reg_loss(p_msg, args.reg_weight, 'sum')
                    loss = loss + reg_loss
                # if len(predictions) > 2:
                #     loss = loss + self.l1_reg_loss(o_msg, args.reg_weight, 'sum')

            a_next = predictions[0]
            v_next = v_cur + a_cur * data.time_unit
            p_next = p_cur + v_cur * data.time_unit  # *c, n, 2

            assert ~a_next.isnan().any(), print('find nan in epoch :', self.epoch, self.batch_idx)

            # update destination, yet do not delete people when they arrive
            dis_to_dest = torch.norm(p_cur - dest_cur, p=2, dim=-1)
            dest_idx_cur[dis_to_dest < 0.5] = dest_idx_cur[dis_to_dest < 0.5] + 1

            dest_idx_cur[dest_idx_cur > dest_num - 1] = dest_idx_cur[dest_idx_cur > dest_num - 1] - 1
            dest_idx_cur_ = dest_idx_cur.unsqueeze(-2).unsqueeze(-1)  # *c, 1, n, 1
            dest_idx_cur_ = dest_idx_cur_.repeat(*([1] * (dest_idx_cur_.dim() - 1) + [2]))
            dest_cur = torch.gather(waypoints, -3, dest_idx_cur_).squeeze()  # *c, n, 2

            # update everyone's state
            p_cur = p_next  # *c, n, 2
            v_cur = v_next
            a_cur = a_next

            # add newly joined pedestrians
            if t < data.num_frames - 1:
                new_idx = new_peds_flag[..., t + 1, :]  # c, n
                if torch.sum(new_idx) > 0:
                    p_cur[new_idx == 1] = data.position[..., t + 1, :, :][new_idx == 1, :]
                    v_cur[new_idx == 1] = data.velocity[..., t + 1, :, :][new_idx == 1, :]
                    a_cur[new_idx == 1] = data.acceleration[..., t + 1, :, :][new_idx == 1, :]
                    dest_cur[new_idx == 1] = data.destination[..., t + 1, :, :][new_idx == 1, :]
                    dest_idx_cur[new_idx == 1] = data.dest_idx[..., t + 1, :][new_idx == 1]

            # calculate features
            ped_features, obs_features, dest_features = self.get_relative_features(
                p_cur.unsqueeze(-3), v_cur.unsqueeze(-3), a_cur.unsqueeze(-3),
                dest_cur.unsqueeze(-3), obstacles, args.topk_ped, args.sight_angle_ped,
                args.dist_threshold_ped, args.topk_obs,
                args.sight_angle_obs, args.dist_threshold_obs)  # c,1,n,k,2

            self_features = torch.cat((dest_features.squeeze(), v_cur, a_cur, desired_speed), dim=-1)
            state_features = [ped_features.squeeze(), obs_features.squeeze(), self_features]

        # 后处理：删除label本来就有碰撞的collision collisions: c,t,n,要删除的应该是有collision的所有t。。
        if args.new_collision_loss_flag:
            label_collisions = torch.sum(label_collisions, dim=-2, keepdim=True)
            label_hard_collisions = torch.sum(label_hard_collisions, dim=-2, keepdim=True)
            label_collisions = label_collisions.repeat(1, collisions.shape[1], 1)
            label_hard_collisions = label_hard_collisions.repeat(1, hard_collisions.shape[1], 1)
            collisions[label_collisions > 0] = 0
            hard_collisions[label_hard_collisions > 0] = 0

        self.collision_count += torch.sum(collisions).item()
        self.hard_collision_count += torch.sum(hard_collisions).item()

        p_res[mask_p_ == 0] = 0.  # delete 'nan'
        data.labels[mask_p_ == 0] = 0.  # delete 'nan'
        mse_loss = self.multiple_rollout_mse_loss(p_res, data.labels[:, :, :, :2], args.time_decay, reduction='sum')

        loss = loss + mse_loss
        collision_loss, hard_collision_loss, collision_pred_loss, collision_pred_acc = [torch.tensor(0.,
                                                                                                     device=loss.device)] * 4
        if args.collision_loss_weight > 0:
            if args.collision_loss_version == 'v0':
                collision_loss = self.multiple_rollout_collision_loss(
                    p_res, data.labels[:, :, :, :2], args.time_decay, args.collision_focus_weight, collisions,
                    reduction='sum')
                hard_collision_loss = self.multiple_rollout_collision_loss(
                    p_res, data.labels[:, :, :, :2], args.time_decay, args.collision_focus_weight, hard_collisions,
                    reduction='sum')
            elif args.collision_loss_version == 'v2':
                collision_loss = self.multiple_rollout_collision_loss(
                    p_res, data.labels[:, :, :, :2], args.time_decay, args.collision_focus_weight, collisions,
                    reduction='sum', abnormal_mask=data.abnormal_mask)
                hard_collision_loss = self.multiple_rollout_collision_loss(
                    p_res, data.labels[:, :, :, :2], args.time_decay, args.collision_focus_weight, hard_collisions,
                    reduction='sum', abnormal_mask=data.abnormal_mask)

            collision_loss = collision_loss * args.collision_loss_weight
            hard_collision_loss = hard_collision_loss * args.collision_loss_weight * args.hard_collision_penalty

            loss = loss + collision_loss + hard_collision_loss

        if args.teacher_weight > 0:
            a_mse_loss = self.multiple_rollout_mse_loss(
                a_res, data.labels[..., 4:6], args.time_decay, reduction='sum', reverse=True)
            loss = loss + a_mse_loss * args.teacher_weight

        if args.collision_pred_weight > 0:
            collision_pred_loss = F.binary_cross_entropy(pred_collisions, true_collision,
                                                         reduction='sum') * args.collision_pred_weight
            collision_pred_acc = torch.sum(torch.round(pred_collisions) == true_collision) / true_collision.numel()
            loss = loss + collision_pred_loss

        return loss, mse_loss, collision_loss, hard_collision_loss, collision_pred_loss, collision_pred_acc, reg_loss

    def run_single_step(self, ped_state, destination, obstacles, steps):
        pass

    def run(self, ped_state, destination, obstacles, steps):
        pass

    def prepare_symbolic_regression_data(self, data: DATA.PointwisePedData):
        """
        先做p_msg
        Args:
            data:
        """
        args = self.args
        self.load_model(args, set_model=True, finetune_flag=args.finetune_flag)

        with torch.no_grad():
            polar_base = data.self_features[..., -5:-3]  # c, t, n, 2
            polar_base = DATA.Pedestrians.get_heading_direction(polar_base)
            polar_base = polar_base.unsqueeze(-2).repeat(*([1] * (polar_base.dim() - 1)), data.ped_features.shape[-2],
                                                         1)
            polar_base = polar_base.reshape(-1, polar_base.shape[-1])  # x, 2
            polar_base1 = torch.zeros_like(polar_base)
            polar_base1[:, 0] = 1

            features = data.ped_features.reshape(-1, data.ped_features.shape[-1])  # c,t,n,k,6
            coll_pred = DATA.Pedestrians.calculate_collision_label(features).reshape(-1, 1)
            # r = torch.norm(features[:, :2], p=2, dim=-1, keepdim=True)  # n,2
            # v = torch.norm(features[:, 2:4], p=2, dim=-1, keepdim=True)
            # dx_dy = features[:, :2] / r
            # dvx_dvy = features[:, 2:4] / v

            r_thetar = DATA.TimeIndexedPedDataPolarCoor.cart_to_polar(features[:, :2], polar_base)
            v_thetav = DATA.TimeIndexedPedDataPolarCoor.cart_to_polar(features[:, 2:4], polar_base)
            v_thetav[v_thetav > 4.5] = 0

            theta_r2 = DATA.TimeIndexedPedDataPolarCoor.cart_to_polar(features[:, :2], polar_base1)
            theta_r2 = theta_r2[..., 1:2] + 3.1415926
            theta_r2[theta_r2 > 3.1415926] -= 2 * 3.1415926
            features = torch.cat((r_thetar, v_thetav, theta_r2, coll_pred), dim=-1)

            self.model.eval()
            state_features = [data.ped_features, data.obs_features, data.self_features]
            predictions = self.model(*state_features)
            p_msg = predictions[1]  # n,k,dim
            if len(predictions) > 2:
                o_msg = predictions[2]

            p_msg = p_msg.reshape(-1, p_msg.shape[-1])

            filter_idx = torch.sum(torch.abs(features), dim=-1)
            features = features[filter_idx > 0]
            p_msg = p_msg[filter_idx > 0]

            if p_msg.shape[-1] > 2:
                # 非bottleneck模式下，加了强正则，然后对方差最大的2个维度回归
                sorted_pmsg = torch.sort(torch.std(p_msg, dim=0), dim=-1, descending=True)
                print(sorted_pmsg.values[:10])
                labels = p_msg[:, sorted_pmsg.indices]
            else:
                # bottleneck模式
                labels = DATA.TimeIndexedPedDataPolarCoor.cart_to_polar(p_msg, polar_base1)

        return [features.cpu().numpy(), labels.cpu().numpy()]

    def prepare_symbolic_regression_data_polar(self, data: DATA.PointwisePedData):
        """
        先做p_msg
        Args:
            data:
        """
        args = self.args
        self.load_model(args, set_model=True, finetune_flag=args.finetune_flag)

        with torch.no_grad():
            features = data.ped_features.reshape(-1, data.ped_features.shape[-1])
            features = features[:, :4]  # r, theta_r, v, theta_v

            self.model.eval()
            state_features = [data.ped_features, data.obs_features, data.self_features]
            predictions = self.model(*state_features)
            p_msg = predictions[1]  # n,k,dim
            if len(predictions) > 2:
                o_msg = predictions[2]

            p_msg = p_msg.reshape(-1, p_msg.shape[-1])

            p_msg = p_msg[features[:, 0] > 1e-8]
            features = features[features[:, 0] > 1e-8]

        return [features.cpu().numpy(), p_msg.cpu().numpy()]

    def save_cpu_model(self):
        args = self.args
        self.load_model(args, set_model=True, finetune_flag=args.finetune_flag)
        self.save_model(args, finetune_flag=args.finetune_flag, cpu_version=True)


class ResMLPSimulator(BaseSimulator):
    """docstring for ResMLPSimulator"""

    def __init__(self, args):
        super(ResMLPSimulator, self).__init__()
        self.arg = args
        self.model = None
        self.xx

    def get_single_rollout(self):
        pass

    def get_multiple_rollouts(self, **kwargs):
        pass
