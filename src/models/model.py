# -*- coding: utf-8 -*-
"""
@author: zgz

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')
import data.data as DATA


def activation_layer(act_name, negative_slope=0.1):
    """Construct activation layers
    Args:
        act_name: str or nn.Module, name of activation function
        hidden_size: int, used for Dice activation
        dice_dim: int, used for Dice activation
    Return:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU()
        elif act_name.lower() == 'leaky_relu':
            act_layer = nn.LeakyReLU(negative_slope)
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer


class MLP(nn.Module):
    """docstring for MLP"""

    def __init__(
            self,
            input_size,
            layer_sizes,
            activation=nn.ReLU(),
            dropout=0,
            output_act=nn.Identity()):
        super(MLP, self).__init__()
        self.dropout = dropout
        layer_sizes = [input_size] + layer_sizes
        self.mlp = self.build_mlp(layer_sizes, output_act, activation)

    def build_mlp(self, layer_sizes, output_act=nn.Identity(), activation=nn.ReLU()):
        layers = []
        for i in range(len(layer_sizes) - 1):
            act = activation if i < len(layer_sizes) - 2 else output_act
            layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), act]
            if self.dropout:
                layers += nn.Dropout(self.dropout)
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ResBlock(nn.Module):
    def __init__(self, in_dim, hidden_units, activation, dropout=0, use_bn=False):
        super(ResBlock, self).__init__()
        self.use_bn = use_bn
        self.activation = activation

        self.lin = MLP(in_dim, hidden_units, activation, dropout, activation)
        if self.use_bn:
            raise NotImplementedError('bn in resblock has not been implemented!')

    def forward(self, x):
        return self.lin(x) + x


class ResDNN(nn.Module):
    """The Multi Layer Percetron with Residuals
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most
          common situation would be a 2D input with shape
          ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``.
          For instance, for a 2D input with shape ``(batch_size, input_dim)``,
          the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **hidden_units**:list of list, which contains the layer number and
          units in each layer.
            - e.g., [[5], [5,5], [5,5]]
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied
          to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation.
    """

    def __init__(self, input_dim, hidden_units, activation=nn.ReLU(), dropout=0, use_bn=False):
        super(ResDNN, self).__init__()
        if input_dim != hidden_units[0][0]:
            raise ValueError('In ResBlock, the feature size must be equal to the hidden \
                size! input_dim:{}, hidden_size: {}'.format(input_dim, hidden_units[0]))
        self.dropout = nn.Dropout(dropout)
        self.use_bn = use_bn
        self.hidden_units = hidden_units
        self.hidden_units[0] = [input_dim] + self.hidden_units[0]
        self.resnet = nn.ModuleList(
            [ResBlock(h[0], h[1:], activation, use_bn) for h in hidden_units])

    def forward(self, x):
        for i in range(len(self.hidden_units)):
            out = self.resnet[i](x)
            out = self.dropout(out)
        return out


class BaseSimModel(nn.Module):
    """
    Inputs:
        ped_features: N * k1 * dim
        obs_features: N * k2 * dim
        self_features: N * dim
    Processor inputs:
        ped_features: N * (K1+k2) * emb_size1
        dest_features: N * emb_size2
    """

    def __init__(self, args):
        super(BaseSimModel, self).__init__()

        self.ped_feature_dim = args.ped_feature_dim
        self.obs_feature_dim = args.obs_feature_dim
        self.self_feature_dim = args.self_feature_dim
        ped_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        obs_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        self_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        ped_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        self_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        dec_hidden_units = [args.decoder_hidden_size for _ in range(args.decoder_hidden_layers)]
        dropout = args.dropout
        activation = activation_layer(args.activation)

        # encoder
        self.ped_encoder = MLP(self.ped_feature_dim, ped_enc_hidden_units)
        if self.obs_feature_dim > 0:
            self.obs_encoder = MLP(self.obs_feature_dim, obs_enc_hidden_units)
        self.self_encoder = MLP(self.self_feature_dim, self_enc_hidden_units)

        # processor
        self.ped_processor = ResDNN(ped_enc_hidden_units[-1],
                                    ped_pro_hidden_units)
        self.self_processor = ResDNN(self_enc_hidden_units[-1],
                                     self_pro_hidden_units, activation, dropout)

        # decoder
        self.ped_decoder = MLP(ped_pro_hidden_units[-1][-1] + self_pro_hidden_units[-1][-1],
                               dec_hidden_units)
        self.predictor = MLP(dec_hidden_units[-1], [2])

    def forward(self, ped_features, obs_features, self_features):
        ped_embeddings = self.ped_encoder(ped_features)
        if self.obs_feature_dim > 0:
            obs_embeddings = self.obs_encoder(obs_features)
            ped_embeddings = torch.cat((ped_embeddings, obs_embeddings), dim=1)
        self_embeddings = self.self_encoder(self_features)

        ped_embeddings = self.ped_processor(ped_embeddings)
        self_embeddings = self.self_processor(self_embeddings)

        ped_embeddings = torch.sum(ped_embeddings, 1)
        ped_embeddings = torch.cat((ped_embeddings, self_embeddings), dim=1)
        ped_embeddings = self.ped_decoder(ped_embeddings)

        predictions = self.predictor(ped_embeddings)

        return predictions


class BaseSimModel1(nn.Module):
    """
    在base的基础上，把destination改成自己编码；
    Inputs:
        ped_features: N * k1 * dim
        obs_features: N * k2 * dim
        self_features: N * dim
    Processor inputs:
        ped_features: N * (K1+k2) * emb_size1
        dest_features: N * emb_size2
    """

    def __init__(self, args):
        super(BaseSimModel1, self).__init__()

        self.ped_feature_dim = args.ped_feature_dim
        self.obs_feature_dim = args.obs_feature_dim
        self.self_feature_dim = args.ped_feature_dim
        ped_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        obs_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        self_enc_hidden_units = [int(args.encoder_hidden_size / 2) for _ in range(args.encoder_hidden_layers)]
        ped_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        self_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        dec_hidden_units = [args.decoder_hidden_size for _ in range(args.decoder_hidden_layers)]
        dropout = args.dropout
        activation = activation_layer(args.activation)

        # encoder
        self.ped_encoder = MLP(self.ped_feature_dim, ped_enc_hidden_units)
        if self.obs_feature_dim > 0:
            self.obs_encoder = MLP(self.obs_feature_dim, obs_enc_hidden_units)
        self.self_encoder1 = MLP(2, self_enc_hidden_units)
        self.self_encoder2 = MLP(self.self_feature_dim - 2, self_enc_hidden_units)

        # processor
        self.ped_processor = ResDNN(ped_enc_hidden_units[-1],
                                    ped_pro_hidden_units, activation, dropout)
        self.self_processor = ResDNN(self_enc_hidden_units[-1] * 2,
                                     self_pro_hidden_units, activation, dropout)

        # decoder
        self.ped_decoder = MLP(ped_pro_hidden_units[-1][-1] + self_pro_hidden_units[-1][-1],
                               dec_hidden_units)
        self.predictor = MLP(dec_hidden_units[-1], [2])

    def forward(self, ped_features, obs_features, self_features):
        ped_embeddings = self.ped_encoder(ped_features)
        if self.obs_feature_dim > 0:
            obs_embeddings = self.obs_encoder(obs_features)
            ped_embeddings = torch.cat((ped_embeddings, obs_embeddings), dim=1)
        self_embeddings1 = self.self_encoder1(self_features[:, :2])
        self_embeddings2 = self.self_encoder2(self_features[:, 2:])
        self_embeddings = torch.cat((self_embeddings1, self_embeddings2), dim=-1)

        ped_embeddings = self.ped_processor(ped_embeddings)
        self_embeddings = self.self_processor(self_embeddings)

        ped_embeddings = torch.sum(ped_embeddings, 1)
        ped_embeddings = torch.cat((ped_embeddings, self_embeddings), dim=1)
        ped_embeddings = self.ped_decoder(ped_embeddings)

        predictions = self.predictor(ped_embeddings)

        return predictions


class BaseSimModel2(nn.Module):
    '''
    在base1的基础上，把embedding完的self_features维度扩展后和 ped_features拼起来一起过processer
    相当于每个人产生的力由相对坐标和绝对坐标共同决定；
    Inputs:
        ped_features: N * k1 * dim
        obs_features: N * k2 * dim
        self_features: N * dim
    Processor inputs:
        ped_features: N * (K1+k2) * emb_size1
        dest_features: N * emb_size2
    '''

    def __init__(self, args):
        super(BaseSimModel2, self).__init__()

        self.ped_feature_dim = args.ped_feature_dim
        self.obs_feature_dim = args.obs_feature_dim
        self.self_feature_dim = args.ped_feature_dim
        ped_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        obs_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        self_enc_hidden_units = [int(args.encoder_hidden_size / 2) for _ in range(args.encoder_hidden_layers)]
        ped_pro_hidden_units = [[args.processor_hidden_size * 2] for _ in range(args.processor_hidden_layers)]
        dec_hidden_units = [args.decoder_hidden_size for _ in range(args.decoder_hidden_layers)]
        dropout = args.dropout
        activation = activation_layer(args.activation)

        # encoder
        self.ped_encoder = MLP(self.ped_feature_dim, ped_enc_hidden_units)
        if self.obs_feature_dim > 0:
            self.obs_encoder = MLP(self.obs_feature_dim, obs_enc_hidden_units)
        self.self_encoder1 = MLP(2, self_enc_hidden_units)
        self.self_encoder2 = MLP(self.self_feature_dim - 2, self_enc_hidden_units)

        # processor
        self.ped_processor = ResDNN(ped_enc_hidden_units[-1] + self_enc_hidden_units[-1] * 2,
                                    ped_pro_hidden_units, activation, dropout)

        # decoder
        self.ped_decoder = MLP(ped_pro_hidden_units[-1][-1],
                               dec_hidden_units)
        self.predictor = MLP(dec_hidden_units[-1], [2])

    def forward(self, ped_features, obs_features, self_features):
        ped_embeddings = self.ped_encoder(ped_features)
        if self.obs_feature_dim > 0:
            obs_embeddings = self.obs_encoder(obs_features)
            ped_embeddings = torch.cat((ped_embeddings, obs_embeddings), dim=1)
        self_embeddings1 = self.self_encoder1(self_features[:, :2])
        self_embeddings2 = self.self_encoder2(self_features[:, 2:])
        self_embeddings = torch.cat((self_embeddings1, self_embeddings2), dim=-1)

        self_embeddings = self_embeddings.unsqueeze(1).repeat(1, ped_embeddings.shape[1], 1)
        ped_embeddings = torch.cat((ped_embeddings, self_embeddings), dim=-1)

        ped_embeddings = self.ped_processor(ped_embeddings)

        ped_embeddings = torch.sum(ped_embeddings, 1)
        ped_embeddings = self.ped_decoder(ped_embeddings)

        predictions = self.predictor(ped_embeddings)

        return predictions


class BaseSimModel3(nn.Module):
    '''
    在base1的基础上，把destination改为：单位向量
    Inputs:
        ped_features: N * k1 * dim
        obs_features: N * k2 * dim
        self_features: N * dim
    Processor inputs:
        ped_features: N * (K1+k2) * emb_size1
        dest_features: N * emb_size2
    '''

    def __init__(self, args):
        super(BaseSimModel3, self).__init__()

        self.ped_feature_dim = args.ped_feature_dim
        self.obs_feature_dim = args.obs_feature_dim
        self.self_feature_dim = args.ped_feature_dim
        ped_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        obs_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        self_enc_hidden_units = [int(args.encoder_hidden_size / 2) for _ in range(args.encoder_hidden_layers)]
        ped_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        self_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        dec_hidden_units = [args.decoder_hidden_size for _ in range(args.decoder_hidden_layers)]
        dropout = args.dropout
        activation = activation_layer(args.activation)

        # encoder
        self.ped_encoder = MLP(self.ped_feature_dim, ped_enc_hidden_units)
        if self.obs_feature_dim > 0:
            self.obs_encoder = MLP(self.obs_feature_dim, obs_enc_hidden_units)
        self.self_encoder1 = MLP(2, self_enc_hidden_units)
        self.self_encoder2 = MLP(self.self_feature_dim - 2, self_enc_hidden_units)

        # processor
        self.ped_processor = ResDNN(ped_enc_hidden_units[-1],
                                    ped_pro_hidden_units, activation, dropout)
        self.self_processor = ResDNN(self_enc_hidden_units[-1] * 2,
                                     self_pro_hidden_units, activation, dropout)

        # decoder
        self.ped_decoder = MLP(ped_pro_hidden_units[-1][-1] + self_pro_hidden_units[-1][-1],
                               dec_hidden_units)
        self.predictor = MLP(dec_hidden_units[-1], [2])

    def forward(self, ped_features, obs_features, self_features):
        ped_embeddings = self.ped_encoder(ped_features)
        if self.obs_feature_dim > 0:
            obs_embeddings = self.obs_encoder(obs_features)
            ped_embeddings = torch.cat((ped_embeddings, obs_embeddings), dim=1)
        destination = self_features[:, :2] / torch.norm(self_features[:, :2], p=2, dim=1, keepdim=True)

        self_embeddings1 = self.self_encoder1(destination)
        self_embeddings2 = self.self_encoder2(self_features[:, 2:])
        self_embeddings = torch.cat((self_embeddings1, self_embeddings2), dim=-1)

        ped_embeddings = self.ped_processor(ped_embeddings)
        self_embeddings = self.self_processor(self_embeddings)

        ped_embeddings = torch.sum(ped_embeddings, 1)
        ped_embeddings = torch.cat((ped_embeddings, self_embeddings), dim=1)
        ped_embeddings = self.ped_decoder(ped_embeddings)

        predictions = self.predictor(ped_embeddings)

        return predictions


class BaseSimModel4(nn.Module):
    '''
        在base的基础上，把destination改为：单位向量
    Inputs:
        ped_features: N * k1 * dim
        obs_features: N * k2 * dim
        self_features: N * dim
    Processor inputs:
        ped_features: N * (K1+k2) * emb_size1
        dest_features: N * emb_size2
    '''

    def __init__(self, args):
        super(BaseSimModel4, self).__init__()

        self.args = args
        self.ped_feature_dim = args.ped_feature_dim
        self.obs_feature_dim = args.obs_feature_dim
        self.self_feature_dim = args.ped_feature_dim
        ped_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        obs_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        self_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        ped_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        self_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        dec_hidden_units = [args.decoder_hidden_size for _ in range(args.decoder_hidden_layers)]
        dropout = args.dropout
        activation = activation_layer(args.activation)

        # encoder
        self.ped_encoder = MLP(self.ped_feature_dim, ped_enc_hidden_units)
        if self.obs_feature_dim > 0:
            self.obs_encoder = MLP(self.obs_feature_dim, obs_enc_hidden_units)
        self.self_encoder = MLP(self.self_feature_dim, self_enc_hidden_units)

        # processor
        self.ped_processor = ResDNN(ped_enc_hidden_units[-1],
                                    ped_pro_hidden_units, activation, dropout)
        self.self_processor = ResDNN(self_enc_hidden_units[-1],
                                     self_pro_hidden_units, activation, dropout)

        # decoder
        self.ped_decoder = MLP(ped_pro_hidden_units[-1][-1] + self_pro_hidden_units[-1][-1],
                               dec_hidden_units)
        self.predictor = MLP(dec_hidden_units[-1], [2])

    def forward(self, ped_features, obs_features, self_features):
        ped_embeddings = self.ped_encoder(ped_features)
        if self.obs_feature_dim > 0:
            obs_embeddings = self.obs_encoder(obs_features)
            ped_embeddings = torch.cat((ped_embeddings, obs_embeddings), dim=1)
        self_features[:, :2] = self_features[:, :2] / torch.norm(self_features[:, :2], p=2, dim=1, keepdim=True)
        self_embeddings = self.self_encoder(self_features)

        ped_embeddings = self.ped_processor(ped_embeddings)
        self_embeddings = self.self_processor(self_embeddings)

        ped_embeddings = torch.sum(ped_embeddings, 1)
        ped_embeddings = torch.cat((ped_embeddings, self_embeddings), dim=1)
        ped_embeddings = self.ped_decoder(ped_embeddings)

        predictions = self.predictor(ped_embeddings)

        return predictions


class BaseSimModel5(nn.Module):
    '''
    在base的基础上，把embedding完的self_features维度扩展后和 ped_features拼起来一起过processer
    相当于每个人产生的力由相对坐标和绝对坐标共同决定；
    同时，destination改为单位向量
    Inputs:
        ped_features: N * k1 * dim
        obs_features: N * k2 * dim
        self_features: N * dim
    Processor inputs:
        ped_features: N * (K1+k2) * emb_size1
        dest_features: N * emb_size2
    '''

    def __init__(self, args):
        super(BaseSimModel5, self).__init__()

        self.ped_feature_dim = args.ped_feature_dim
        self.obs_feature_dim = args.obs_feature_dim
        self.self_feature_dim = args.ped_feature_dim
        ped_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        obs_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        self_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        ped_pro_hidden_units = [[args.processor_hidden_size * 2] for _ in range(args.processor_hidden_layers)]
        dec_hidden_units = [args.decoder_hidden_size for _ in range(args.decoder_hidden_layers)]
        dropout = args.dropout
        activation = activation_layer(args.activation)

        # encoder
        self.ped_encoder = MLP(self.ped_feature_dim, ped_enc_hidden_units)
        if self.obs_feature_dim > 0:
            self.obs_encoder = MLP(self.obs_feature_dim, obs_enc_hidden_units)
        self.self_encoder = MLP(self.self_feature_dim, self_enc_hidden_units)

        # processor
        self.ped_processor = ResDNN(ped_enc_hidden_units[-1] + self_enc_hidden_units[-1],
                                    ped_pro_hidden_units, activation, dropout)

        # decoder
        self.ped_decoder = MLP(ped_pro_hidden_units[-1][-1],
                               dec_hidden_units)
        self.predictor = MLP(dec_hidden_units[-1], [2])

    def forward(self, ped_features, obs_features, self_features):
        ped_embeddings = self.ped_encoder(ped_features)
        if self.obs_feature_dim > 0:
            obs_embeddings = self.obs_encoder(obs_features)
            ped_embeddings = torch.cat((ped_embeddings, obs_embeddings), dim=1)
        self_features[:, :2] = self_features[:, :2] / torch.norm(self_features[:, :2], p=2, dim=1, keepdim=True)
        self_embeddings = self.self_encoder(self_features)

        self_embeddings = self_embeddings.unsqueeze(1).repeat(1, ped_embeddings.shape[1], 1)
        ped_embeddings = torch.cat((ped_embeddings, self_embeddings), dim=-1)

        ped_embeddings = self.ped_processor(ped_embeddings)

        ped_embeddings = torch.sum(ped_embeddings, 1)
        ped_embeddings = self.ped_decoder(ped_embeddings)

        predictions = self.predictor(ped_embeddings)

        return predictions


class BaseSimModel6(nn.Module):
    '''
    在base的基础上加入和其他行人（障碍物）的绝对距离
    Inputs:
        ped_features: N * k1 * dim
        obs_features: N * k2 * dim
        self_features: N * dim
    Processor inputs:
        ped_features: N * (K1+k2) * emb_size1
        dest_features: N * emb_size2
    '''

    def __init__(self, args):
        super(BaseSimModel6, self).__init__()
        self.args = args
        self.ped_feature_dim = args.ped_feature_dim
        self.obs_feature_dim = args.obs_feature_dim
        self.self_feature_dim = args.ped_feature_dim
        ped_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        obs_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        self_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        ped_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        self_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        dec_hidden_units = [args.decoder_hidden_size for _ in range(args.decoder_hidden_layers)]
        dropout = args.dropout
        activation = activation_layer(args.activation)

        # encoder
        self.ped_encoder = MLP(self.ped_feature_dim + 1, ped_enc_hidden_units)
        if self.obs_feature_dim > 0:
            self.obs_encoder = MLP(self.obs_feature_dim, obs_enc_hidden_units)
        self.self_encoder = MLP(self.self_feature_dim, self_enc_hidden_units)

        # processor
        self.ped_processor = ResDNN(ped_enc_hidden_units[-1],
                                    ped_pro_hidden_units, activation, dropout)
        self.self_processor = ResDNN(self_enc_hidden_units[-1],
                                     self_pro_hidden_units, activation, dropout)

        # decoder
        self.ped_decoder = MLP(ped_pro_hidden_units[-1][-1] + self_pro_hidden_units[-1][-1],
                               dec_hidden_units)
        self.predictor = MLP(dec_hidden_units[-1], [2])

    def forward(self, ped_features, obs_features, self_features):
        ped_features_new = torch.zeros(
            (ped_features.shape[0], ped_features.shape[1],
             ped_features.shape[2] + 1)).to(self.args.device)
        ped_features_new[:, :, 1:] = ped_features[:, :, :]
        dist = torch.norm(ped_features[:, :, :2], p=2, dim=-1)
        ped_features_new[:, :, 0] = dist
        ped_embeddings = self.ped_encoder(ped_features_new)
        if self.obs_feature_dim > 0:
            obs_embeddings = self.obs_encoder(obs_features)
            ped_embeddings = torch.cat((ped_embeddings, obs_embeddings), dim=1)
        self_embeddings = self.self_encoder(self_features)

        ped_embeddings = self.ped_processor(ped_embeddings)
        self_embeddings = self.self_processor(self_embeddings)

        ped_embeddings = torch.sum(ped_embeddings, 1)
        ped_embeddings = torch.cat((ped_embeddings, self_embeddings), dim=1)
        ped_embeddings = self.ped_decoder(ped_embeddings)

        predictions = self.predictor(ped_embeddings)

        return predictions


class BaseSimModel7(nn.Module):
    '''
        在base的基础上，把destination改为：单位向量 + 绝对距离
    Inputs:
        ped_features: N * k1 * dim
        obs_features: N * k2 * dim
        self_features: N * dim
    Processor inputs:
        ped_features: N * (K1+k2) * emb_size1
        dest_features: N * emb_size2
    '''

    def __init__(self, args):
        super(BaseSimModel7, self).__init__()

        self.args = args
        self.ped_feature_dim = args.ped_feature_dim
        self.obs_feature_dim = args.obs_feature_dim
        self.self_feature_dim = args.ped_feature_dim + 1
        ped_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        obs_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        self_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        ped_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        self_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        dec_hidden_units = [args.decoder_hidden_size for _ in range(args.decoder_hidden_layers)]
        dropout = args.dropout
        activation = activation_layer(args.activation)

        # encoder
        self.ped_encoder = MLP(self.ped_feature_dim, ped_enc_hidden_units)
        if self.obs_feature_dim > 0:
            self.obs_encoder = MLP(self.obs_feature_dim, obs_enc_hidden_units)
        self.self_encoder = MLP(self.self_feature_dim, self_enc_hidden_units)

        # processor
        self.ped_processor = ResDNN(ped_enc_hidden_units[-1],
                                    ped_pro_hidden_units, activation, dropout)
        self.self_processor = ResDNN(self_enc_hidden_units[-1],
                                     self_pro_hidden_units, activation, dropout)

        # decoder
        self.ped_decoder = MLP(ped_pro_hidden_units[-1][-1] + self_pro_hidden_units[-1][-1],
                               dec_hidden_units)
        self.predictor = MLP(dec_hidden_units[-1], [2])

    def forward(self, ped_features, obs_features, self_features):
        ped_embeddings = self.ped_encoder(ped_features)
        if self.obs_feature_dim > 0:
            obs_embeddings = self.obs_encoder(obs_features)
            ped_embeddings = torch.cat((ped_embeddings, obs_embeddings), dim=1)
        self_features_new = torch.zeros(self_features.shape[0], self_features.shape[1] + 1).to(self.args.device)
        self_features_new[:, 3:] = self_features[:, 2:]
        self_features_new[:, :2] = self_features[:, :2] / torch.norm(self_features[:, :2], p=2, dim=1, keepdim=True)
        self_features_new[:, 2] = torch.norm(self_features[:, :2], p=2, dim=1)
        self_embeddings = self.self_encoder(self_features_new)

        ped_embeddings = self.ped_processor(ped_embeddings)
        self_embeddings = self.self_processor(self_embeddings)

        ped_embeddings = torch.sum(ped_embeddings, 1)
        ped_embeddings = torch.cat((ped_embeddings, self_embeddings), dim=1)
        ped_embeddings = self.ped_decoder(ped_embeddings)

        predictions = self.predictor(ped_embeddings)

        return predictions


class BaseNDSimModel(nn.Module):
    """
    Inputs:
        ped_features: N * k1 * dim
        obs_features: N * k2 * dim
        self_features: N * dim
    Processor inputs:
        ped_features: N * (K1+k2) * emb_size1
        dest_features: N * emb_size2
    todo: corrector加的位置有问题；
    """

    def __init__(self, args):
        super(BaseNDSimModel, self).__init__()

        self.args = args
        self.ped_feature_dim = args.ped_feature_dim
        self.obs_feature_dim = args.obs_feature_dim
        self.self_feature_dim = args.self_feature_dim
        ped_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        obs_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        self_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        ped_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        self_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        dec_hidden_units = [args.decoder_hidden_size for _ in range(args.decoder_hidden_layers)]
        dropout = args.dropout
        activation = activation_layer(args.activation)

        # encoder
        self.ped_encoder = MLP(self.ped_feature_dim, ped_enc_hidden_units)
        if self.obs_feature_dim > 0:
            self.obs_encoder = MLP(self.obs_feature_dim, obs_enc_hidden_units)
        self.self_encoder = MLP(self.self_feature_dim, self_enc_hidden_units)

        # processor
        self.ped_processor = ResDNN(ped_enc_hidden_units[-1],
                                    ped_pro_hidden_units, activation, dropout)
        self.self_processor = ResDNN(self_enc_hidden_units[-1],
                                     self_pro_hidden_units, activation, dropout)

        # decoder
        self.ped_decoder = MLP(ped_pro_hidden_units[-1][-1] + self_pro_hidden_units[-1][-1],
                               dec_hidden_units)

        # correction net
        cor_hidden_units = [[args.decoder_hidden_size] * 2 for _ in range(args.correction_hidden_layers)]
        self.corrector = ResDNN(dec_hidden_units[-1], cor_hidden_units, activation, dropout)

        self.predictor = MLP(dec_hidden_units[-1], [2])

    def forward(self, ped_features, obs_features, self_features):
        ped_embeddings = self.ped_encoder(ped_features)
        if self.obs_feature_dim > 0:
            obs_embeddings = self.obs_encoder(obs_features)
            ped_embeddings = torch.cat((ped_embeddings, obs_embeddings), dim=1)
        self_embeddings = self.self_encoder(self_features)

        ped_embeddings = self.ped_processor(ped_embeddings)
        self_embeddings = self.self_processor(self_embeddings)

        ped_embeddings = torch.sum(ped_embeddings, 1)
        ped_embeddings = torch.cat((ped_embeddings, self_embeddings), dim=1)
        ped_embeddings = self.ped_decoder(ped_embeddings)

        ped_embeddings = self.corrector(ped_embeddings)

        predictions = self.predictor(ped_embeddings)

        return predictions


class PINNSF(nn.Module):
    """
    Inputs:
        ped_features: (C *) N * k1 * dim
        obs_features: (C *) N * k2 * dim
        self_features: (C *) N * dim
    Returns:
        predictions (acceleration): (C *) N * 2
    """

    def __init__(self, args):
        super(PINNSF, self).__init__()

        self.tau = 2
        self.ped_feature_dim = args.ped_feature_dim
        self.obs_feature_dim = args.obs_feature_dim
        self.self_feature_dim = args.self_feature_dim
        ped_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        obs_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        ped_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        obs_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        dec_hidden_units = [args.decoder_hidden_size for _ in range(args.decoder_hidden_layers)]
        dropout = args.dropout
        activation = activation_layer(args.activation)

        # encoder
        self.ped_encoder = MLP(self.ped_feature_dim, ped_enc_hidden_units)
        # if self.obs_feature_dim > 0:
        self.obs_encoder = MLP(6, obs_enc_hidden_units)

        # processor
        self.ped_processor = ResDNN(ped_enc_hidden_units[-1], ped_pro_hidden_units, activation, dropout)
        self.obs_processor = ResDNN(obs_enc_hidden_units[-1], obs_pro_hidden_units, activation, dropout)

        # decoder: node function
        self.ped_decoder = MLP(ped_pro_hidden_units[-1][-1], dec_hidden_units)
        self.obs_decoder = MLP(obs_pro_hidden_units[-1][-1], dec_hidden_units)

        self.ped_predictor = MLP(dec_hidden_units[-1], [2])
        self.obs_predictor = MLP(dec_hidden_units[-1], [2])
        # self.obs_predictor = self.ped_predictor

    def forward(self, ped_features, obs_features, self_features):
        assert (self_features.shape[-1] == 7), 'Error: PINN model do not accept inputs of historical velocity'

        ped_embeddings = self.ped_encoder(ped_features)
        ped_embeddings = self.ped_processor(ped_embeddings)
        ped_msgs = ped_embeddings
        ped_embeddings = torch.sum(ped_embeddings, dim=-2)
        ped_embeddings = self.ped_decoder(ped_embeddings)
        pred_acc_ped = self.ped_predictor(ped_embeddings)

        if self.obs_feature_dim > 0:
            obs_embeddings = self.obs_encoder(obs_features)
            obs_embeddings = self.obs_processor(obs_embeddings)
            obs_msgs = obs_embeddings
            obs_embeddings = torch.sum(obs_embeddings, dim=-2)
            obs_embeddings = self.obs_decoder(obs_embeddings)
            pred_acc_ped += self.obs_predictor(obs_embeddings)

        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau

        predictions = pred_acc_ped + pred_acc_dest

        out = [predictions, ped_msgs]
        if self.obs_feature_dim > 0:
            out.append(obs_msgs)
        return out


class PINNSF_polar(nn.Module):
    """
    Inputs:
        ped_features: (C *) N * k1 * dim
        obs_features: (C *) N * k2 * dim
        self_features: (C *) N * dim
    Returns:
        predictions (acceleration): (C *) N * 2
    """

    def __init__(self, args):
        super(PINNSF_polar, self).__init__()

        self.tau = 2
        self.ped_feature_dim = args.ped_feature_dim
        self.obs_feature_dim = args.obs_feature_dim
        self.self_feature_dim = args.self_feature_dim
        ped_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        obs_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        ped_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        obs_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        dec_hidden_units = [args.decoder_hidden_size for _ in range(args.decoder_hidden_layers)]
        dropout = args.dropout
        activation = activation_layer(args.activation)

        # encoder
        self.ped_encoder = MLP(self.ped_feature_dim, ped_enc_hidden_units)
        # if self.obs_feature_dim > 0:
        self.obs_encoder = MLP(6, obs_enc_hidden_units)

        # processor
        self.ped_processor = ResDNN(ped_enc_hidden_units[-1], ped_pro_hidden_units, activation, dropout)
        self.obs_processor = ResDNN(obs_enc_hidden_units[-1], obs_pro_hidden_units, activation, dropout)

        # decoder: node function
        self.ped_decoder = MLP(ped_pro_hidden_units[-1][-1], dec_hidden_units)
        self.obs_decoder = MLP(obs_pro_hidden_units[-1][-1], dec_hidden_units)

        self.ped_predictor = MLP(dec_hidden_units[-1], [2])
        self.obs_predictor = MLP(dec_hidden_units[-1], [2])
        # self.obs_predictor = self.ped_predictor

    def forward(self, ped_features, obs_features, self_features):
        assert (self_features.shape[-1] == 7), 'Error: PINN model do not accept inputs of historical velocity'

        polar_base = self_features[..., -5:-3]  # c, t, n, 2
        polar_base = DATA.Pedestrians.get_heading_direction(polar_base)

        ped_embeddings = self.ped_encoder(ped_features)
        ped_embeddings = self.ped_processor(ped_embeddings)
        ped_msgs = ped_embeddings
        ped_embeddings = torch.sum(ped_embeddings, dim=-2)
        ped_embeddings = self.ped_decoder(ped_embeddings)
        pred_acc_ped = self.ped_predictor(ped_embeddings)
        pred_acc_ped = DATA.TimeIndexedPedDataPolarCoor.polar_to_cart(pred_acc_ped, polar_base)

        if self.obs_feature_dim > 0:
            obs_embeddings = self.obs_encoder(obs_features)
            obs_embeddings = self.obs_processor(obs_embeddings)
            obs_msgs = obs_embeddings
            obs_embeddings = torch.sum(obs_embeddings, dim=-2)
            obs_embeddings = self.obs_decoder(obs_embeddings)
            pred_acc_obs = self.obs_predictor(obs_embeddings)
            pred_acc_ped += DATA.TimeIndexedPedDataPolarCoor.polar_to_cart(pred_acc_obs, polar_base)

        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau

        predictions = pred_acc_ped + pred_acc_dest

        out = [predictions, ped_msgs]
        if self.obs_feature_dim > 0:
            out.append(obs_msgs)
        return out


class PINNSF2(nn.Module):
    """
    Inputs:
        ped_features: (C *) N * k1 * dim
        obs_features: (C *) N * k2 * dim
        self_features: (C *) N * dim
    Returns:
        predictions (acceleration): (C *) N * 2
    """

    def __init__(self, args):
        super(PINNSF2, self).__init__()

        self.tau = 2 + torch.tensor(0., requires_grad=True)
        self.ped_feature_dim = args.ped_feature_dim
        self.obs_feature_dim = args.obs_feature_dim
        self.self_feature_dim = args.self_feature_dim
        ped_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        obs_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        ped_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        obs_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        dec_hidden_units = [args.decoder_hidden_size for _ in range(args.decoder_hidden_layers)]
        dropout = args.dropout
        activation = activation_layer(args.activation)

        # encoder
        self.ped_encoder = MLP(self.ped_feature_dim, ped_enc_hidden_units)
        if self.obs_feature_dim > 0:
            self.obs_encoder = MLP(self.obs_feature_dim, obs_enc_hidden_units)

        # processor
        self.ped_processor = ResDNN(ped_enc_hidden_units[-1], ped_pro_hidden_units, activation, dropout)
        self.obs_processor = ResDNN(obs_enc_hidden_units[-1], obs_pro_hidden_units, activation, dropout)

        # decoder: node function
        self.ped_decoder = MLP(ped_pro_hidden_units[-1][-1], dec_hidden_units)
        self.obs_decoder = MLP(obs_pro_hidden_units[-1][-1], dec_hidden_units)

        self.ped_predictor = MLP(dec_hidden_units[-1], [2])
        self.obs_predictor = MLP(dec_hidden_units[-1], [2])
        # self.obs_predictor = self.ped_predictor

    def forward(self, ped_features, obs_features, self_features):
        assert (self_features.shape[-1] == 7), 'Error: PINN model do not accept inputs of historical velocity'

        ped_embeddings = self.ped_encoder(ped_features)
        ped_embeddings = self.ped_processor(ped_embeddings)
        ped_msgs = ped_embeddings
        ped_embeddings = torch.sum(ped_embeddings, dim=-2)
        ped_embeddings = self.ped_decoder(ped_embeddings)
        pred_acc_ped = self.ped_predictor(ped_embeddings)

        if self.obs_feature_dim > 0:
            obs_embeddings = self.obs_encoder(obs_features)
            obs_embeddings = self.obs_processor(obs_embeddings)
            obs_msgs = obs_embeddings
            obs_embeddings = torch.sum(obs_embeddings, dim=-2)
            obs_embeddings = self.obs_decoder(obs_embeddings)
            pred_acc_ped += self.obs_predictor(obs_embeddings)

        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau

        predictions = pred_acc_ped + pred_acc_dest

        out = [predictions, ped_msgs]
        if self.obs_feature_dim > 0:
            out.append(obs_msgs)
        return out


class attn_pooling(nn.Module):
    """
    """

    def __init__(self, dim):
        super(attn_pooling, self).__init__()
        self.get_weights = MLP(dim, [dim, 1])

    def forward(self, x):
        """
        Args:
            x: c,n,k,dim
        Returns:
            tensor: c, n, dim
        """
        attn = torch.exp(self.get_weights(x))
        attn = torch.softmax(attn, dim=-2)
        dim = list(range(x.dim()))
        dim[-1], dim[-2] = dim[-2], dim[-1]
        x = torch.matmul(x.permute(*dim), attn)
        return x.squeeze()


class PINNSF_residual(nn.Module):
    """
    Inputs:
        ped_features: (C *) N * k1 * dim
        obs_features: (C *) N * k2 * dim
        self_features: (C *) N * dim
    Returns:
        predictions (acceleration): (C *) N * 2
    """

    def __init__(self, args):
        super(PINNSF_residual, self).__init__()

        self.tau = 2
        self.ped_feature_dim = args.ped_feature_dim
        self.obs_feature_dim = args.obs_feature_dim
        self.self_feature_dim = args.self_feature_dim
        ped_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        obs_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        ped_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        obs_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        res_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.res_hidden_layers)]
        dec_hidden_units = [args.decoder_hidden_size for _ in range(args.decoder_hidden_layers)]
        dropout = args.dropout
        activation = activation_layer(args.activation)

        # encoder
        self.ped_encoder = MLP(self.ped_feature_dim, ped_enc_hidden_units)
        if self.obs_feature_dim > 0:
            self.obs_encoder = MLP(self.obs_feature_dim, obs_enc_hidden_units)

        # processor
        self.ped_processor = ResDNN(ped_enc_hidden_units[-1], ped_pro_hidden_units, activation, dropout)
        self.obs_processor = ResDNN(obs_enc_hidden_units[-1], obs_pro_hidden_units, activation, dropout)

        # decoder: node function
        self.ped_decoder = MLP(ped_pro_hidden_units[-1][-1], dec_hidden_units)
        self.obs_decoder = MLP(obs_pro_hidden_units[-1][-1], dec_hidden_units)

        self.ped_predictor = MLP(dec_hidden_units[-1], [2])
        self.obs_predictor = MLP(dec_hidden_units[-1], [2])
        # self.obs_predictor = self.ped_predictor

        self.corrector = nn.ModuleList(
            [ResDNN(ped_enc_hidden_units[-1], res_pro_hidden_units, activation, dropout),
             attn_pooling(res_pro_hidden_units[-1][-1]),
             MLP(res_pro_hidden_units[-1][-1], [int(res_pro_hidden_units[-1][-1] / 2), 2])]
        )
        # self.residual_processor = ResDNN(ped_enc_hidden_units[-1], res_pro_hidden_units, activation, dropout)
        # self.residual_predictor = MLP(res_pro_hidden_units[-1], [2])

    def forward(self, ped_features, obs_features, self_features):
        assert (self_features.shape[-1] == 7), 'Error: PINN model do not accept inputs of historical velocity'

        ped_embeddings = self.ped_encoder(ped_features)
        res_embeddings = ped_embeddings
        ped_embeddings = self.ped_processor(ped_embeddings)
        ped_msgs = ped_embeddings
        ped_embeddings = torch.sum(ped_embeddings, dim=-2)
        ped_embeddings = self.ped_decoder(ped_embeddings)
        pred_acc_ped = self.ped_predictor(ped_embeddings)

        if self.obs_feature_dim > 0:
            obs_embeddings = self.obs_encoder(obs_features)
            obs_embeddings = self.obs_processor(obs_embeddings)
            obs_msgs = obs_embeddings
            obs_embeddings = torch.sum(obs_embeddings, dim=-2)
            obs_embeddings = self.obs_decoder(obs_embeddings)
            pred_acc_ped += self.obs_predictor(obs_embeddings)

        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau

        residual = self.corrector[0](res_embeddings)
        residual = self.corrector[1](residual)
        residual = self.corrector[2](residual)

        predictions = pred_acc_ped + pred_acc_dest + residual

        out = [predictions, ped_msgs]
        if self.obs_feature_dim > 0:
            out.append(obs_msgs)
        return out


class PINNSF_bottleneck(nn.Module):
    """
    Inputs:
        ped_features: (C *) N * k1 * dim
        obs_features: (C *) N * k2 * dim
        self_features: (C *) N * dim
    Returns:
        predictions (acceleration): (C *) N * 2
    """

    def __init__(self, args):
        super(PINNSF_bottleneck, self).__init__()

        self.tau = 2
        self.ped_feature_dim = args.ped_feature_dim
        self.obs_feature_dim = args.obs_feature_dim
        self.self_feature_dim = args.self_feature_dim
        ped_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        obs_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        ped_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        obs_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        dec_hidden_units = [args.decoder_hidden_size for _ in range(args.decoder_hidden_layers)]
        dropout = args.dropout
        activation = activation_layer(args.activation)

        # encoder
        self.ped_encoder = MLP(self.ped_feature_dim, ped_enc_hidden_units)
        # if self.obs_feature_dim > 0:
        self.obs_encoder = MLP(6, obs_enc_hidden_units)

        # processor
        self.ped_processor = ResDNN(ped_enc_hidden_units[-1], ped_pro_hidden_units, activation, dropout)
        self.obs_processor = ResDNN(obs_enc_hidden_units[-1], obs_pro_hidden_units, activation, dropout)

        # decoder: node function
        self.ped_decoder = MLP(ped_pro_hidden_units[-1][-1], dec_hidden_units)
        self.obs_decoder = MLP(obs_pro_hidden_units[-1][-1], dec_hidden_units)

        self.ped_predictor = MLP(dec_hidden_units[-1], [2])
        self.obs_predictor = MLP(dec_hidden_units[-1], [2])
        # self.obs_predictor = self.ped_predictor

    def forward(self, ped_features, obs_features, self_features):
        assert (self_features.shape[-1] == 7), 'Error: PINN model do not accept inputs of historical velocity'

        ped_embeddings = self.ped_encoder(ped_features)
        ped_embeddings = self.ped_processor(ped_embeddings)
        ped_embeddings = self.ped_decoder(ped_embeddings)
        pred_acc_ped = self.ped_predictor(ped_embeddings)
        ped_msgs = pred_acc_ped
        pred_acc_ped = torch.sum(pred_acc_ped, dim=-2)

        if self.obs_feature_dim > 0:
            obs_embeddings = self.obs_encoder(obs_features)
            obs_embeddings = self.obs_processor(obs_embeddings)
            obs_embeddings = self.obs_decoder(obs_embeddings)
            pred_acc_obs = self.obs_predictor(obs_embeddings)
            obs_msgs = pred_acc_obs
            pred_acc_obs = torch.sum(pred_acc_obs, dim=-2)
            pred_acc_ped = pred_acc_ped + pred_acc_obs

        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau

        predictions = pred_acc_ped + pred_acc_dest

        out = [predictions, ped_msgs]
        if self.obs_feature_dim > 0:
            out.append(obs_msgs)
        return out


class PINNSF_bottleneck_multitask(nn.Module):
    """
    Inputs:
        ped_features: (C *) N * k1 * dim
        obs_features: (C *) N * k2 * dim
        self_features: (C *) N * dim
    Returns:
        predictions (acceleration): (C *) N * 2
    """

    def __init__(self, args):
        super(PINNSF_bottleneck_multitask, self).__init__()

        if args.dataset_name in {'ucy'}:
            self.tau = 5/6
        else:
            self.tau = 2
        self.ped_feature_dim = args.ped_feature_dim
        self.obs_feature_dim = args.obs_feature_dim
        self.self_feature_dim = args.self_feature_dim
        ped_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        obs_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        ped_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        obs_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        dec_hidden_units = [args.decoder_hidden_size for _ in range(args.decoder_hidden_layers)]
        dropout = args.dropout
        activation = activation_layer(args.activation)

        # encoder
        self.ped_encoder = MLP(self.ped_feature_dim, ped_enc_hidden_units)
        # if self.obs_feature_dim > 0:
        self.obs_encoder = MLP(6, obs_enc_hidden_units)

        # processor
        self.ped_processor = ResDNN(ped_enc_hidden_units[-1], ped_pro_hidden_units, activation, dropout)
        self.obs_processor = ResDNN(obs_enc_hidden_units[-1], obs_pro_hidden_units, activation, dropout)

        # decoder: node function
        self.ped_decoder = MLP(ped_pro_hidden_units[-1][-1], dec_hidden_units)
        self.obs_decoder = MLP(obs_pro_hidden_units[-1][-1], dec_hidden_units)

        self.ped_predictor = MLP(dec_hidden_units[-1], [2])
        self.obs_predictor = MLP(dec_hidden_units[-1], [2])
        # self.obs_predictor = self.ped_predictor

        self.ped_collision_predictor = MLP(dec_hidden_units[-1], [dec_hidden_units[-1], 1])

    def forward(self, ped_features, obs_features, self_features):
        assert (self_features.shape[-1] == 7), 'Error: PINN model do not accept inputs of historical velocity'

        ped_embeddings = self.ped_encoder(ped_features)
        ped_embeddings = self.ped_processor(ped_embeddings)
        ped_embeddings = self.ped_decoder(ped_embeddings)
        ped_collision = ped_embeddings
        pred_acc_ped = self.ped_predictor(ped_embeddings)
        ped_msgs = pred_acc_ped
        pred_acc_ped = torch.sum(pred_acc_ped, dim=-2)

        if self.obs_feature_dim > 0:
            obs_embeddings = self.obs_encoder(obs_features)
            obs_embeddings = self.obs_processor(obs_embeddings)
            obs_embeddings = self.obs_decoder(obs_embeddings)
            pred_acc_obs = self.obs_predictor(obs_embeddings)
            obs_msgs = pred_acc_obs
            pred_acc_obs = torch.sum(pred_acc_obs, dim=-2)
            pred_acc_ped = pred_acc_ped + pred_acc_obs

        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau

        predictions = pred_acc_ped + pred_acc_dest

        pred_collision = self.ped_collision_predictor(ped_collision)
        pred_collision = torch.sigmoid(pred_collision).squeeze()

        out = [predictions, ped_msgs]
        if self.obs_feature_dim > 0:
            out.append(obs_msgs)
        out.append(pred_collision)
        return out


class PINNSF_multitask(nn.Module):
    """
    Inputs:
        ped_features: (C *) N * k1 * dim
        obs_features: (C *) N * k2 * dim
        self_features: (C *) N * dim
    Returns:
        predictions (acceleration): (C *) N * 2
    """

    def __init__(self, args):
        super(PINNSF_multitask, self).__init__()

        if args.dataset_name in {'ucy'}:
            self.tau = 5/6
        else:
            self.tau = 0.5
        self.ped_feature_dim = args.ped_feature_dim
        self.obs_feature_dim = args.obs_feature_dim
        self.self_feature_dim = args.self_feature_dim
        ped_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        obs_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        ped_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        obs_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        dec_hidden_units = [args.decoder_hidden_size for _ in range(args.decoder_hidden_layers)]
        dropout = args.dropout
        activation = activation_layer(args.activation)

        # encoder
        self.ped_encoder = MLP(self.ped_feature_dim, ped_enc_hidden_units)
        # if self.obs_feature_dim > 0:
        self.obs_encoder = MLP(6, obs_enc_hidden_units)

        # processor
        self.ped_processor = ResDNN(ped_enc_hidden_units[-1], ped_pro_hidden_units, activation, dropout)
        self.obs_processor = ResDNN(obs_enc_hidden_units[-1], obs_pro_hidden_units, activation, dropout)

        # decoder: node function
        self.ped_decoder = MLP(ped_pro_hidden_units[-1][-1], dec_hidden_units)
        self.obs_decoder = MLP(obs_pro_hidden_units[-1][-1], dec_hidden_units)

        self.ped_predictor = MLP(dec_hidden_units[-1], [2])
        self.obs_predictor = MLP(dec_hidden_units[-1], [2])
        # self.obs_predictor = self.ped_predictor

        self.ped_collision_predictor = MLP(ped_pro_hidden_units[-1][-1], [dec_hidden_units[-1], 1])

    def forward(self, ped_features, obs_features, self_features):
        assert (self_features.shape[-1] == 7), 'Error: PINN model do not accept inputs of historical velocity'

        ped_embeddings = self.ped_encoder(ped_features)
        ped_embeddings = self.ped_processor(ped_embeddings)
        ped_msgs = ped_embeddings
        ped_embeddings = torch.sum(ped_embeddings, dim=-2)
        ped_embeddings = self.ped_decoder(ped_embeddings)
        pred_acc_ped = self.ped_predictor(ped_embeddings)

        if self.obs_feature_dim > 0:
            obs_embeddings = self.obs_encoder(obs_features)
            obs_embeddings = self.obs_processor(obs_embeddings)
            obs_msgs = obs_embeddings
            obs_embeddings = torch.sum(obs_embeddings, dim=-2)
            obs_embeddings = self.obs_decoder(obs_embeddings)
            pred_acc_ped += self.obs_predictor(obs_embeddings)

        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau

        predictions = pred_acc_ped + pred_acc_dest

        pred_collision = self.ped_collision_predictor(ped_msgs)
        pred_collision = torch.sigmoid(pred_collision).squeeze()

        out = [predictions, ped_msgs]
        if self.obs_feature_dim > 0:
            out.append(obs_msgs)
        out.append(pred_collision)
        return out

class PINNSF_polar_bottleneck_collision(nn.Module):
    """
    Inputs:
        ped_features: (C *) N * k1 * dim
        obs_features: (C *) N * k2 * dim
        self_features: (C *) N * dim
    Returns:
        predictions (acceleration): (C *) N * 2
    """

    def __init__(self, args):
        super(PINNSF_polar_bottleneck_collision, self).__init__()

        self.time_unit = args.time_unit
        self.collision_threshold = args.collision_threshold
        self.tau = 2
        self.ped_feature_dim = args.ped_feature_dim
        self.obs_feature_dim = args.obs_feature_dim
        self.self_feature_dim = args.self_feature_dim
        ped_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        obs_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        ped_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        obs_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        dec_hidden_units = [args.decoder_hidden_size for _ in range(args.decoder_hidden_layers)]
        dropout = args.dropout
        activation = activation_layer(args.activation)

        # encoder
        self.ped_encoder = MLP(self.ped_feature_dim, ped_enc_hidden_units)
        # if self.obs_feature_dim > 0:
        self.obs_encoder = MLP(6, obs_enc_hidden_units)

        # processor
        self.ped_processor = ResDNN(ped_enc_hidden_units[-1], ped_pro_hidden_units, activation, dropout)
        self.obs_processor = ResDNN(obs_enc_hidden_units[-1], obs_pro_hidden_units, activation, dropout)

        # decoder: node function
        self.ped_decoder = MLP(ped_pro_hidden_units[-1][-1], dec_hidden_units)
        self.obs_decoder = MLP(obs_pro_hidden_units[-1][-1], dec_hidden_units)

        self.ped_predictor = MLP(dec_hidden_units[-1], [2])
        self.obs_predictor = MLP(dec_hidden_units[-1], [2])
        # self.obs_predictor = self.ped_predictor

    def forward(self, ped_features, obs_features, self_features):
        assert (self_features.shape[-1] == 7), 'Error: PINN model do not accept inputs of historical velocity'

        polar_base = self_features[..., -5:-3]  # c, t, n, 2
        polar_base = DATA.Pedestrians.get_heading_direction(polar_base)

        ped_embeddings = self.ped_encoder(ped_features)
        ped_embeddings = self.ped_processor(ped_embeddings)
        ped_embeddings = self.ped_decoder(ped_embeddings)
        pred_acc_ped = self.ped_predictor(ped_embeddings)
        ped_msgs = pred_acc_ped
        pred_acc_ped = torch.sum(pred_acc_ped, dim=-2)
        pred_acc_ped = DATA.TimeIndexedPedDataPolarCoor.polar_to_cart(pred_acc_ped, polar_base)

        if self.obs_feature_dim > 0:
            obs_embeddings = self.obs_encoder(obs_features)
            obs_embeddings = self.obs_processor(obs_embeddings)
            obs_embeddings = self.obs_decoder(obs_embeddings)
            pred_acc_obs = self.obs_predictor(obs_embeddings)
            obs_msgs = pred_acc_obs
            pred_acc_obs = torch.sum(pred_acc_obs, dim=-2)
            pred_acc_ped += DATA.TimeIndexedPedDataPolarCoor.polar_to_cart(pred_acc_obs, polar_base)

        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau

        predictions = pred_acc_ped + pred_acc_dest  # c,t,n,2

        # collision modeling, step 1: get physical quantities
        reaction_radius = self.collision_threshold + 1.34 * 2 * self.time_unit
        # reaction_radius = 0.01
        pji = ped_features[..., :2]  # pb - pa, c,t,n,k,2
        pji_ = pji.clone()
        pji_[pji_.isnan()] = 0.
        pji = pji_
        norm_pji = torch.norm(pji, p=2, dim=-1)  # c,t,n,k
        norm_pji = norm_pji + 1e-6  # to avoid zero division
        nji = pji / norm_pji.unsqueeze(-1)  # c,t,n,k,2
        vi = self_features[..., 2:4]  # va, c,t,n,2
        vji = ped_features[..., 2:4]  # vb - va, c,t,n,k,2
        vi = vi.unsqueeze(-2).repeat(*[1]*(vi.dim()-1), vji.shape[-2], 1)
        vj = vji + vi  # vb, c,t,n,k,2

        collision_flag = ((reaction_radius >= norm_pji) & (norm_pji > 1e-4)).float()  # if collision_flag==1, two collapse. c,t,n,k

        # if torch.sum(collision_flag) > 0:
        #     print('find collision {}'.format(torch.sum(collision_flag)))

        interaction_flag = torch.sum(vi * pji, dim=-1) * torch.sum(vj * (-pji), dim=-1)
        interaction_flag_ = interaction_flag.clone()
        interaction_flag_[interaction_flag_.isnan()] = 0
        interaction_flag_[interaction_flag_ <= 0] = 0
        interaction_flag_[interaction_flag_ > 0] = 1.
        encounter_flag = collision_flag * interaction_flag_   # c,t,n,k 代表一个人和他视野内的k个人是否是面对面的碰撞
        chasing_flag = collision_flag * (1 - interaction_flag_)

        # step 2: 处理相向碰撞 (ai_c: 法向存在一个加速度使得下一step速度趋向0; ai_nji:预测加速度不能有朝向碰撞方向的）
        norm_pji_ = norm_pji * encounter_flag
        norm_pji2 = norm_pji_.clone()
        norm_pji2[norm_pji2 < 1e-4] += 100
        idx = torch.min(norm_pji2, dim=-1).indices  # c,t,n
        idx = idx.unsqueeze(-1).unsqueeze(-1).repeat(*[1]*(idx.dim()), 1, 2)
        nji_c = torch.gather(nji, -2, idx).squeeze()  # c,t,n,2
        ai_c = -torch.sum(vi[..., 0, :] * nji_c, dim=-1, keepdim=True) * nji_c / self.time_unit  # c,t,n,2
        ai_c = ai_c * (torch.sum(encounter_flag, dim=-1, keepdim=True) > 0)  # 只留下有碰撞的人
        predictions_ = predictions * (torch.sum(encounter_flag, dim=-1, keepdim=True) > 0)  # c,t,n,2
        ai_nji = torch.sum(predictions_ * nji_c, dim=-1, keepdim=True)
        ai_nji = ai_nji * (ai_nji > 0)
        predictions_ = predictions_ - ai_nji * nji_c
        predictions_ = predictions_ + ai_c
        predictions = predictions + predictions_

        # step 3: 处理同向碰撞
        norm_pji_ = norm_pji * chasing_flag
        norm_pji2 = norm_pji_.clone()
        norm_pji2[norm_pji2 < 1e-4] += 100
        idx = torch.min(norm_pji2, dim=-1).indices  # c,t,n
        idx = idx.unsqueeze(-1).unsqueeze(-1).repeat(*[1]*(idx.dim()), 1, 2)
        nji_c = torch.gather(nji, -2, idx).squeeze()  # c,t,n,2
        vji_c = torch.gather(vji, -2, idx).squeeze()  # c,t,n,2
        ai_c = torch.sum(vji_c * nji_c, dim=-1, keepdim=True)
        ai_c_ = ai_c * (ai_c < 0)  # i比j要快才会进一步碰撞
        ai_c_ = ai_c_ * nji_c / self.time_unit  # c,t,n,2
        ai_c_ = ai_c_ * (torch.sum(chasing_flag, dim=-1, keepdim=True) > 0)  # 只留下有碰撞的人
        predictions_ = predictions * (torch.sum(chasing_flag, dim=-1, keepdim=True) > 0)  # c,t,n,2
        ai_nji = torch.sum(predictions_ * nji_c, dim=-1, keepdim=True)
        ai_nji = ai_nji * (ai_nji > 0) * (ai_c < 0)  # 当且仅当i比j快，且预测加速度还在法向方向的时候需要处理
        predictions_ = predictions_ - ai_nji * nji_c
        predictions_ = predictions_ + ai_c_
        predictions = predictions + predictions_

        out = [predictions, ped_msgs]
        if self.obs_feature_dim > 0:
            out.append(obs_msgs)
        return out


class PINNSF_polar_bottleneck(nn.Module):
    """
    Inputs:
        ped_features: (C *) N * k1 * dim
        obs_features: (C *) N * k2 * dim
        self_features: (C *) N * dim
    Returns:
        predictions (acceleration): (C *) N * 2
    """

    def __init__(self, args):
        super(PINNSF_polar_bottleneck, self).__init__()

        self.time_unit = args.time_unit
        self.collision_threshold = args.collision_threshold
        self.tau = 2
        self.ped_feature_dim = args.ped_feature_dim
        self.obs_feature_dim = args.obs_feature_dim
        self.self_feature_dim = args.self_feature_dim
        ped_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        obs_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        ped_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        obs_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        dec_hidden_units = [args.decoder_hidden_size for _ in range(args.decoder_hidden_layers)]
        dropout = args.dropout
        activation = activation_layer(args.activation)

        # encoder
        self.ped_encoder = MLP(self.ped_feature_dim, ped_enc_hidden_units)
        # if self.obs_feature_dim > 0:
        self.obs_encoder = MLP(6, obs_enc_hidden_units)

        # processor
        self.ped_processor = ResDNN(ped_enc_hidden_units[-1], ped_pro_hidden_units, activation, dropout)
        self.obs_processor = ResDNN(obs_enc_hidden_units[-1], obs_pro_hidden_units, activation, dropout)

        # decoder: node function
        self.ped_decoder = MLP(ped_pro_hidden_units[-1][-1], dec_hidden_units)
        self.obs_decoder = MLP(obs_pro_hidden_units[-1][-1], dec_hidden_units)

        self.ped_predictor = MLP(dec_hidden_units[-1], [2])
        self.obs_predictor = MLP(dec_hidden_units[-1], [2])
        # self.obs_predictor = self.ped_predictor

    def forward(self, ped_features, obs_features, self_features):
        assert (self_features.shape[-1] == 7), 'Error: PINN model do not accept inputs of historical velocity'

        polar_base = self_features[..., -5:-3]  # c, t, n, 2
        polar_base = DATA.Pedestrians.get_heading_direction(polar_base)

        ped_embeddings = self.ped_encoder(ped_features)
        ped_embeddings = self.ped_processor(ped_embeddings)
        ped_embeddings = self.ped_decoder(ped_embeddings)
        pred_acc_ped = self.ped_predictor(ped_embeddings)

        ped_polar_base = polar_base.unsqueeze(-2).repeat(*([1]*(polar_base.dim()-1)), pred_acc_ped.shape[-2], 1)
        pred_acc_ped = DATA.TimeIndexedPedDataPolarCoor.polar_to_cart(pred_acc_ped, ped_polar_base)
        ped_msgs = pred_acc_ped
        pred_acc_ped = torch.sum(pred_acc_ped, dim=-2)

        if self.obs_feature_dim > 0:
            obs_embeddings = self.obs_encoder(obs_features)
            obs_embeddings = self.obs_processor(obs_embeddings)
            obs_embeddings = self.obs_decoder(obs_embeddings)
            pred_acc_obs = self.obs_predictor(obs_embeddings)
            obs_polar_base = polar_base.unsqueeze(-2).repeat(*([1]*(polar_base.dim()-1)), pred_acc_obs.shape[-2], 1)
            pred_acc_obs = DATA.TimeIndexedPedDataPolarCoor.polar_to_cart(pred_acc_obs, obs_polar_base)
            obs_msgs = pred_acc_obs
            pred_acc_obs = torch.sum(pred_acc_obs, dim=-2)
            pred_acc_ped += pred_acc_obs

        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau

        predictions = pred_acc_ped + pred_acc_dest  # c,t,n,2

        out = [predictions, ped_msgs]
        if self.obs_feature_dim > 0:
            out.append(obs_msgs)
        return out


class Base_test(nn.Module):
    """
    仅仅用目标力的社会力模型
    Inputs:
        ped_features: (C *) N * k1 * dim
        obs_features: (C *) N * k2 * dim
        self_features: (C *) N * dim
    Returns:
        predictions (acceleration): (C *) N * 2
    """

    def __init__(self, args):
        super(Base_test, self).__init__()

        self.tau = 2
        self.ped_feature_dim = args.ped_feature_dim
        self.obs_feature_dim = args.obs_feature_dim
        self.self_feature_dim = args.self_feature_dim
        ped_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        obs_enc_hidden_units = [args.encoder_hidden_size for _ in range(args.encoder_hidden_layers)]
        ped_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        obs_pro_hidden_units = [[args.processor_hidden_size] for _ in range(args.processor_hidden_layers)]
        dec_hidden_units = [args.decoder_hidden_size for _ in range(args.decoder_hidden_layers)]
        dropout = args.dropout
        activation = activation_layer(args.activation)

        # encoder
        self.ped_encoder = MLP(self.ped_feature_dim, ped_enc_hidden_units)
        # if self.obs_feature_dim > 0:
        self.obs_encoder = MLP(6, obs_enc_hidden_units)

        # processor
        self.ped_processor = ResDNN(ped_enc_hidden_units[-1], ped_pro_hidden_units, activation, dropout)
        self.obs_processor = ResDNN(obs_enc_hidden_units[-1], obs_pro_hidden_units, activation, dropout)

        # decoder: node function
        self.ped_decoder = MLP(ped_pro_hidden_units[-1][-1], dec_hidden_units)
        self.obs_decoder = MLP(obs_pro_hidden_units[-1][-1], dec_hidden_units)

        self.ped_predictor = MLP(dec_hidden_units[-1], [2])
        self.obs_predictor = MLP(dec_hidden_units[-1], [2])
        # self.obs_predictor = self.ped_predictor

    def forward(self, ped_features, obs_features, self_features):
        assert (self_features.shape[-1] == 7), 'Error: PINN model do not accept inputs of historical velocity'

        ped_embeddings = self.ped_encoder(ped_features)
        ped_embeddings = self.ped_processor(ped_embeddings)
        ped_msgs = ped_embeddings
        ped_embeddings = torch.sum(ped_embeddings, dim=-2)
        ped_embeddings = self.ped_decoder(ped_embeddings)
        pred_acc_ped = self.ped_predictor(ped_embeddings)

        if self.obs_feature_dim > 0:
            obs_embeddings = self.obs_encoder(obs_features)
            obs_embeddings = self.obs_processor(obs_embeddings)
            obs_msgs = obs_embeddings
            obs_embeddings = torch.sum(obs_embeddings, dim=-2)
            obs_embeddings = self.obs_decoder(obs_embeddings)
            pred_acc_ped += self.obs_predictor(obs_embeddings)

        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau

        predictions = pred_acc_ped + pred_acc_dest

        out = [predictions, pred_acc_dest]
        return out