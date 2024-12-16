"""
Defines a pytorch policy as the agent's actor

Functions to edit:
    2. forward
    3. update
"""

import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            # HACK: 这里的是激活函数, 这里默认用的Tanh作为激活函数, 到时候可以把这个作为自定义参数然后对比一下效果
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(nn.Tanh())
        in_size = size
    layers.append(nn.Linear(in_size, output_size))

    mlp = nn.Sequential(*layers)
    return mlp


class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):    # 该类继承BasePolicy, nn.Module
    """
    Defines an MLP for supervised learning which maps observations to continuous
    actions.

    Attributes
    ----------
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
        其实就是期望值
        往常我们用神经网络作为policy就只是输出一个具体的值, 但是连续动作空间我们要得到的是一个分布, 然后从分布中采样才得到具体的值
        所以这个mean_net就是输出期望值
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions
        既然是得到一个分布, 那这个logstd就是输出标准差(std)
        有了期望值和标准差, 我们就可以采样得到具体的动作值了
        那为啥名字要带个log? 因为神经网络的输出有正有负, 所以输出的我们就令它为logstd, 然后我们自己算出std

    Methods
    -------
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim    # The dim of action space
        self.ob_dim = ob_dim    # The dim of observation space
        self.n_layers = n_layers    # 隐层数量
        self.size = size            # 隐层神经元个数
        self.learning_rate = learning_rate  # 学习率
        self.training = training    # 是否是训练模式(默认是True), 暂且不管
        self.nn_baseline = nn_baseline  # 是否使用神经网络基线(默认是False), 暂且不管

        # 构建一个名为mean_net的MLP, 负责输出期望
        self.mean_net = build_mlp(
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers, size=self.size,
        )
        self.mean_net.to(ptu.device)    # ptu.device已经在run_hw1.py里的INIT部分初始化过了

        # 因为输出有ac_dim个, 所以需要的标准差就是ac_dim个, 所以我就声明个大小为ac_dim的tensor出来
        # 然后因为std也需要可训练, 所以就用nn.Parameter把tensor包装一下, 使得这个tensor也可以参与训练的反向传播和更新
        self.logstd = nn.Parameter(
            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
        self.logstd.to(ptu.device)

        # 定义一个优化器, 这个优化器负责神经网络和期望tensor的参数
        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate
        )

    def save(self, filepath):
        """
        :param filepath: path to save MLP
        """
        # 因为该类继承nn.Module, 所以该类有state_dict()参数
        # 这里相当于把这个类的所有参数保存到filepath中了
        torch.save(self.state_dict(), filepath)

    def forward(self, observation: torch.FloatTensor) -> Any:
        """
        Defines the forward pass of the network

        :param observation: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        # TODO: implement the forward pass of the network.
        # You can return anything you want, but you should be able to differentiate
        # through it. For example, you can return a torch.FloatTensor. You can also
        # return more flexible objects, such as a
        # `torch.distributions.Distribution` object. It's up to you!
        
        observation = observation.float().to(ptu.device)
        mean = self.mean_net(observation)   # [batch_size, ac_dim]
        # logstd是标准差的对数，转换为标准差
        std = torch.exp(self.logstd)  # [ac_dim]
        std = std.expand_as(mean)   # 将std扩展为 [batch_size, ac_dim] 的形状，和mean一致
        # 使用mean和std来定义正态分布
        action_dist = torch.distributions.Normal(mean, std)
        # 这里要重采样, 这样才能支持反向传播
        sampled_action = action_dist.rsample()  # [batch_size, ac_dim]
        return sampled_action

    def update(self, observations, actions):
        """
        Updates/trains the policy

        :param observations: observation(s) to query the policy
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        # TODO: update the policy and return the loss

        observations = observations.float().to(ptu.device)
        predicted = self.forward(observations)    # [batch_size, ac_dim]
        loss = F.mse_loss(predicted, actions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            # You can add extra logging information here, but keep this line
            # 这里返回训练日志的内容
            'Training Loss': ptu.to_numpy(loss),
        }
