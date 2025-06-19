from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution
import zuko
from sbi.inference import FMPE
from sbi.neural_nets.estimators.flowmatching_estimator import FlowMatchingEstimator, VectorFieldNet
from sbi.neural_nets.net_builders.flowmatching_nets import GLU
from sbi.utils.nn_utils import get_numel
from sbi.utils.sbiutils import standardizing_net, z_score_parser, z_standardization
from sbi.utils.user_input_checks import check_data_device


class ResNetBlock(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, condition_dim: int, activation_fn: nn.Module, batch_norm: bool, dropout: float):
        super(ResNetBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity()
        self.glu1 = GLU(hidden_dim, condition_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim) if batch_norm else nn.Identity()
        self.glu2 = GLU(input_dim, condition_dim)
        self.activation = activation_fn
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.glu1(out, condition)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.glu2(out, condition)
        out = self.dropout(out)

        return out + residual


class ResNetWithGLUConditioning(VectorFieldNet):
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: Sequence[int],
        condition_dim: int,
        output_dim: int,
        activation_fn: nn.Module = nn.GELU(),
        batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        super(ResNetWithGLUConditioning, self).__init__()
        self.blocks = nn.ModuleList()
        for hidden_dim in hidden_layers:
            self.blocks.append(ResNetBlock(input_dim, hidden_dim, condition_dim, activation_fn, batch_norm, dropout))
        self.final_fc = nn.Linear(input_dim, output_dim)

    def forward(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor:
        glu_condition = torch.cat([t, theta], dim=-1)
        out = x
        for block in self.blocks:
            out = block(out, glu_condition)
        out = self.final_fc(out)
        return out


def build_resnet_flowmatcher(
    batch_x: Tensor,
    batch_y: Tensor,
    hidden_layers: Sequence[int],
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    activation: str = "gelu",
    batch_norm: bool = False,
    dropout: float = 0.0,
    num_freqs: int = 3,
    embedding_net: nn.Module = nn.Identity(),
    **kwargs,
) -> FlowMatchingEstimator:

    check_data_device(batch_x, batch_y)
    x_numel = get_numel(batch_x)  # Number of parameters
    y_numel = get_numel(batch_y, embedding_net=embedding_net)  # Number of observations

    activation_fn = {"relu": nn.ReLU(), "elu": nn.ELU(), "gelu": nn.GELU()}.get(activation, nn.GELU())

    vectorfield_net = ResNetWithGLUConditioning(
        input_dim=y_numel,
        hidden_layers=hidden_layers,
        condition_dim=(x_numel + 2 * num_freqs),
        output_dim=x_numel,
        activation_fn=activation_fn,
        batch_norm=batch_norm,
        dropout=dropout,
    )

    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x_bool:
        t_mean, t_std = z_standardization(batch_x, structured_x)
        z_score_transform = torch.distributions.AffineTransform(-t_mean / t_std, 1 / t_std)
    else:
        z_score_transform = zuko.transforms.IdentityTransform()

    z_score_y_bool, structured_y = z_score_parser(z_score_y)
    if z_score_y_bool:
        embedding_net = nn.Sequential(standardizing_net(batch_y, structured_y), embedding_net)

    return FlowMatchingEstimator(
        net=vectorfield_net,
        input_shape=batch_x[0].shape,
        condition_shape=batch_y[0].shape,
        zscore_transform_input=z_score_transform,
        embedding_net=embedding_net,
        num_freqs=num_freqs,
        **kwargs,
    )


class CustomFMPE(FMPE):

    def __init__(
        self,
        prior: Optional[Distribution],
        device: str = "cpu",
        hidden_layers: Sequence[int] = (32, 64, 128, 256, 512, 1024, 1024, 1024, 512, 128, 64, 32),
        activation: str = "gelu",
        batch_norm: bool = False,
        dropout: float = 0.0
    ):

        self._build_neural_net = lambda batch_theta, batch_x: build_resnet_flowmatcher(
            batch_theta,
            batch_x,
            hidden_layers=hidden_layers,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

        super().__init__(
            prior=prior,
            density_estimator=self._build_neural_net,
            device=device,
        )
