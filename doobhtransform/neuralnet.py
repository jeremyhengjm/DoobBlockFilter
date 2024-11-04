"""
A module to approximate functions with neural networks.
"""

import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        layer_widths,
        activate_final=False,
        activation_fn=torch.nn.LeakyReLU(),
    ):
        super(MLP, self).__init__()
        layers = []
        prev_width = input_dim
        for layer_width in layer_widths:
            layers.append(torch.nn.Linear(prev_width, layer_width))
            prev_width = layer_width
        self.input_dim = input_dim
        self.layer_widths = layer_widths
        self.layers = torch.nn.ModuleList(layers)
        self.activate_final = activate_final
        self.activation_fn = activation_fn

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)
        if self.activate_final:
            x = self.activation_fn(x)
        return x


class V_Network(torch.nn.Module):

    def __init__(self, num_obs, dimension_state, dimension_obs, config):
        super().__init__()
        input_dimension = dimension_state + dimension_obs
        layers = config["layers"]
        self.standardization = config.get("standardization")
        self.net = MLP(input_dimension, layer_widths=layers + [1])
        self.net = torch.nn.ModuleList(
            [MLP(input_dimension, layer_widths=layers + [1]) for t in range(num_obs)]
        )

    def forward(self, t, x, y):
        # t (int), x.shape = (N, d),y.shape=(N, p)
        N = x.shape[0]

        if len(y.shape) == 1:
            y_ = y.repeat((N, 1))
        else:
            y_ = y

        if self.standardization:
            x_c = (x - self.standardization["x_mean"]) / self.standardization["x_std"]
            y_c = (y_ - self.standardization["y_mean"]) / self.standardization["y_std"]
        else:
            x_c = x
            y_c = y_

        h = torch.cat([x_c, y_c], -1)  # size (N, 1+d+p)
        out = torch.squeeze(self.net[t](h))  # size (N)
        return out


class Z_Network(torch.nn.Module):
    def __init__(self, num_obs, dimension_state, dimension_obs, config):
        super().__init__()
        layers = config["layers"]
        self.standardization = config.get("standardization")
        input_dimension = dimension_state + dimension_obs + 1
        self.net = torch.nn.ModuleList(
            [
                MLP(input_dimension, layer_widths=layers + [dimension_state])
                for t in range(num_obs)
            ]
        )

    def forward(self, t, s, x, y):
        # t (int), s.shape = [1], x.shape = (N, d)
        N = x.shape[0]
        if len(s.shape) == 0:
            s_ = s.repeat((N, 1))
        else:
            s_ = s

        if len(y.shape) == 1:
            y_ = y.repeat((N, 1))
        else:
            y_ = y

        if self.standardization:
            x_c = (x - self.standardization["x_mean"]) / self.standardization["x_std"]
            y_c = (y_ - self.standardization["y_mean"]) / self.standardization["y_std"]
        else:
            x_c = x
            y_c = y_

        h = torch.cat([s_, x_c, y_c], -1)
        out = self.net[t](h)
        return out
