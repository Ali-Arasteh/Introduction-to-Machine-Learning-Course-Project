import torch
import torch.nn as nn


def build_mlp(input_size, output_size, n_layers, size, activation=nn.Tanh(), output_activation=None):
    """
        #inputs:
            input_size: dimension of inputs.
            output_size: dimension of outputs.
            n_layers: number of the layers in the Sequential model.
            size: width of each hidden layer.(for the sake of simplicity, we take the width of all the hidden layers the same)
            activation: activation function used after each hidden layer.
            output_activation: activation function used in the last layer.

        #outputs:
            model: the implemented model.
    """
    # Sequentially append the layers to the list.
    # Use Xavier initialization for the weights.
    # Hint: Look at nn.Linear and nn.init.
    layers = []
    layers.append(nn.Linear(input_size, size))
    layers.append(activation)
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(size, size))
        layers.append(activation)
    layers.append(nn.Linear(size, output_size))
    if output_activation is not None:
        layers.append(output_activation)
    model = nn.Sequential(*layers)
    model.apply(init_weights)
    return model

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

device = torch.device("cuda")
dtype = torch.float32