"""
Classes below are similar to torch.nn existing in pytorch.
Implemented for education purposes.
"""

import torch
import torch.nn.functional as F


#-------------------------------------- Linear Layer --------------------------------------#

class Linear:
    """Linear layer of neural net with number of inputs and outpus as arguments.
    """

    def __init__(self, fan_in, fan_out, bias=True):
        """Randomly generates weight and bias tensors to be used in linear layer.
        Kaiming adjustment is applied to weight tensor to prevent excessive saturation at the beginning.
        Bias initially set to zero to prevent saturation of activation function.

        Args:
            fan_in (int): Number of inputs.
            fan_out (int): Number of outputs.
            bias (bool, optional): Whether bias tensor is needed. Defaults to True.
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.bias = bias
        self.W = torch.randn((fan_in, fan_out)) / (fan_in ** 0.5)
        self.B = torch.zeros(fan_out) if bias else None

    def parameters(self):
        """Returns parameters of layer.
        """
        return [self.W, self.B] if self.bias else [self.W]

    def __call__(self, input_data):
        """Forward pass of the layer.
        """
        self.out = input_data @ self.W
        if self.bias:
            self.out += self.B
        return self.out

#-------------------------------------- Tanh Layer --------------------------------------#


class Tanh:
    """Tanh activation function module.
    Tanh(x) := (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """

    def parameters(self):
        return []

    def __call__(self, x):
        """Forward pass of Tanh.
        """
        self.out = torch.tanh(x)
        return self.out

#-------------------------------------- Batch Normalization --------------------------------------#


class BatchNorm1D:
    """The idea of batch normalization is to normalize input to activation function.
    Paper: https://arxiv.org/pdf/1502.03167.pdf
    For each batch, firstly mean and variance is calculated.
    Secondly, xi will be transformed to xi_hat := (xi - avg) / sqrt(var + eps)
    Finally, a linear transformation is passed: gamma * xi_hat + beta.
    To avoid calculation of mean and variance as a separate procedure, a momentum is being used to update these statistics.
    These statistics will be maintained to be used in forward pass during the training, or testing.
    """

    def __init__(self, dim_features, eps=1e-5, momentum=0.1, training=True):
        self.dim_features = dim_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = torch.ones(dim_features)
        self.beta = torch.zeros(dim_features)
        self.running_mean = torch.zeros(dim_features)
        self.running_var = torch.ones(dim_features)

    def parameters(self):
        return [self.gamma, self.beta]

    def __call__(self, x, train=True):
        """Implementation of batch normalization forward pass.

        Args:
            x: Input data.
            train: whether in training mode. If true, running mean and variance being update.
        """
        if x.ndim == 2:
            dim = 0
        if x.ndim == 3:
            dim = (0, 1)
        if train:
            # Compute batch statistics
            batch_mean = x.mean(dim, keepdim=True)
            batch_var = x.var(dim, keepdim=True)

            # Update running statistics using momentum
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * \
                    self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * \
                    self.running_var + self.momentum * batch_var

            # Normalize input using batch statistics
            x_hat = (x - batch_mean) / ((self.eps + batch_var) ** 0.5)
        else:
            # Normalize input using running statistics
            x_hat = (x - self.running_mean) / \
                ((self.eps + self.running_var) ** 0.5)

        # Scale and shift normalized input using learnable parameters
        self.out = self.gamma * x_hat + self.beta

        return self.out


#-------------------------------------- Embedding --------------------------------------#

class Embedding:
    """Embedding changes the dimension of input to a space with dimension provided as args.
    """

    def __init__(self, input_dim, output_dim):
        """
        Args:
            input_dim: dimension of input space.
            output_dim: dimension of output space.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A = torch.randn((self.input_dim, self.output_dim))

    def parameters(self):
        return [self.A]

    def __call__(self, x):
        """Embedding functionality.

        Args:
            x: typically a tensor with input information.

        Returns:
            A tensor which is representation of input in embedded space.
        """
        self.out = self.A[x]
        return self.out


#---------------------------------------- Flattern ----------------------------------------#

class Flatten:
    """Flattens a tensor into a 2D array while keeping the first dimension.
    """

    def __init__(self, n):
        self.n = n

    def parameters(self):
        return[]

    def __call__(self, x):
        a, b, c = x.shape
        x = x.view(a, b // self.n, c * self.n)
        if x.shape[1] == 1:
            x = x.squeeze()
        self.out = x
        return self.out


#---------------------------------------- Sequential ----------------------------------------#

class Sequential:
    """Runs forward pass for a list of layers sequentially.
    """

    def __init__(self, layers):
        self.layers = layers

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def __call__(self, x, train=True):
        for l in self.layers:
            if isinstance(l, BatchNorm1D):
                x = l(x, train)
            else:
                x = l(x)
        self.out = x
        return self.out
