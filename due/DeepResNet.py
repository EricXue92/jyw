import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepResNet(nn.Module):

    def __init__(self, input_dim, num_layers = 3, num_hidden = 128,
                 activation = "relu", num_outputs = 1, dropout_rate = 0.1):

        super().__init__()

        # Defines class meta data.
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.num_hidden = num_hidden
        self.dropout_rate = dropout_rate

        # Defines the First layers.
        self.input_layer = nn.Linear(input_dim, num_hidden)
        # self.input_layer.weight.requires_grad = False  # Non-trainable

        # Residual layers
        self.residual_layers = nn.ModuleList([self.make_dense_layer() for _ in range(num_layers)])

        # Defines the output layer.
        if self.num_outputs is not None:
            self.last = self.make_output_layer(num_outputs)

        # Activation function
        self.activation_function = self.get_activation_function(activation)

    def forward(self, inputs):

        hidden = self.input_layer(inputs)
        hidden = self.activation_function(hidden)  # Apply activation after input layer

        # Computes the ResNet hidden representations.
        for i in range(self.num_layers):
            resid = self.activation_function(self.residual_layers[i](hidden))
            resid = F.dropout(resid, p = self.dropout_rate, training=self.training)
            hidden = hidden + resid  #  # Residual connection

        # Output layer
        if self.num_outputs is not None:
            hidden = self.last(hidden)
        return hidden

    def make_dense_layer(self):
        """Uses the Dense layer as the hidden layer."""
        return nn.Linear(self.num_hidden, self.num_hidden)

    def make_output_layer(self, num_outputs):
        """Uses the Dense layer as the output layer."""
        return nn.Linear(self.num_hidden, num_outputs)

    def get_activation_function(self, activation):
        """Returns the activation function based on the input string."""
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU()
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation function: {activation}")
        return activations[activation]
