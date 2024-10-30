
import torch.nn as nn
import torch.nn.functional as F
from due.layers import spectral_norm_fc

class SpectralNormResNet(nn.Module):
    def __init__(self, input_dim, features, depth, spectral_normalization,
                 coeff=0.95, n_power_iterations=1, dropout_rate=0.1,
                 num_outputs=None, activation="relu"):

        super().__init__()
        """
        ResFNN architecture
        Introduced in SNGP: https://arxiv.org/abs/2006.10108
        """

        # Input layer
        self.first = nn.Linear(input_dim, features)

        # Residual layers
        self.residuals = nn.ModuleList(
            [ nn.Linear(features, features) for i in range(depth)] )

        self.dropout = nn.Dropout(dropout_rate)

        # # Apply spectral normalization if enabled
        if spectral_normalization:
            self.first = spectral_norm_fc( self.first, coeff=coeff, n_power_iterations = n_power_iterations )

            self.residuals = nn.ModuleList([
                spectral_norm_fc(layer, coeff=coeff, n_power_iterations=n_power_iterations)
                for layer in self.residuals
            ])

        # Output layer (optional)
        self.num_outputs = num_outputs

        if num_outputs is not None:
            self.last = nn.Linear(features, num_outputs)

        # Activation function
        self.activation = self.get_activation_function(activation)

    def forward(self, x):
        x = self.activation(self.first(x))
        for residual in self.residuals:
            x = x + self.dropout(self.activation(residual(x)))
        if self.num_outputs is not None:
            x = self.last(x)
        return x


    @staticmethod
    def get_activation_function(activation):
        """Returns the activation function based on the input string."""
        activations = {
            "relu": F.relu,
            "elu": F.elu
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation function: {activation}")
        return activations[activation]


if __name__ == "__main__":
    pass
