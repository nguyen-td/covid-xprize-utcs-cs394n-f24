import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, num_inputs=182, num_outputs=12, hidden_layers=None):
        super(Network, self).__init__()

        # Define the number of inputs and outputs
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # Define the hidden layers if provided; otherwise, use no hidden layers
        if hidden_layers is None:
            hidden_layers = []

        # Initialize hidden layers as a module list
        self.hidden_layers = nn.ModuleList()

        # Define the input size for the first hidden layer
        input_size = num_inputs

        # Create hidden layers
        for hidden_size in hidden_layers:
            self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            self._initialize_weights(self.hidden_layers[-1])
            input_size = hidden_size

        # Output layer
        self.output_layer = nn.Linear(input_size, num_outputs)
        self._initialize_weights(self.output_layer)

    def _initialize_weights(self, layer):
        # Initialize weights using a normal distribution with mean 0 and std 1
        nn.init.normal_(layer.weight, mean=0.0, std=1.0)

        # Initialize biases with a normal distribution and clamp between -30 and 30
        nn.init.normal_(layer.bias, mean=0.0, std=1.0)
        layer.bias.data = torch.clamp(layer.bias.data, min=-30.0, max=30.0)

    def forward(self, x):
        # Forward pass through hidden layers with sigmoid activation
        for layer in self.hidden_layers:
            x = torch.sigmoid(layer(x))

        # Forward pass through the output layer with sigmoid activation
        x = torch.sigmoid(self.output_layer(x))
        return x
