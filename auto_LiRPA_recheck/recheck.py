import os

os.environ['AUTOLIRPA_DEBUG'] = '1'

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from torch import nn, tensor, no_grad
import numpy as np


W_1 = tensor([[1., -1.],
              [2., -2.]])
b_1 = tensor([1., 1.])
W_2 = tensor([[1., -1.],
              [2., -2.]])
b_2 = tensor([2., 2.])
W_3 = tensor([[-1., 1.]])


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features = 2, out_features = 2, bias = True)
        self.layer2 = nn.Linear(in_features = 2, out_features = 2, bias = True)
        self.layer3 = nn.Linear(in_features = 2, out_features = 1, bias = False)
        self.relu = nn.ReLU()
        self._set_weights()

    @no_grad()
    def _set_weights(self):
        self.layer1.weight.copy_(W_1)
        self.layer1.bias.copy_(b_1)
        self.layer2.weight.copy_(W_2)
        self.layer2.bias.copy_(b_2)
        self.layer3.weight.copy_(W_3)

    def forward(self, x):
        x = self.layer1(x)
        x_shortcut = x
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x + x_shortcut)
        return x


model = MyModel()
model.eval()
my_input = tensor([[0., 0.]])
# Wrap the model with auto_LiRPA.
model = BoundedModule(model, my_input, verbose=True)
# Define perturbation. Here we add Linf perturbation to input data.
ptb = PerturbationLpNorm(norm = np.inf, eps = 1)
# Make the input a BoundedTensor with the pre-defined perturbation.
my_input = BoundedTensor(my_input, ptb)
# Regular forward propagation using BoundedTensor works as usual.
prediction = model(my_input)
# Compute LiRPA bounds using the backward mode bound propagation (CROWN).
lb, ub = model.compute_bounds(x = (my_input,), method = "CROWN")
# Print result
print(f'{lb = }\n{ub = }')