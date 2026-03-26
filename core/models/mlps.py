import torch
import torch.nn.functional as F
from torch import nn
from fla.modules import ShortConvolution


class MLP(nn.Module):
  def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256, **kwargs):
    super().__init__()
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
    self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

  def forward(self, x):
    # x: (bsz, T, dim)
    return self.fc2(F.silu(self.fc1(x)))


class SwiGLU(nn.Module):
  """fused GLU with an added conv, inspired by the Canon Paper"""

  def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256, **kwargs):
    super().__init__()
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    self.hidden_dim = hidden_dim
    self.fc1 = nn.Linear(dim, 2 * hidden_dim, bias=False)
    self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

  def forward(self, x):
    # x: (bsz, T, dim)
    x, z = self.fc1(x).split(self.hidden_dim, dim=2)
    return self.fc2(F.silu(x) * z)
  
class ConvSwiGLU(nn.Module):
  """fused GLU"""

  def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256,conv_size=4,conv_activation='identity',  **kwargs):
    super().__init__()
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    self.hidden_dim = hidden_dim
    self.fc1 = nn.Linear(dim, 2 * hidden_dim, bias=False)
    self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
    self.conv = ShortConvolution(hidden_dim, conv_size, activation=conv_activation)

  def forward(self, x):
    # x: (bsz, T, dim)
    x = self.fc1(x)
    x = self.conv(x)
    x,z = x.split(self.hidden_dim, dim=2)
    return self.fc2(F.silu(x) * z)

class MLPReluSquared(nn.Module):
  """MLP with ReLU squared"""

  def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256, **kwargs):
    super().__init__()
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
    self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

  def forward(self, x):
    # x: (bsz, T, dim)
    return self.fc2(F.relu(self.fc1(x)).pow(2))
