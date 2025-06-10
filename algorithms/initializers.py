import torch
import torch.nn as nn
from numpy import sqrt

from warnings import filterwarnings
filterwarnings("ignore")


"""

  -- Initializer --

  Writer : Al-Fariqy Raihan Azhwar
  NPM    : 202143501514
  Class  : R8Q
  Email  : alfariqyraihan@gmail.com

"""


class Initializer:
  """
    [ name ]:
      xavier_initialization (return dtype: None)

    [ parameters ]
      - kernel_parameter    (dtype: nn.Parameter)

    [ description ]
      Xavier Initialization (Glorot Initialization)
  """
  def xavier_initialization(
    self, kernel_weight: nn.Parameter
  ) -> None:
    if kernel_weight is not None:
      input_dim, output_dim = kernel_weight.shape

      # formula:
      #   sqrt(6 / (input + output))
      scaling_limit: float  = sqrt(6 / (input_dim + output_dim))
      
      with torch.no_grad():
        kernel_weight.uniform_(-scaling_limit, scaling_limit)


  """
    [ name ]:
      orthogonal_initialization (return dtype: None)

    [ parameters ]
      - recurrent_parameter     (dtype: nn.Parameter)

    [ description ]
      Orthogonal Initialization
  """
  def orthogonal_initialization(
    self, recurrent_weight: nn.Parameter
  ) -> None:
    if recurrent_weight is not None:
      input_dim, output_dim = recurrent_weight.shape

      # random matrix with normal distribution (a)
      random_matrix: torch.Tensor = torch.empty(input_dim, output_dim) \
        .normal_(mean = 0.0, std = 1.0)

      # -- QR decomposition --
      # formula: 
      #   q, r = linalg.qr(a)
      orthogonal_matrix, upper_triangular = torch.linalg.qr(random_matrix)

      # ensure the determinant of Q remains positive (preserve orientation)
      # formulas:
      #   d = diag(r)
      #   q = q * sign(d)
      diagonal_signs: torch.Tensor = torch.sign(torch.diag(upper_triangular))
      adjusted_orthogonal_matrix: torch.Tensor = orthogonal_matrix * diagonal_signs

      with torch.no_grad(): recurrent_weight.copy_(adjusted_orthogonal_matrix)


  """
    [ name ]:
      zeros_initialization (return dtype: None)

    [ parameters ]
      - bias_parameter     (dtype: nn.Parameter)

    [ description ]
      Zeros Initialization
  """
  def zeros_initialization(
    self, bias_weight: nn.Parameter
  ) -> None:
    with torch.no_grad(): bias_weight.fill_(0)
