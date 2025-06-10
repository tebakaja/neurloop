import torch
import torch.nn as nn

from typing import List

from algorithms.bidirectional import Bidirectional
from algorithms.gated_recurrent_unit import GatedRecurrentUnit

from warnings import filterwarnings
filterwarnings("ignore")


"""

  -- Forecasting BiGRU Model -- 

  Writer : Al-Fariqy Raihan Azhwar
  NPM    : 202143501514
  Class  : R8Q
  Email  : alfariqyraihan@gmail.com

"""


class ForecastingBiGRUModel(nn.Module):
  def __init__(
    self, 
    input_size:          int, 
    hidden_sizes:        List[int], 
    output_size:         int, 
    dropout_probability: float = 0.3, 
    action:              str   = 'training'
  ) -> None:
    super(ForecastingBiGRUModel, self).__init__()
    self.action:                str = action
    self.hidden_sizes:    List[int] = hidden_sizes
    self.dropout_probability: float = dropout_probability

    # stack of bidirectional gru layers
    # example:
    ##  hidden_sizes = [128, 64, 32]
    ##  stacks:
    ##   - Bidirectional(input_size = 128, hidden_size = 128, ...) 
    ##      -> output (tensor shape): 128 * 2 (because it's bidirectional..)
    ##
    ##   - Bidirectional(input_size = 128 * 2, hidden_size = 64,  ...) 
    ##       -> output (tensor shape):  64 * 2 (because it's bidirectional..)
    ##
    ##   - Bidirectional(input_size =  64 * 2, hidden_size = 32,  ...) 
    ##       -> output (tensor shape):  32 * 2 (because it's bidirectional..)
    ##
    bidirectional_layers: List[Bidirectional] = []
    for layer_index, hidden_size in enumerate(hidden_sizes):
      current_input_size: int = input_size if layer_index == 0 \
        else hidden_sizes[layer_index - 1] * 2
      bidirectional_layers.append(
        Bidirectional(current_input_size, hidden_size, GatedRecurrentUnit, self.action)
      )

    # stack of bidirectional gru layers to nn.ModuleList
    self.bidirectional_gru_layers: nn.ModuleList = nn.ModuleList(bidirectional_layers)

    # dense layer (hidden)
    ## example:
    ##  hidden_sizes = [128, 64, 32] -> nn.Linear(32 * 2, 10)
    self.hidden_dense_layer: nn.Linear = nn.Linear(hidden_sizes[-1] * 2, 10)

    # output layer (or fully connected layer)
    self.output_layer: nn.Linear = nn.Linear(10, output_size)


  """
    [ name ]:
      __tensor_dropout (return dtype: torch.Tensor)

    [ parameters ]
      - input_tensor        (dtype: torch.Tensor)
      - dropout_probability (dtype: float)

    [ description ]
      dropout tensor mechanism, for prevent overfitting
  """
  def __tensor_dropout(
    self, 
    input_tensor:        torch.Tensor, 
    dropout_probability: float
  ) -> torch.Tensor:
    if self.training:
      dropout_mask: torch.Tensor = \
        (torch.rand_like(input_tensor) > dropout_probability).float()
      return dropout_mask * input_tensor / (1.0 - dropout_probability)
    return input_tensor


  """
    [ name ]:
      forward (return dtype: torch.Tensor)

    [ parameters ]
      - input_sequence (dtype: torch.Tensor)

    [ description ]
      forward propagation steps
  """
  def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
    ##
    ##  example stack of layers:
    ##
    ##    - hidd_layer_1 = Bidirectional(input_size = 128, hidden_size = 128, ...) (input_sequence)
    ##    - drop_layer_1 = Dropout(input_tensor = hidd_layer_1,  dropout_probability = {float})
    ##
    ##    - hidd_layer_2 = Bidirectional(input_size = 128 * 2, hidden_size = 64, ...) (drop_layer_1)
    ##    - drop_layer_2 = Dropout(input_tensor = hidd_layer_2,  dropout_probability = {float})
    ##
    ##    - hidd_layer_3 = Bidirectional(input_size = 64 * 2, hidden_size = 32, ...) (drop_layer_2)
    ##    - drop_layer_3 = Dropout(input_tensor = hidd_layer_3,  dropout_probability = {float})
    ##
    for bidirectional_layer in self.bidirectional_gru_layers:
      layer_output:   torch.Tensor = bidirectional_layer(input_sequence)
      input_sequence: torch.Tensor = self.__tensor_dropout(layer_output, self.dropout_probability)

    dense_layer: torch.Tensor = self.hidden_dense_layer(input_sequence[:, -1, :])
    relu_layer:  torch.Tensor = torch.relu(dense_layer)

    fully_connected_layer: torch.Tensor = self.output_layer(relu_layer)
    return fully_connected_layer
