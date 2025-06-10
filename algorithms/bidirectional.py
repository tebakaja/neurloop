from typing import List, Tuple
from concurrent.futures import Future, ThreadPoolExecutor

import torch
import torch.nn as nn

from warnings import filterwarnings
filterwarnings("ignore")


"""

  -- Bidirectional --

  Writer : Al-Fariqy Raihan Azhwar
  NPM    : 202143501514
  Class  : R8Q
  Email  : alfariqyraihan@gmail.com

"""

class Bidirectional(nn.Module):
  def __init__(
    self,
    input_size:     int, 
    hidden_size:    int,
    rnn_cell_class: nn.Module,
    action:         str = 'training'
  ) -> None:
    super(Bidirectional, self).__init__()
    self.action:      str = action
    self.hidden_size: int = hidden_size

    # forward cell
    self.forward_rnn_cell:  nn.Module = rnn_cell_class(input_size, hidden_size)

    # backward cell
    self.backward_rnn_cell: nn.Module = rnn_cell_class(input_size, hidden_size)

  
  """
    [ name ]:
      __forward_pass (return dtype: torch.Tensor)

    [ parameters ]
      - sequence_length      (dtype: int)
      - input_sequence       (dtype: torch.Tensor)
      - forward_hidden_state (dtype: torch.Tensor)

    [ description ]
      forward pass
  """
  def __forward_pass(
    self,
    sequence_length:      int, 
    input_sequence:       torch.Tensor,
    forward_hidden_state: torch.Tensor
  ) -> torch.Tensor:
    forward_outputs: List[torch.Tensor] = []
    for timestep in range(sequence_length):
      current_input: torch.Tensor = input_sequence[:, timestep, :]

      forward_hidden_state: torch.Tensor = \
        self.forward_rnn_cell(current_input, forward_hidden_state)
      forward_outputs.append(forward_hidden_state)

    forward_outputs: torch.Tensor = \
      torch.stack(tensors = forward_outputs, dim = 1)
    return forward_outputs


  """
    [ name ]:
      __backward_pass (return dtype: torch.Tensor)

    [ parameters ]
      - sequence_length       (dtype: int)
      - input_sequence        (dtype: torch.Tensor)
      - backward_hidden_state (dtype: torch.Tensor)

    [ description ]
      backward pass
  """
  def __backward_pass(
    self,
    sequence_length:       int, 
    input_sequence:        torch.Tensor,
    backward_hidden_state: torch.Tensor
  ) -> torch.Tensor:
    backward_outputs: List[torch.Tensor] = []
    for timestep in reversed(range(sequence_length)):
      current_input: torch.Tensor = input_sequence[:, timestep, :]

      backward_hidden_state: torch.Tensor = \
        self.backward_rnn_cell(current_input, backward_hidden_state)
      backward_outputs.append(backward_hidden_state)

    backward_outputs.reverse()
    backward_outputs: torch.Tensor = \
      torch.stack(tensors = backward_outputs, dim = 1)
    return backward_outputs


  """
    [ name ]:
      __parallel_forward_backward (return dtype: Tuple[torch.Tensor])

    [ parameters ]
      - sequence_length       (dtype: int)
      - input_sequence        (dtype: torch.Tensor)
      - forward_hidden_state  (dtype: torch.Tensor)
      - backward_hidden_state (dtype: torch.Tensor)

    [ description ]
      parallel forward and backward pass
  """
  def __parallel_forward_backward(
    self, 
    sequence_length:       int, 
    input_sequence:        torch.Tensor, 
    forward_hidden_state:  torch.Tensor, 
    backward_hidden_state: torch.Tensor
  ) -> Tuple[torch.Tensor]:
    with ThreadPoolExecutor(max_workers = 2) as executor:
      future_forward: Future[torch.Tensor] = executor.submit(
        self.__forward_pass, sequence_length, 
        input_sequence, forward_hidden_state
      )

      future_backward: Future[torch.Tensor] = executor.submit(
        self.__backward_pass, sequence_length, 
        input_sequence, backward_hidden_state
      )

      forward_outputs:  torch.Tensor = future_forward.result()
      backward_outputs: torch.Tensor = future_backward.result()
    return forward_outputs, backward_outputs


  """
    [ name ]:
      forward (return dtype: torch.Tensor)

    [ parameters ]
      - input_sequence (dtype: torch.Tensor)

    [ description ]
      forward bidirectional steps
  """
  def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, _ = input_sequence.shape

    forward_hidden_state:  torch.Tensor = torch.zeros(
      batch_size, self.hidden_size, 
      device = input_sequence.device
    )

    backward_hidden_state: torch.Tensor = torch.zeros(
      batch_size, self.hidden_size,
      device = input_sequence.device
    )

    if self.action == 'training':
      # forward and backward pass
      forward_outputs, backward_outputs = self.__parallel_forward_backward(
        sequence_length, input_sequence, forward_hidden_state, backward_hidden_state
      )
    
    else:
      # forward pass
      forward_outputs: torch.Tensor = self.__forward_pass(
        sequence_length, input_sequence, forward_hidden_state
      )

      # backward pass
      backward_outputs: torch.Tensor = self.__backward_pass(
        sequence_length, input_sequence, backward_hidden_state
      )

    # concatenate forward and backward outputs
    bidirectional_output: torch.Tensor = \
      torch.cat([forward_outputs, backward_outputs], dim = -1)
    return bidirectional_output
    