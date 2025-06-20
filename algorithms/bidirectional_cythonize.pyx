import torch
import torch.nn as nn
from concurrent.futures import Future, ThreadPoolExecutor

from warnings import filterwarnings
filterwarnings("ignore")


"""

  -- Bidirectional (Cythonize) --

  Writer : Al-Fariqy Raihan Azhwar
  NPM    : 202143501514
  Class  : R8Q
  Email  : alfariqyraihan@gmail.com

"""


"""
  [ name ]:
    forward_pass (return dtype: torch.Tensor)

  [ parameters ]
    - sequence_length      (dtype: int)
    - input_sequence       (dtype: torch.Tensor)
    - forward_hidden_state (dtype: torch.Tensor)

  [ description ]
    forward pass
"""
cdef object __forward_pass(
  int    sequence_length, 
  object input_sequence,
  object forward_hidden_state,
  object forward_rnn_cell
):
  forward_outputs = []
  for timestep in range(sequence_length):
    current_input: torch.Tensor = input_sequence[:, timestep, :]

    forward_hidden_state: torch.Tensor = \
      forward_rnn_cell(current_input, forward_hidden_state)
    forward_outputs.append(forward_hidden_state)

  forward_outputs: torch.Tensor = \
    torch.stack(tensors = forward_outputs, dim = 1)
  return forward_outputs


"""
  [ name ]:
    backward_pass (return dtype: torch.Tensor)

  [ parameters ]
    - sequence_length       (dtype: int)
    - input_sequence        (dtype: torch.Tensor)
    - backward_hidden_state (dtype: torch.Tensor)

  [ description ]
    backward pass
"""
cdef object __backward_pass(
  int    sequence_length, 
  object input_sequence,
  object backward_hidden_state,
  object backward_rnn_cell
):
  backward_outputs = []
  for timestep in reversed(range(sequence_length)):
    current_input: torch.Tensor = input_sequence[:, timestep, :]

    backward_hidden_state: torch.Tensor = \
      backward_rnn_cell(current_input, backward_hidden_state)
    backward_outputs.append(backward_hidden_state)

  backward_outputs.reverse()
  backward_outputs: torch.Tensor = \
    torch.stack(tensors = backward_outputs, dim = 1)
  return backward_outputs


"""
  [ name ]:
    parallel_forward_backward (return dtype: Tuple[torch.Tensor])

  [ parameters ]
    - sequence_length       (dtype: int)
    - input_sequence        (dtype: torch.Tensor)
    - forward_hidden_state  (dtype: torch.Tensor)
    - backward_hidden_state (dtype: torch.Tensor)

  [ description ]
    parallel forward and backward pass
"""
cdef object __parallel_forward_backward(
  int    sequence_length, 
  object input_sequence, 

  object forward_hidden_state, 
  object backward_hidden_state,

  object forward_rnn_cell,
  object backward_rnn_cell
):
  with ThreadPoolExecutor(max_workers = 2) as executor:
    future_forward: Future[torch.Tensor] = executor.submit(
      __forward_pass, sequence_length, 
      input_sequence, forward_hidden_state, forward_rnn_cell
    )

    future_backward: Future[torch.Tensor] = executor.submit(
      __backward_pass, sequence_length, 
      input_sequence, backward_hidden_state, backward_rnn_cell
    )

    forward_outputs:  torch.Tensor = future_forward.result()
    backward_outputs: torch.Tensor = future_backward.result()
  return forward_outputs, backward_outputs


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
      forward_outputs, backward_outputs = __parallel_forward_backward(
          sequence_length, input_sequence,
          forward_hidden_state, backward_hidden_state,
          self.forward_rnn_cell, self.backward_rnn_cell
        )
    
    else:
      # forward pass
      forward_outputs: torch.Tensor = __forward_pass(
          sequence_length, input_sequence,
          forward_hidden_state, self.forward_rnn_cell
        )

      # backward pass
      backward_outputs: torch.Tensor = __backward_pass(
          sequence_length, input_sequence, 
          backward_hidden_state, self.backward_rnn_cell
        )

    # concatenate forward and backward outputs
    bidirectional_output: torch.Tensor = \
      torch.cat([forward_outputs, backward_outputs], dim = -1)
    return bidirectional_output
    