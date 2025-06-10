import torch
import torch.nn as nn

from algorithms.initializers import Initializer

from warnings import filterwarnings
filterwarnings("ignore")


"""

  -- Gated Recurrent Unit (GRU) --

  Writer : Al-Fariqy Raihan Azhwar
  NPM    : 202143501514
  Class  : R8Q
  Email  : alfariqyraihan@gmail.com

"""

class GatedRecurrentUnit(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(GatedRecurrentUnit, self).__init__()

    self.input_size:  int  = input_size
    self.hidden_size: int  = hidden_size
    self.initializer: Initializer = Initializer()

    ###############################################
    # --- reset gate weight ---
    ## name: kernel_reset_gate_weight
    ## tensor dim:
    ##    input_size (features), hidden_size (neuron)
    self.kernel_reset_gate_weight: nn.Parameter = \
      nn.Parameter(torch.randn(input_size, hidden_size))

    ## name: recurrent_reset_gate_weight
    ## tensor dim:
    ##    hidden_size (neuron), hidden_size (neuron)
    self.recurrent_reset_gate_weight: nn.Parameter = \
      nn.Parameter(torch.randn(hidden_size, hidden_size))

    ## name: bias_reset_gate_weight
    ## tensor dim:
    ##    hidden_size (neuron)
    self.bias_reset_gate_weight: nn.Parameter = \
      nn.Parameter(torch.randn(hidden_size))
    # --- reset gate weight ---
    ###############################################


    ###############################################
    # --- update gate weight ---
    ## name: kernel_update_gate_weight
    ## tensor dim:
    ##    input_size (features), hidden_size (neuron)
    self.kernel_update_gate_weight: nn.Parameter = \
      nn.Parameter(torch.randn(input_size, hidden_size))

    ## name: recurrent_update_gate_weight
    ## tensor dim:
    ##    hidden_size (neuron), hidden_size (neuron)
    self.recurrent_update_gate_weight: nn.Parameter = \
      nn.Parameter(torch.randn(hidden_size, hidden_size))

    ## name: bias_update_gate_weight
    ## tensor dim:
    ##    hidden_size (neuron)
    self.bias_update_gate_weight: nn.Parameter = \
      nn.Parameter(torch.randn(hidden_size))
    # --- update gate weight ---
    ###############################################


    ###############################################
    # --- candidate activation weight ---
    ## name: kernel_candidate_activation_weight
    ## tensor dim:
    ##    input_size (features), hidden_size (neuron)
    self.kernel_candidate_activation_weight: nn.Parameter = \
      nn.Parameter(torch.randn(input_size, hidden_size))

    ## name: recurrent_candidate_activation_weight
    ## tensor dim:
    ##    hidden_size (neuron), hidden_size (neuron)
    self.recurrent_candidate_activation_weight: nn.Parameter = \
      nn.Parameter(torch.randn(hidden_size, hidden_size))

    ## name: bias_candidate_activation_weight
    ## tensor dim:
    ##    hidden_size (neuron)
    self.bias_candidate_activation_weight: nn.Parameter = \
      nn.Parameter(torch.randn(hidden_size))
    # --- candidate activation weight ---
    ###############################################

    self._init_parameters()


  def _init_parameters(self):
    # --- reset gate weight initialization ---
    self.initializer.xavier_initialization(kernel_weight        = self.kernel_reset_gate_weight)
    self.initializer.orthogonal_initialization(recurrent_weight = self.recurrent_reset_gate_weight)
    self.initializer.zeros_initialization(bias_weight           = self.bias_reset_gate_weight)

    # --- update gate weight initialization ---
    self.initializer.xavier_initialization(kernel_weight        = self.kernel_update_gate_weight)
    self.initializer.orthogonal_initialization(recurrent_weight = self.recurrent_update_gate_weight)
    self.initializer.zeros_initialization(bias_weight           = self.bias_update_gate_weight)

    # --- candidate activation weight initialization ---
    self.initializer.xavier_initialization(kernel_weight        = self.kernel_candidate_activation_weight)
    self.initializer.orthogonal_initialization(recurrent_weight = self.recurrent_candidate_activation_weight)
    self.initializer.zeros_initialization(bias_weight           = self.bias_candidate_activation_weight)


  def forward(
    self, current_input: torch.Tensor,
    hidden_state_prev:   torch.Tensor
  ) -> torch.Tensor:
    # -- reset gate --
    # formula:
    #   sigmoid((kernel * input) + (recurrent * h_prev) + bias)
    reset_gate: torch.Tensor = torch.sigmoid(
      torch.einsum('ij,jk->ik', current_input, self.kernel_reset_gate_weight) +
      torch.einsum('ij,jk->ik', hidden_state_prev, self.recurrent_reset_gate_weight) +
      self.bias_reset_gate_weight
    )


    # -- update gate --
    # formula:
    #   sigmoid((kernel * input) + (recurrent * h_prev) + bias)
    update_gate: torch.Tensor = torch.sigmoid(
      torch.einsum('ij,jk->ik', current_input, self.kernel_update_gate_weight) +
      torch.einsum('ij,jk->ik', hidden_state_prev, self.recurrent_update_gate_weight) +
      self.bias_update_gate_weight
    )


    # -- candidate activation --
    # formula:
    #   tanh((kernel * input) + (reset_gate * (recurrent * h_prev)) + bias)
    candidate_activation: torch.Tensor = torch.tanh(
      torch.einsum('ij,jk->ik', current_input, self.kernel_candidate_activation_weight) +
      reset_gate * torch.einsum('ij,jk->ik', hidden_state_prev, self.recurrent_candidate_activation_weight) +
      self.bias_candidate_activation_weight
    )


    # -- final hidden state --
    # formula:
    #   ((1 - update_gate) * h_prev) + (update_gate * candidate_activation)
    final_hidden_state: torch.Tensor = (1 - update_gate) * \
      hidden_state_prev + update_gate * candidate_activation

    return final_hidden_state
