import torch
from typing import List, Literal

from warnings import filterwarnings
filterwarnings("ignore")


"""

  -- Optimizer --

  Writer : Al-Fariqy Raihan Azhwar
  NPM    : 202143501514
  Class  : R8Q
  Email  : alfariqyraihan@gmail.com

"""


class AdaptiveMomentEstimation:
  def __init__(
    self, parameters: List[torch.Tensor],
    learning_rate:    float = 0.001, 
    beta1:            float = 0.9, 
    beta2:            float = 0.999, 
    epsilon:          float = 1e-8
  ) -> None:
    self.parameters: List[torch.Tensor] = list(parameters)

    self.learning_rate: float = learning_rate
    self.epsilon:       float = epsilon
    self.beta1:         float = beta1
    self.beta2:         float = beta2

    # exponential moving average of gradient (first moment, momentum)
    self.first_moment_estimates: List[torch.Tensor] = \
      [torch.zeros_like(parameter) for parameter in self.parameters]

    # exponential moving average of squared gradient (second moment, velocity)
    self.second_moment_estimates: List[torch.Tensor] = \
      [torch.zeros_like(parameter) for parameter in self.parameters]

    # timestep (for bias correction)
    self.timestep: Literal = 0


  """
    [ name ]:
      zero_grad (return dtype: None)

    [ description ]
      set gradients to zero before backward
  """
  def zero_grad(self) -> None:
    for parameter in self.parameters:
      if parameter.grad is not None: parameter.grad.zero_()


  """
    [ name ]:
      step (return dtype: None)

    [ description ]
      update parameters
  """
  def step(self) -> None:
    self.timestep += 1
    for _idx, parameter in enumerate(self.parameters):
      if parameter.grad is None: continue

      gradient: torch.Tensor = parameter.grad

      # first moment estimates (momentum)
      # formula: 
      #   momentum[i] = beta1 * momentum[i] + (1 - beta1) * gradient
      self.first_moment_estimates[_idx] = self.beta1 * \
        self.first_moment_estimates[_idx] + (1 - self.beta1) * gradient

      # second moment estimates (velocity)
      # formula: 
      #   velocity[i] = beta2 * velocity[i] + (1 - beta2) * gradient ** 2
      self.second_moment_estimates[_idx] = self.beta2 * \
        self.second_moment_estimates[_idx] + (1 - self.beta2) * gradient ** 2

      # corrected first moment (momentum)
      # formula: 
      #   m_hat = momentum[i] / (1 - beta1 ** t)
      corrected_first_moment = self.first_moment_estimates[_idx] \
        / (1 - self.beta1 ** self.timestep)

      # corrected second moment (velocity)
      # formula: 
      #   v_hat = velocity[i] / (1 - beta2 ** t)
      corrected_second_moment = self.second_moment_estimates[_idx] \
        / (1 - self.beta2 ** self.timestep)

      # formula: 
      #   param.data -= lr * m_hat / (sqrt(v_hat) + epsilon)
      parameter.data -= self.learning_rate * corrected_first_moment \
        / (torch.sqrt(corrected_second_moment) + self.epsilon)
