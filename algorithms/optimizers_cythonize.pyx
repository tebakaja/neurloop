import torch

from warnings import filterwarnings
filterwarnings("ignore")


"""

  -- Optimizer (Cythonize) --

  Writer : Al-Fariqy Raihan Azhwar
  NPM    : 202143501514
  Class  : R8Q
  Email  : alfariqyraihan@gmail.com

"""


cdef class AdaptiveMomentEstimation:
  cdef object parameters
  cdef float  learning_rate

  cdef float beta1
  cdef float beta2
  cdef float epsilon
  cdef int   timestep

  cdef object first_moment_estimates
  cdef object second_moment_estimates


  def __init__(
    self, parameters,
    float learning_rate = 0.001, 
    float beta1         = 0.9, 
    float beta2         = 0.999, 
    float epsilon       = 1e-8
  ):
    self.parameters = list(parameters)

    self.learning_rate = learning_rate
    self.epsilon       = epsilon
    self.beta1         = beta1
    self.beta2         = beta2

    # exponential moving average of gradient (first moment, momentum)
    self.first_moment_estimates  = [torch.zeros_like(parameter) for parameter in self.parameters]

    # exponential moving average of squared gradient (second moment, velocity)
    self.second_moment_estimates = [torch.zeros_like(parameter) for parameter in self.parameters]

    # timestep (for bias correction)
    self.timestep = 0


  """
    [ name ]:
      zero_grad (return dtype: None)

    [ description ]
      set gradients to zero before backward
  """
  cpdef void zero_grad(self):
    for parameter in self.parameters:
      if parameter.grad is not None: parameter.grad.zero_()

  
  """
    [ name ]:
      step (return dtype: None)

    [ description ]
      update parameters
  """
  cpdef void step(self):
    self.timestep += 1
    for _idx, parameter in enumerate(self.parameters):
      if parameter.grad is None: continue

      gradient = parameter.grad

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

