import numpy as np
import torch
from sklearn.preprocessing import normalize

from .base import BaseSimulation
from .._cython.risk import erf_risk

class TensorErfCommiteeMachine():
  def __init__(self, input_size, hidden_size, W):
    self.input_size = float(input_size)
    self.hidden_size = float(hidden_size)
    self.W = torch.from_numpy(W).float()

  def __call__(self, x):
    return (
      torch.mean(
        torch.erf(
            torch.tensordot(self.W, x, dims=([-1],[-1]))/np.sqrt(2*self.input_size)
          ),
          axis = 0
      )
    )

  def get_weight(self):
    return self.W.numpy()

class TensorSimulation(BaseSimulation):
  def __init__(self, d, p, k, gamma, Wt, W0, noise = 0., activation = 'erf', disable_QM_save = False, extra_metrics = {}):
    if activation != 'erf':
      raise NotImplementedError('Numpy simulation can compute only erf activation function.')
    super().__init__(d, p, k, gamma, Wt, W0, noise, activation, disable_QM_save, extra_metrics)
    self.teacher = TensorErfCommiteeMachine(d, k, Wt)
    self.student = TensorErfCommiteeMachine(d, p, W0)
    self.theoretical_risk = erf_risk

  def _gradient_descent_step(self, y_student, y_teacher_noised, x):
    p = int(self.student.hidden_size)
    prefactor = self.gamma * (y_student-y_teacher_noised) /p * np.sqrt(2/(np.pi*self.d))
    hidden_node = torch.tensordot(self.student.W, x, ([-1],[-1]))
    xx = torch.cat([x]*p)
    hh = torch.cat([hidden_node]*self.d,1)
    self.student.W -= prefactor * torch.exp(-0.5/self.d*((hh)**2)) * xx