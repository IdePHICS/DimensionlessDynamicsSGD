import math
import torch
import numpy as np

from .base import BaseSimulation
from .._cython.risk import erf_risk, square_risk


class CommiteeMachine(torch.nn.Module):
  def __init__(self, input_size, hidden_size, W, activation = 'square', teacher = False):
    super(CommiteeMachine, self).__init__()
    self.input_size = float(input_size)
    self.hidden_size = float(hidden_size)
    self.layer = torch.nn.Linear(input_size, hidden_size, bias=False)

    if activation == 'erf':
      self.activation = lambda x: torch.erf(x/math.sqrt(2))
    elif activation == 'square':
      self.activation = lambda x: x**2
    elif activation == 'relu':
      self.activation = lambda x: torch.maximum(x, torch.zeros(x.shape[-1]))
      
    if teacher:
      with torch.no_grad():
        self.layer.weight = torch.nn.Parameter(torch.tensor(W).float())
    else:
      self.layer.weight = torch.nn.Parameter(torch.tensor(W).float())

  def forward(self, x):
    return torch.mean(
      self.activation(
        self.layer(x)/math.sqrt(self.input_size)
      ),
      axis = -1
    )
  
  @torch.no_grad()
  def get_weight(self):
    return self.layer.weight.numpy()


class Simulation(BaseSimulation):
  def __init__(self, d, p, k, gamma, Wt, W0, noise = 0., activation = 'square', disable_QM_save = False, extra_metrics = {}):
    super().__init__(d, p, k, gamma, Wt, W0, noise, activation, disable_QM_save, extra_metrics)
    self.teacher = CommiteeMachine(d, k, activation=activation, W=Wt, teacher=True)
    self.student = CommiteeMachine(d, p, activation=activation, W=W0)

    if activation == 'erf':
      self.theoretical_risk = erf_risk
    elif activation == 'square':
      self.theoretical_risk = square_risk
    elif activation == 'relu':
      raise NotImplementedError

  def _gradient_descent_step(self, y_student, y_teacher_noised, x):
    loss = self.loss(y_student, y_teacher_noised)
    self.student.zero_grad()
    loss.backward()
    for param in self.student.parameters():
      param.data.sub_(param.grad.data * self.gamma)

class NormalizedSphericalConstraintSimulation(Simulation):
  def _gradient_descent_step(self, y_student, y_teacher_noised, x):
    loss = self.loss(y_student, y_teacher_noised)
    self.student.zero_grad()
    loss.backward()
    for param in self.student.parameters():
      param.data.sub_(param.grad.data * self.gamma)
      param.data = np.sqrt(self.d) * torch.nn.functional.normalize(param.data)

class LagrangeSphericalConstraintSimulation(Simulation):
  """
  This class turned out to be not useful, but still I keep it just for historical reasons
  """
  def _gradient_descent_step(self, y_student, y_teacher_noised, x):
    loss = self.loss(y_student, y_teacher_noised)
    self.student.zero_grad()
    loss.backward()
    for param in self.student.parameters():
      w_norm = torch.sum(param.data*param.data, dim=1,keepdims=True).repeat(1,self.d)
      projection_coeffs = torch.sum(param.grad.data*param.data, dim=1,keepdims=True).repeat(1,self.d)/w_norm
      # print(torch.sum((param.grad.data - projection_coeffs*param.data)*param.data))
      param.data.sub_(self.gamma*(param.grad.data - projection_coeffs*param.data))
      # param.data = np.sqrt(self.d) * torch.nn.functional.normalize(param.data)
      