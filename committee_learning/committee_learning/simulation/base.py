from tqdm import tqdm
import math
import torch
import numpy as np
import warnings

from .._config.python import scalar_type

class BaseSimulation():
  def __init__(self, d: int, p: int, k: int, gamma: float, Wt: np.array, W0: np.array, noise: float, activation: str, disable_QM_save: bool, extra_metrics: dict):
    self.d = d
    self.p = p
    self.k = k
    self.noise = noise
    self.gamma = gamma
    self.sqrt_noise = math.sqrt(noise)
    self.activation = activation
    self.loss = lambda ys, yt: 0.5*(ys-yt)**2

    self.saved_steps = list()
    self.saved_risks = list()
    self.saved_Ms = list()
    self.saved_Qs = list()

    self._completed_steps = 0

    if not disable_QM_save and extra_metrics != {}:
      warnings.warn('Storing Q,M and having extra metrics is redundant! Consider to store matrices and compute from results, or just compute withut storing the Q and M.')
    self.disable_QM_save = disable_QM_save
    self.extra_metrics = extra_metrics
    for metric_name in extra_metrics.keys():
      setattr(self, metric_name, list())

    # The two Commitee machines has to be initialized!
    # It is needed also the neacher weights matrix, as np.array((k,d))
    self.student = None
    self.teacher = None
    self.Wt = Wt

    # In children of this class must be defined: 
    self.P = self.Wt @ self.Wt.T / self.d
    #  - self.theoretical_risk(Q,M,P): a function that evaluates the theoretical risk

  def fit(self, steps, n_saved_points = 100, show_progress = True):
    n_saved_points = min(n_saved_points, steps)
    plot_frequency = max(1,int(steps/n_saved_points))

    for step in tqdm(range(self._completed_steps+1,self._completed_steps+steps+1), disable= not show_progress):
      # Add data if necessary
      if step%plot_frequency == 0:
        self.saved_steps.append(step)
        Ws = self.student.get_weight()

        M = (Ws @ self.Wt.T/self.d).astype(scalar_type)
        Q = (Ws @ Ws.T/self.d).astype(scalar_type)

        # Store metrics
        if not self.disable_QM_save:
          self.saved_Ms.append(M)
          self.saved_Qs.append(Q)

        self.saved_risks.append(self.theoretical_risk(Q,M,self.P))

        for metric_name, metric in self.extra_metrics.items():
          metric_list = getattr(self, metric_name)
          metric_list.append(metric(Q,M,self.P))

      # Compute the sample
      x = torch.normal(0., 1., (1,self.d,))
      y_student = self.student(x)
      with torch.no_grad():
        y_teacher_noised = self.teacher(x) + self.sqrt_noise*torch.normal(0.,1.,(1,))
      
      # Gradient descent
      self._gradient_descent_step(y_student, y_teacher_noised,x)

    self._completed_steps += steps

  def fit_logscale(self, decades, save_per_decade = 100, seed = None, show_progress = True):
    if seed is not None:
      torch.manual_seed(seed)
    for d in range(int(np.ceil(decades))+1):
      self.fit(int(10**min(d,decades))-self._completed_steps, save_per_decade, show_progress=show_progress)