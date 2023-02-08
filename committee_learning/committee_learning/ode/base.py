import numpy as np
import math
from tqdm import tqdm

from .._config.python import scalar_type

"""
These classes should be the base for all ODE integrations.
"""

class BaseODE():
  def __init__(self, P0, Q0, M0, dt, noise_term):
    assert(Q0.shape[0] == Q0.shape[1])
    assert(Q0.shape[1] == M0.shape[0])
    assert(M0.shape[1] == P0.shape[0])
    assert(P0.shape[0] == P0.shape[1])
    
    self.dt = scalar_type(dt)
    self.p = Q0.shape[0]
    self.k = P0.shape[1]

    self.P = np.array(P0, ndmin=2, dtype=scalar_type)
    self.M = np.array(M0, ndmin=2, dtype=scalar_type)
    # remember to initialize Q or Qorth

    self._simulated_time = scalar_type(0.)

    self.saved_times = []
    self.saved_risks = []

    self.noise_term = noise_term

    """
    Include here the other variables you want to store.

    Methods you need to define to make a class work:
     - self.risk(): compute the current state risk
     - self._save_step()
     - self._save_update()
    """

  def fit(self, time, n_saved_points=20, show_progress=True):
    discrete_steps = int(time/self.dt)
    n_saved_points = min(n_saved_points, discrete_steps)
    save_frequency = max(1, int(discrete_steps/n_saved_points))

    for step in tqdm(range(discrete_steps), disable=not show_progress):
      # Add data if necessary
      if step%save_frequency == 0:
        self._save_step(step)
      self._update_step()

    self._simulated_time += time

  def fit_logscale(self, decades, save_per_decade = 100, show_progress=True):
    assert(10**decades>self.dt)
    d_min = int(math.log(self.dt,10))
    for d in range(d_min,decades+1):
      self.fit(10**d-self._simulated_time, save_per_decade, show_progress=show_progress)

  def _save_step(self, step):
    "This is just the base _save_step() method. You should override it in children classes"
    self.saved_times.append(self._simulated_time + self.dt * (step+1))
    self.saved_risks.append(self.risk())


class BaseFullODE(BaseODE):
  def __init__(self, P0, Q0, M0, dt, noise_term = True, gamma_over_p = None, noise = None, quadratic_terms = False):
    super().__init__(P0, Q0, M0, dt, noise_term)

    self.Q = np.array(Q0, ndmin=2, dtype=scalar_type)
    if noise_term:
      self._gamma_over_p = scalar_type(gamma_over_p)
      self.noise = scalar_type(noise)
    self.quadratic_terms = quadratic_terms
    

    self.saved_Ms = []
    self.saved_Qs = []

  def _save_step(self, step):
    super()._save_step(step)
    self.saved_Ms.append(np.copy(self.M))
    self.saved_Qs.append(np.copy(self.Q))

class BaseLargePODE(BaseODE):
  """
  This equations are supposed to be used in the regime Q = MM^T + diag(Q^orth).
  We just need to track the diagonal of Q^orth and M, so no need to evolve a p x p matrix,
  that would be unfesible.
  """
  def __init__(self, P0, Q0, M0, dt, offdiagonal = True, d = None, noise_term = True, noise_gamma_over_p = None):
    super().__init__(P0, Q0, M0, dt, noise_term)

    if offdiagonal:
      assert(d is not None)

    self.d = d
    self.offdiagonal = offdiagonal
    self.Qorth = np.array(np.diag(np.array(Q0 - M0@M0.T, ndmin=2, dtype=scalar_type)))
    if noise_term:
      self._noise_gamma_over_p = scalar_type(noise_gamma_over_p)

    self.saved_Ms = []
    self.saved_Qorths = []

  @property
  def Q(self):
    return self.M @ self.M.T + np.diag(self.Qorth)

  def _save_step(self, step):
    super()._save_step(step)
    self.saved_Ms.append(np.copy(self.M))
    self.saved_Qorths.append(np.copy(self.Qorth))

  