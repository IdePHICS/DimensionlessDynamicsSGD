import numpy as np
from tqdm import tqdm

class OrnsteinUhlenbeck():
  def __init__(self, X0, mu, sigma, dt, save_interval, seed = None):
    """"
    We are simulating:
    dX = -mu X dt + sigma dW
    """
    self.ts = [0.]
    self.Xs = [X0]
    self._steps = 0
    self._currentX = X0
    self.mu = mu
    self.sigma = sigma
    self.dt = dt 
    self.save_interval = save_interval
    self.rng = np.random.default_rng(seed)

  def _step(self):
    self._steps += 1
    self._currentX += (
      - self.mu * self._currentX * self.dt + 
      self.sigma * self.rng.normal() * np.sqrt(self.dt)
    )
    current_time = (self._steps * self.dt)
    if self._steps % int(self.save_interval/self.dt) == 0:
      self.ts.append(current_time)
      self.Xs.append(self._currentX)

  def simulate(self, time, verbose = False):
    for _ in tqdm(range(int(time/self.dt)+1), disable=not verbose):
      self._step()