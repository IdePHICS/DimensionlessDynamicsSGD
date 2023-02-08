
from random import random
from ...ode.core import SquaredActivationODE, SphericalSquaredActivationODE

import numpy as np
from scipy.linalg import sqrtm

from .variance import _variance_q, _variance_m, _covariance_qm

class PhaseRetrivialSDE(SquaredActivationODE):
  def _q_variance(self):
    q = self.Q[0][0]
    m = self.M[0][0]
    rho = self.P[0][0]
    return max(_variance_q(q,m,rho,self.gamma0,self.noise), 0.)
  
  def _m_variance(self):
    q = self.Q[0][0]
    m = self.M[0][0]
    rho = self.P[0][0]
    return max(_variance_m(q,m,rho,self.gamma0,self.noise), 0.)

  def _qm_covariance(self):
    return 0.

  def __init__(self, noise, gamma, p0, q0, m0, d, dt, seed = None):
    super().__init__(
      p = 1,
      k = 1,
      noise = noise,
      gamma0 = gamma,
      P0 = np.array([[p0]]) if isinstance(p0, float) else p0,
      Q0 = np.array([[q0]]) if isinstance(p0, float) else q0,
      M0 = np.array([[m0]]) if isinstance(p0, float) else m0,
      dt = dt
    )
    self.d = float(d)
    
    if seed is not None:
      seed ^= 22031998 # this line is shuffling the seed just to ensure that I'm not using it on every generator
    self.rng = np.random.default_rng(seed)

    # print(np.sqrt(self._q_variance() * self.dt), flush=True)

  def _step_update(self):
    dQ, dM = super()._step_update()

    # if np.random.randint(1000) == 0:
    #   print(dQ/np.sqrt(self._q_variance()), np.sqrt(self._q_variance()))

    # I have to divide by sqrt(dt) because then I'm multipling by self.dt in the fit() method
    dQ += np.sqrt(self._q_variance() / (self.d*self.dt)) * self.rng.normal(size=(1,1))
    dM += np.sqrt(self._m_variance() / (self.d*self.dt)) * self.rng.normal(size=(1,1))

    return dQ, dM

class NaiveSphericalPhaseRetrivialSDE(PhaseRetrivialSDE, SphericalSquaredActivationODE):
  """
  Apparently, the ineheritance rules are working exaclty as I want! 
  I'm really suprised about this python behaviour. Be careful!

  Source: https://en.wikipedia.org/wiki/Multiple_inheritance#The_diamond_problem
  """
  def _m_variance(self):
    m = self.M[0][0]
    return max(
      368*self.gamma0**2 - 5568*self.gamma0**3 + 31104*self.gamma0**4 + 16*self.gamma0**2*self.noise - 272*self.gamma0**3*self.noise + 1920*self.gamma0**4*self.noise + 32*self.gamma0**4*self.noise**2 - 
      480*self.gamma0**2*m + 2880*self.gamma0**3*m - 24*self.gamma0**2*self.noise*m + 144*self.gamma0**3*self.noise*m - 
      256*self.gamma0**2*m**2 + 11136*self.gamma0**3*m**2 - 73728*self.gamma0**4*m**2 + 8*self.gamma0**2*self.noise*m**2 + 
      272*self.gamma0**3*self.noise*m**2 - 2496*self.gamma0**4*self.noise*m**2 + 480*self.gamma0**2*m**3 - 5760*self.gamma0**3*m**3 - 
      144*self.gamma0**3*self.noise*m**3 - 112*self.gamma0**2*m**4 - 5568*self.gamma0**3*m**4 + 54144*self.gamma0**4*m**4 + 
      576*self.gamma0**4*self.noise*m**4 + 2880*self.gamma0**3*m**5 - 11520*self.gamma0**4*m**6,
      0.
    )

  def _q_variance(self):
    return np.zeros((1,1))

class SphericalPhaseRetrivialSDE(PhaseRetrivialSDE, SphericalSquaredActivationODE):

  def _step_update(self):
    m = self.M[0][0]
    # Variance matrix
    Sigma = np.array([[self._q_variance(), self._qm_covariance()],
                      [self._qm_covariance(), self._m_variance()]]) /self.d
    # std matrix
    sigma_q, sigma_m = sqrtm(Sigma)

    stochastich_term = np.einsum(
      'i,ijk->jk',
      sigma_m - m/2 * sigma_q,
      self.rng.normal(size=(2,1,1))
    ) / np.sqrt(self.dt) # Need to divide by sqrt(dt) because then you multiply by dt

    extra_drift = 3/8 * m * np.dot(sigma_q, sigma_q) - .5 * np.dot(sigma_q, sigma_m)


    dQ, dM = super(PhaseRetrivialSDE, self)._step_update()
    dM += stochastich_term + extra_drift
    return dQ, dM

