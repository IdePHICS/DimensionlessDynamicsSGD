import numpy as np
from sklearn.preprocessing import normalize
from scipy.linalg import orth

from ._config.python import scalar_type

class MockIntialCondition():
  """
  Mock class for initial conditon to be used when loading from file.
  ACHTUNG: don't use as actual initial condition because everything would be 0.
  """
  def __init__(self, p, k, d):
    self.W0 = np.zeros((p,d))
    self.Wt = np.zeros((p,d))
    self.Q = np.zeros((p,p))
    self.M = np.zeros((p,k))
    self.P = np.zeros((k,k))

class PhaseDiagramInitialConditions():
  """
  These class is used for building initial conditions using appendix D of Rodrigo's paper.

  opposite_norm change makes the columns of M matrix to be on unit sphere (instead of rows)

  """
  def __init__(self, p, k, opposite_norm = False, seed = None):
    self.p = p
    self.k = k
    self.rng = np.random.default_rng(seed)
    normalization_axis = 0 if opposite_norm else 1
    self.M = normalize(self.rng.normal(size=(p, k)), axis=normalization_axis, norm='l2')
    self.Q = self.M @ self.M.T
    self.P = np.eye(k)
  
  def weights(self, d, seed = None):
    assert(d >= self.k) # If it is not we can't have P = Id, so this conditctions are impossible
    self.rng = np.random.default_rng(seed)
    Wt = np.sqrt(d)*orth((normalize(self.rng.standard_normal((self.k, d)), axis=1, norm='l2')).T).T
    W0 = self.M @ Wt
    return Wt, W0

  def simulation_weights(self, d_list):
    for d in d_list:
      yield (d,)+self.weights(d)

class RandomNormalInitialConditions():

  def __init__(self, p, k, d, sigma=1., seed = None, spherical = False, orth_teacher = False):
    self.p = p
    self.k = k
    self.d = d

    rng = np.random.default_rng(seed)
    self.Wt = rng.normal(size=(k, d))
    self.W0 = rng.normal(size=(p, d), scale = sigma)
    if spherical:
      "Force the weights vectors to be on the sphere of radius sqrt(d)"
      self.Wt = np.sqrt(d) * normalize(self.Wt, axis = 1)
      self.W0 = np.sqrt(d) * normalize(self.W0, axis = 1)

    if orth_teacher:
      self.Wt = np.sqrt(d)*orth((normalize(self.Wt, axis=1, norm='l2')).T).T

    self.P = self.Wt @ self.Wt.T / scalar_type(d)
    self.M = self.W0 @ self.Wt.T / scalar_type(d)
    self.Q = self.W0 @ self.W0.T / scalar_type(d)


class SymmetricInitialConditions():
  def __init__(self, p, k, epsilon, q0, seed = None):
    assert(0.<=epsilon and epsilon <=1.)
    assert(q0 > 0.)

    self.p = p
    self.k = k
    self.epsilon = scalar_type(epsilon)
    self.q0 = scalar_type(q0)

    self.P = np.eye(k, dtype=scalar_type)
    self.M = scalar_type(epsilon/k) * np.ones((p,k), dtype=scalar_type)
    self.Q = scalar_type((self.epsilon)**2/k) * np.ones((p,p), dtype=scalar_type) + scalar_type(q0*(1-epsilon)**2) * np.eye(p, dtype=scalar_type)

    self.rng = np.random.default_rng(seed)
  
  def _Wmatrices(self, d):
    """
    Take p+k random vectors in R^d, orthonormalize and use the first p of them
    for wS, and the others for wT.
    """
    W_rand = self.rng.normal(size=(self.p+self.k, d))
    W_orth = orth(W_rand.T).T
    # print(np.sum(W_orth[0]*W_orth[7])) # This line checks the normalization
    WS = np.sqrt(self.q0) * scalar_type(np.sqrt(d)) * W_orth[:self.p]
    WT = scalar_type(np.sqrt(d)) * W_orth[self.p:]

    wt_overlap = scalar_type(1./self.k) * np.sum(WT, axis=0)
    WT_overlap = np.tile(wt_overlap,(self.p,1))

    return WT, WS, WT_overlap

  def weights(self, d):
    WT, WS, WT_overlap = self._Wmatrices(d)
    W0 = self.epsilon*WT_overlap + (scalar_type(1.)-self.epsilon) * WS
    return WT, W0

  def simulations(self, d_list):
    for d in d_list:
      yield (d,)+self.weights(d)

class AlmostSymmetricInitialConditions(SymmetricInitialConditions):
  """
  These implementation is not useful... I have to restate the initial consitions.
  """
  def __init__(self, p, k, epsilon, delta, q0):
    assert(epsilon-delta>=0.)
    assert(epsilon+delta<=1.)
    assert(p % 2 == 0)

    super().__init__(p,k,epsilon,q0)
    self.delta = scalar_type(delta)

    half_eye = np.ones((p//2,p//2))
    half_ide = np.eye(p//2)
    half_zer = np.zeros((p//2,p//2))
    self.K = np.block([
      [-half_ide, half_zer],
      [ half_zer, half_ide]
    ])
    H = np.block([
      [half_eye, half_zer],
      [half_zer, half_eye]
    ])
    G = np.block([
      [half_zer, half_eye],
      [half_eye, half_zer]
    ])

    self.M += scalar_type(1./k) * self.delta * self.K @ np.ones((p,k))
    self.Q += scalar_type(1./k) * (self.delta**2 * (H-G) + 2 * self.epsilon*self.delta * self.K @ H)

  def weights(self, d):
    WT, WS, WT_overlap = self._Wmatrices(d)
    W0 = (self.epsilon * np.eye(self.p) + self.delta * self.K) @ WT_overlap + (scalar_type(1.)-self.epsilon) * WS
    return WT, W0