import numpy as np
from tqdm import tqdm
import math

from .._config.python import scalar_type
from .base import BaseODE, BaseFullODE, BaseLargePODE

class BaseSquaredActivationODE(BaseODE):
  def __init__(self, P0, Q0, M0, dt, noise_term):
    super().__init__(P0, Q0, M0, dt, noise_term)
    # Precomputation
    self._1_k = scalar_type(1./self.k)
    self._1_p = scalar_type(1./self.p)
    self._1_pk = scalar_type(1./(self.k*self.p))
    self._1_kk = scalar_type(1./(self.k**2))
    self._1_pp = scalar_type(1./(self.p**2))

    self.trP = np.trace(self.P)
    self.PP = self.P @ self.P
    self.trPP = np.trace(self.PP)

  def risk(self):
    trQ = np.trace(self.Q)
    trQQ = np.trace(self.Q @ self.Q)
    trMMt = np.trace(self.M @ self.M.T)
    two = scalar_type(2)
    Rt = self._1_kk * (self.trP**2 + two*self.trPP)
    Rs = self._1_pp * (trQ**2 + two*trQQ)
    Rst = -two * self._1_pk * (trQ*self.trP + two*trMMt)
    return (Rt + Rs + Rst)/two

class SquaredActivationODE(BaseFullODE, BaseSquaredActivationODE):
  """
  NB: the time scaling used is t = nu * (gamma/(pd)).
  This is different from what I did in Master Internship (where t = nu / d),
  but this is the best choice to use this equations in general.
  """
  # def __init__(self, P0, Q0, M0, dt, quadratic_terms = False, gamma_over_p = 0., noise = 0.):
  #   super().__init__(P0, Q0, M0, dt, quadratic_terms, gamma_over_p, noise)
  #   super().__init__()


  def _compute_Phi_Psi(self):
    """
    This method just compute the RHS of the ODEs:
    dQ/dt = Phi
    dM/dt = Psi
    """
    # Compute all the stuff
    ik = self._1_k
    ip = self._1_p
    
    two = scalar_type(2.)
    four = scalar_type(4.)

    Q = self.Q
    M = self.M
    P = self.P
    QQ = Q @ Q
    QM = Q @ M
    MP = M @ P
    MMt = M @ M.T
    
    trQ = np.trace(Q)
    trP = self.trP
    
    ## Compute M update
    Psi = two * ((trP*ik - trQ*ip)*M + two*(MP*ik-QM*ip))

    ## Compute Q update
    Phi = np.zeros_like(Q)
    Phi += four * ((trP*ik - trQ*ip)*Q + two*(MMt*ik-QQ*ip))

    if self.quadratic_terms:
      ikk = self._1_kk
      ipp = self._1_pp
      ipk = self._1_pk
      eight = scalar_type(8.)
      # PP = self.PP
      MMtQ = MMt @ Q
      QMMt = Q @ MMt
      MPMt = MP @ M.T
      QQQ = Q @ QQ
      trMMt = np.trace(MMt)
      trQQ = np.trace(QQ)
      trPP = self.trPP

      Phitt = ikk * ((trP**2 +two*trPP)*Q + four*trP*MMt + eight*MPMt)
      Phist = -two*ipk*((trP*trQ+2*trMMt)*Q + two*trQ*MMt + two*trP*QQ + four*(MMtQ+QMMt))
      Phiss = ipp * ((trQ**2+two*trQQ)*Q + four*trQ*QQ + eight*QQQ)
      Phi += four*self._gamma_over_p*(Phitt + Phist + Phiss)

    if self.noise_term:
      Phi += four*self._gamma_over_p*self.noise*Q

    return Phi, Psi

  def _update_step(self):
    Phi, Psi = self._compute_Phi_Psi()
    self.Q += Phi * self.dt
    self.M += Psi * self.dt

class SphericalSquaredActivationODE(SquaredActivationODE):
  def _compute_Phi_Psi(self):
    # Unconstrainted updtes
    Phi, Psi = super()._compute_Phi_Psi()

    diagQ = np.diag(Phi)
    row_diagQ = np.tile(diagQ, (int(self.p),1))


    Phi_constraint = self.Q*(row_diagQ+row_diagQ.T)/scalar_type(2)
    Psi_constraint = self.M*np.tile(diagQ, (int(self.k),1)).T/scalar_type(2)

    Phi -= Phi_constraint
    Psi -= Psi_constraint
    return Phi, Psi


class LargePSquaredActivationODE(BaseLargePODE, BaseSquaredActivationODE):
  """
  P0 must be the identity matrix!
  """

  def _compute_Psi_Gamma(self):
    diagQ = np.einsum('jr,jr->j', self.M, self.M) + self.Qorth # Q = MMt + Qorth
    trQ = np.sum(diagQ)
    QM = np.linalg.multi_dot([self.M, self.M.T, self.M]) + np.einsum('j,jr->jr', self.Qorth, self.M) # QM = MMtM + QorthM

    Psi = 2*(1. - trQ*self._1_p) * self.M + 4 * (self.M*self._1_k - QM *self._1_p)
    Gamma = 4*(1. - trQ*self._1_p) * self.Qorth - 8*self._1_p * diagQ * self.Qorth

    # Offdiagonal correction
    if self.offdiagonal:
      QQorth_correction = self.Qorth*(np.sum(self.Qorth)-self.Qorth)/(self.d-self.k)
      Gamma += -8*self._1_p * QQorth_correction

    # Noise term
    if self.noise_term:
      Gamma += 4*self._noise_gamma_over_p * diagQ

    return Psi, Gamma

  def risk(self):
    """
    Need to rempliment the risk because the original BaseSquaredActivationODE uses self.Q that is extremly inefficcent.
    """
    two = scalar_type(2)
    MMt = self.M @ self.M.T
    trMMt = np.trace(MMt)
    trQ = trMMt + np.sum(self.Qorth)
    trQQ = np.sum(MMt**2) + two * np.einsum('jl,j->', MMt, self.Qorth) + np.sum(self.Qorth**2) # Q**2 = MMtMMt + 2MMtQorth + Qorth**2
    Rt = self._1_kk * (self.trP**2 + two*self.trPP)
    Rs = self._1_pp * (trQ**2 + two*trQQ)
    if self.offdiagonal:
      # This is the correction to trQQ, the only term where a Qorth**2 appears
      Rs += self._1_pp * two*np.sum(np.einsum('i,j->ij', self.Qorth, self.Qorth)-np.diag(self.Qorth)**2)/(self.d-self.k)
    Rst = -two * self._1_pk * (trQ*self.trP + two*trMMt)
    return (Rt + Rs + Rst)/two

  def _update_step(self):
    Psi, Gamma = self._compute_Psi_Gamma()
    self.Qorth += Gamma * self.dt
    self.M += Psi * self.dt



