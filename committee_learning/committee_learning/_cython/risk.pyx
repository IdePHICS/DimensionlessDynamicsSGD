# distutils: language = c++
# distutils: sources = committee_learning/ode/erf_integrals.cpp

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, exp, erf, acos, asin


from .._config.cython cimport *


def square_risk(Q, M, P):
  p = Q.shape[0]
  k = P.shape[0]

  trP = np.trace(P)
  trPP = np.trace(P @ P)
  trQ = np.trace(Q)
  trQQ = np.trace(Q @ Q)
  trMMt = np.trace(M @ M.T)

  Rt = 1./square(k) * (square(trP) + 2.*trPP)
  Rs = 1./square(p) * (square(trQ) + 2.*trQQ)
  Rst = -2./(p*k) * (trQ*trP + 2.*trMMt)

  return float(Rt + Rs + Rst)/2.



cdef extern from '../ode/erf_integrals.cpp':
  cdef DTYPE_t square(DTYPE_t x)

cdef extern from '../ode/erf_integrals.cpp' namespace 'committee_learning::erfode':
  cdef inline DTYPE_t I2(DTYPE_t C11, DTYPE_t C12, DTYPE_t C22)
  cdef inline DTYPE_t I2_C12expectation(DTYPE_t C11, DTYPE_t C12_offset, DTYPE_t C12_Znorm, DTYPE_t C22, unsigned int Z_dimension)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef erf_risk(np.ndarray[DTYPE_t, ndim=2] Q, np.ndarray[DTYPE_t, ndim=2] M, np.ndarray[DTYPE_t, ndim=2] P):
  cdef DTYPE_t risk = 0.

  cdef int p = Q.shape[0]
  cdef int k = P.shape[0]
  cdef DTYPE_t one_over_p = 1./p
  cdef DTYPE_t one_over_k = 1./k

  cdef int j,l

  # Teacher-Teacher
  for j in range(0,k):
    for l in range(0,k):
      risk += square(one_over_k) * I2(P[j,j], P[j,l], P[l,l])
  # Teacher-Student
  for j in range(0,p):
    for l in range(0,k):
      risk -= 2*one_over_p*one_over_k * I2(Q[j,j], M[j,l], P[l,l])
  # Student-Student
  for j in range(0,p):
    for l in range(0,p):
      risk += square(one_over_p) * I2(Q[j,j], Q[j,l], Q[l,l])

  return risk


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def erf_risk_Zexpectation(np.ndarray[DTYPE_t, ndim=1] Qorth, np.ndarray[DTYPE_t, ndim=2] M, np.ndarray[DTYPE_t, ndim=2] P, unsigned int Z_dimension):
  cdef DTYPE_t risk = 0.

  cdef int p = Qorth.shape[0]
  cdef int k = P.shape[0]
  cdef DTYPE_t one_over_p = 1./p
  cdef DTYPE_t one_over_k = 1./k

  cdef np.ndarray[DTYPE_t, ndim=2] MMt = np.matmul(M, np.transpose(M))
  cdef np.ndarray[DTYPE_t, ndim=2] sqrt_QoQo = np.sqrt(np.einsum('j,l->jl', Qorth, Qorth))
  cdef np.ndarray[DTYPE_t, ndim=1] Qdiag = np.diag(MMt) + Qorth

  cdef int j,l

  # Teacher-Teacher
  for j in range(0,k):
    for l in range(0,k):
      risk += square(one_over_k) * I2(P[j,j], P[j,l], P[l,l])
  # Teacher-Student
  for j in range(0,p):
    for l in range(0,k):
      risk -= 2*one_over_p*one_over_k * I2(Qdiag[j], M[j,l], P[l,l])
  # Student-Student
  for j in range(0,p):
    for l in range(0,p):
      risk += square(one_over_p) * I2_C12expectation(Qdiag[j], MMt[j,l], sqrt_QoQo[j,l], Qdiag[l], Z_dimension)

  return risk