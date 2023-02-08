# distutils: language = c++
# distutils: sources = committee_learning/ode/erf_integrals.cpp

import numpy as np
cimport numpy as np
cimport cython
np.import_array() # Cython docs says I've always to use this with NumPy
from libc.math cimport sqrt, exp, erf, acos, asin

from .._config.python import DTYPE
from .._config.cython cimport *


cdef extern from 'erf_integrals.cpp':
  cdef DTYPE_t square(DTYPE_t x)

cdef extern from 'erf_integrals.cpp' namespace 'committee_learning::erfode':
  cdef inline DTYPE_t I2_noise(DTYPE_t C11, DTYPE_t C12, DTYPE_t C22)
  cdef inline DTYPE_t I3(DTYPE_t C11, DTYPE_t C12, DTYPE_t C13, DTYPE_t C22, DTYPE_t C23, DTYPE_t C33)
  cdef inline DTYPE_t I4(DTYPE_t C11, DTYPE_t C12, DTYPE_t C13, DTYPE_t C14, DTYPE_t C22, DTYPE_t C23, DTYPE_t C24, DTYPE_t C33, DTYPE_t C34, DTYPE_t C44)
  cdef inline DTYPE_t I3_C13expectation(DTYPE_t C11, DTYPE_t C12, DTYPE_t C13_offset, DTYPE_t C13_Znorm, DTYPE_t C22, DTYPE_t C23, DTYPE_t C33, unsigned int d)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef erf_updates(np.ndarray[DTYPE_t, ndim=2] Q, np.ndarray[DTYPE_t, ndim=2] M, np.ndarray[DTYPE_t, ndim=2] P, int noise_term, DTYPE_t gamma_over_p, DTYPE_t noise, int quadratic_terms):
  cdef np.ndarray[DTYPE_t, ndim=2] dQ = np.zeros_like(Q, dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=2] dM = np.zeros_like(M, dtype=DTYPE)

  cdef int p = Q.shape[0]
  cdef int k = P.shape[0]
  cdef DTYPE_t one_over_p = 1./p
  cdef DTYPE_t one_over_k = 1./k

  # Indexs for the for cycles (needed to have pure C for-loops)
  cdef int j,l,r,o,q,m,s

  ## Compute dM
  # Fixed the pair (j,r)
  for j in range(p):
    for r in range(0,k):

      # Student
      for l in range(0,p):
        dM[j,r] -= one_over_p * I3(Q[j,j], M[j,r], Q[j,l], P[r,r], M[l,r], Q[l,l])

      # Teacher
      for s in range(0,k):
        dM[j, r] += one_over_k * I3(Q[j,j], M[j,r], M[j,s], P[r,r], P[r,s], P[s,s])

  ### Compute dQ below-diag
  for j in range(0,p):
    for l in range(0,j+1): # We are using the fact that Q is symmetric, so we compute only the below diagonal
      ## I3 contribution
      for m in range(0,p):
        dQ[j,l] -= one_over_p * (
          I3(Q[j,j], Q[j,l], Q[j,m], Q[l,l], Q[l,m], Q[m,m]) + # student-student (jl) 
          I3(Q[l,l], Q[l,j], Q[l,m], Q[j,j], Q[j,m], Q[m,m])   # student-student (lj)
        ) 

      for r in range(0,k):
        dQ[j,l] += one_over_k * (
          I3(Q[j,j], Q[j,l], M[j,r], Q[l,l], M[l,r], P[r,r]) + # student-teacher (jl)
          I3(Q[l,l], Q[l,j], M[l,r], Q[j,j], M[j,r], P[r,r])   # student-teacher (lj)
        )

      ## Noise term
      if noise_term != 0:
        dQ[j,l] += noise * gamma_over_p * I2_noise(Q[j,j], Q[j,l], Q[l,l])
      
      ## I4 contribution
      if quadratic_terms != 0:
        # Student-student
        for o in range(0,p):
          for q in range(0,p):
            dQ[j,l] += gamma_over_p * square(one_over_p) * I4(Q[j,j], Q[j,l], Q[j,o], Q[j,q], Q[l,l], Q[l,o], Q[l,q], Q[o,o], Q[o,q], Q[q,q])

        # Student-teacher
        for o in range(0,p):
          for r in range(0,k):
            dQ[j,l] -= 2 * gamma_over_p * one_over_p*one_over_k * I4(Q[j,j], Q[j,l], Q[j,o], M[j,r], Q[l,l], Q[l,o], M[l,r], Q[o,o], M[o,r], P[r,r])

        # Teacher-Teacher
        for r in range(0,k):
          for s in range(0,k):
            dQ[j,l] += gamma_over_p * square(one_over_k) * I4(Q[j,j], Q[j,l], M[j,r], M[j,s], Q[l,l], M[l,r], M[l,s], P[r,r], P[r,s], P[s,s])
        
      ## Symmetrize Q
      if j != l:
        dQ[l,j] = dQ[j,l]      

  return dQ, dM



#################
# Large p limit #
#################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef largep_erf_updates_onlydiag(np.ndarray[DTYPE_t, ndim=1] Qorth, np.ndarray[DTYPE_t, ndim=2] M, np.ndarray[DTYPE_t, ndim=2] P,  int noise_term, DTYPE_t noise_gamma_over_p):
  cdef np.ndarray[DTYPE_t, ndim=1] dQorth = np.zeros_like(Qorth, dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=2] dM = np.zeros_like(M, dtype=DTYPE)
  
  cdef np.ndarray[DTYPE_t, ndim=2] Q = np.matmul(M, np.transpose(M)) + np.diag(Qorth)
  cdef int p = Qorth.shape[0]
  cdef int k = P.shape[0]
  cdef DTYPE_t one_over_p = 1./p
  cdef DTYPE_t one_over_k = 1./k

  cdef DTYPE_t student_term, teacher_term

  # Indexs for the for cycles (needed to have pure C for-loops)
  cdef int j,l,r,s

  for j in range(p):
    ## Compute dM
    for r in range(0,k):
      # Student
      for l in range(0,p):
        dM[j,r] -= one_over_p * I3(Q[j,j], M[j,r], Q[j,l], P[r,r], M[l,r], Q[l,l])

      # Teacher
      for s in range(0,k):
        dM[j,r] += one_over_k * I3(Q[j,j], M[j,r], M[j,s], P[r,r], P[r,s], P[s,s])
    
    ## Compute dQorth
    teacher_term = 0.
    student_term = 0.
    for r in range(0,k):
      teacher_term += I3(Q[j,j], Qorth[j], M[j,r], Qorth[j], 0., P[r,r])

    for l in range(0,p):
      student_term += I3(Q[j,j], Qorth[j], Q[j,l], Qorth[j], 0., Q[l,l])

    dQorth[j] += 2. * (teacher_term*one_over_k - student_term*one_over_p) 

    ## Noise term
    if noise_term:
      dQorth[j] += noise_gamma_over_p * square(one_over_p) * I2_noise(Q[j,j], Q[j,j], Q[j,j])    

  return dQorth, dM





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef largep_erf_updates_offdiag(np.ndarray[DTYPE_t, ndim=1] Qorth, np.ndarray[DTYPE_t, ndim=2] M, np.ndarray[DTYPE_t, ndim=2] P,  int noise_term, DTYPE_t noise_gamma_over_p, unsigned int Z_dimension):
  cdef np.ndarray[DTYPE_t, ndim=1] dQorth = np.zeros_like(Qorth, dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=2] dM = np.zeros_like(M, dtype=DTYPE)
  
  cdef np.ndarray[DTYPE_t, ndim=2] MMt = np.matmul(M, np.transpose(M))
  cdef np.ndarray[DTYPE_t, ndim=2] sqrt_QoQo = np.sqrt(np.einsum('j,l->jl', Qorth, Qorth))
  cdef np.ndarray[DTYPE_t, ndim=1] Qdiag = np.diag(MMt) + Qorth
  cdef int p = Qorth.shape[0]
  cdef int k = P.shape[0]
  cdef DTYPE_t one_over_p = 1./p
  cdef DTYPE_t one_over_k = 1./k

  cdef DTYPE_t C11,C12,C13,C22,C23,C33, L3, student_term, teacher_term

  # Indexs for the for cycles (needed to have pure C for-loops)
  cdef int j,l,r,s

  for j in range(p):
    ## Compute dM
    for r in range(0,k):
      # Student
      for l in range(0,p):
        dM[j,r] -= one_over_p * I3_C13expectation(Qdiag[j], M[j,r], MMt[j,l], sqrt_QoQo[j,l], P[r,r], M[l,r], Qdiag[l], Z_dimension)

      # Teacher
      for s in range(0,k):
        dM[j, r] += one_over_k * I3(Qdiag[j], M[j,r], M[j,s], P[r,r], P[r,s], P[s,s])
    
    ## Compute dQorth
    teacher_term = 0.
    student_term = 0.
    for r in range(0,k):
      teacher_term += I3(Qdiag[j], Qorth[j], M[j,r], Qorth[j], 0., P[r,r])

    for l in range(0,p):
      student_term += I3_C13expectation(Qdiag[j], Qorth[j], MMt[j,l], sqrt_QoQo[j,l], Qorth[j], 0., Qdiag[l], Z_dimension)

    dQorth[j] += 2. * (teacher_term*one_over_k - student_term*one_over_p)

    ## Noise term
    if noise_term:
      dQorth[j] += noise_gamma_over_p * square(one_over_p) * I2_noise(Qdiag[j], Qdiag[j], Qdiag[j])
       

  return dQorth, dM
