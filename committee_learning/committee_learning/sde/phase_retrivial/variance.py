
def _variance_q(q,m,rho,gamma,noise):
  return (
    48*gamma**2*noise*q**2 + 128*gamma**4*noise**2*q**2 - 1344*gamma**3*noise*q**3 + 
    1536*gamma**2*q**4 + 9600*gamma**4*noise*q**4 - 28800*gamma**3*q**5 + 
    162720*gamma**4*q**6 + 1088*gamma**3*noise*q*m**2 - 2688*gamma**2*q**2*m**2 - 
    16512*gamma**4*noise*q**2*m**2 + 77376*gamma**3*q**3*m**2 - 
    593280*gamma**4*q**4*m**2 + 320*gamma**2*m**4 + 2304*gamma**4*noise*m**4 - 
    33024*gamma**3*q*m**4 + 474624*gamma**4*q**2*m**4 - 46080*gamma**4*m**6 + 
    256*gamma**3*noise*q**2*rho - 384*gamma**2*q**3*rho - 2688*gamma**4*noise*q**3*rho + 
    9024*gamma**3*q**4*rho - 57600*gamma**4*q**5*rho + 
    1088*gamma**2*q*m**2*rho + 6528*gamma**4*noise*q*m**2*rho - 
    49536*gamma**3*q**2*m**2*rho + 473472*gamma**4*q**3*m**2*rho + 
    10752*gamma**3*m**4*rho - 336384*gamma**4*q*m**4*rho + 
    128*gamma**2*q**2*rho**2 + 768*gamma**4*noise*q**2*rho**2 - 
    3840*gamma**3*q**3*rho**2 + 28224*gamma**4*q**4*rho**2 + 
    16704*gamma**3*q*m**2*rho**2 - 254592*gamma**4*q**2*m**2*rho**2 + 
    78336*gamma**4*m**4*rho**2 + 1344*gamma**3*q**2*rho**3 - 
    13824*gamma**4*q**3*rho**3 + 79488*gamma**4*q*m**2*rho**3 + 
    4896*gamma**4*q**2*rho**4
  )

def _variance_m(q,m,rho,gamma,noise):
  return (
    8*gamma**2*noise*m**2 + 324*gamma**2*q**2*m**2 - 192*gamma**2*m**4 + 
    4*gamma**2*noise*q*rho + 60*gamma**2*q**3*rho - 504*gamma**2*q*m**2*rho - 
    72*gamma**2*q**2*rho**2 + 324*gamma**2*m**2*rho**2 + 60*gamma**2*q*rho**3
  )

def _covariance_qm(q,m,rho,gamma,noise):
  return (
    0.
  )