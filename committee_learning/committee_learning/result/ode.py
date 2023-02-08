import numpy as np
import datetime
import hashlib

from .base import BaseResult
from ..ode.base import BaseFullODE, BaseLargePODE
from ..ode.square import BaseSquaredActivationODE
from ..ode.erf import BaseErfActivationODE

class BaseODEResult(BaseResult):
  def __init__(self, initial_condition = None, id = 0, **kattributes):
    super().__init__(initial_condition=initial_condition, id=id, **kattributes)

  def from_ode(self, ode):
    self.timestamp= str(datetime.datetime.now())
    self.p=int(ode.p)
    self.k=int(ode.k)
    self.dt=float(ode.dt)
    self.P=np.array(ode.P).tolist()
    self.simulated_time=float(ode._simulated_time)
    self.save_per_decade=int(len(ode.saved_times)/np.log10(ode._simulated_time/ode.dt)) if len(ode.saved_times)>0 else None
    self.times=np.array(ode.saved_times).tolist()
    self.risks=np.array(ode.saved_risks).tolist()
    self.Ms=np.array(ode.saved_Ms).tolist()
    if issubclass(type(ode), BaseSquaredActivationODE):
      self.activation = 'squared'
    elif issubclass(type(ode), BaseErfActivationODE):
      self.activation = 'erf'
    else:
      raise TypeError('Unrecognized activation function for type '+ode.__class__.__name__)
    self.noise_term = ode.noise_term

  def from_file_or_run(self, ode, decades, save_per_decade = 100, path='',show_progress=True, force_run=False, force_read=False):
    self.from_ode(ode)
    self.simulated_time = float(10**decades)
    try:
      if force_run:
        raise FileNotFoundError
      self.from_file(path=path)
    except FileNotFoundError as file_error:
      if force_read:
        raise file_error
      ode.fit_logscale(decades, save_per_decade=save_per_decade, show_progress=show_progress)
      self.from_ode(ode)
      self.to_file(path=path)


## OdeResult History
# - 0.1: first usable version
# - 0.2: BIG change -- this becomes the Result Class for all ODEs
# - 0.3: noise_term

class FullODEResult(BaseODEResult):
  def __init__(self, initial_condition = None, id = 0, **kattributes):
    super().__init__(initial_condition=initial_condition, id=id, **kattributes)

    # This is an identifier for which file version I'm using to store files
    self.version = '0.3'
  
  def from_ode(self, ode: BaseFullODE):
    super().from_ode(ode)
    self.noise=float(ode.noise)
    self.gamma_over_p=float(ode._gamma_over_p)
    self.quadratic_terms = ode.quadratic_terms
    self.Qs=np.array(ode.saved_Qs).tolist()
  
  def get_initial_condition_id(self):
    # Achtung!! Changing this function make all previous generated data unacessible!
    # Consider producing a script of conversion before apply modifications.
    ic_string = self.initial_condition
    if ic_string is None:
      ic_string = np.random.randint(int(1e9))

    datastring = '_'.join([
      str(ic_string),
      f"{self.activation}"
      f"{self.p}",
      f"{self.k}",
      f"{self.noise:.6f}",
      f"{self.quadratic_terms}",
      f"{self.gamma_over_p:.6f}",
      f"{self.simulated_time:.6f}",
      f"{self.dt:.6f}",
      f'{self.noise_term}',
      f"{self.id}"
    ])
    return hashlib.md5(datastring.encode('utf-8')).hexdigest()


## LargePErfOdeResult History
# - 0.1: first usable version
# - 0.2: BIG change -- this becomes the Result Class for all LargeP ODEs
# - 0.3: noise_term, offdiagonal
# - 0.4: added noise_gamma_over_p

class LargePODEResult(BaseODEResult):
  def __init__(self, initial_condition = None, id = 0, **kattributes):
    super().__init__(initial_condition=initial_condition, id=id, **kattributes)

    # This is an identifier for which file version I'm using to store files
    self.version = '0.4'
  
  def from_ode(self, ode: BaseLargePODE):
    super().from_ode(ode)
    self.Qorths=np.array(ode.saved_Qorths).tolist()
    self.offdiagonal = ode.offdiagonal
    self.noise_gamma_over_p=float(ode._noise_gamma_over_p) if ode._noise_gamma_over_p is not None else 'None'
  
  def get_initial_condition_id(self):
    # Achtung!! Changing this function make all previous generated data unacessible!
    # Consider producing a script of conversion before apply modifications.
    ic_string = self.initial_condition
    if ic_string is None:
      ic_string = np.random.randint(int(1e9))

    datastring = '_'.join([
      str(ic_string),
      f'{self.activation}'
      f'{self.p}',
      f'{self.k}',
      f'{self.simulated_time:.6f}',
      f'{self.dt:.6f}',
      f'{self.noise_term}',
      f'{self.noise_gamma_over_p}',
      f'{self.offdiagonal}',
      f'{self.id}'
    ])
    return hashlib.md5(datastring.encode('utf-8')).hexdigest()
