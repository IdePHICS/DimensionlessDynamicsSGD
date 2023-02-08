import numpy as np
import datetime
import hashlib

from ..ode import SquaredActivationODE
from .base import BaseResult
from ..utilities import upper_bound


## SquareOdeResult History
# - 0.2: first usable version
# - 0.3: fix a big bug in the M and Q saving process. All previous versions are crap
# - 0.4: now id number is part of id string. Adding also the separator


class SquareODEResult(BaseResult):
  def __init__(self, initial_condition = None, id = 0, **kattributes):
    super().__init__(initial_condition=initial_condition, id=id, **kattributes)

    # This is an identifier for which file version I'm using to store files
    self.version = '0.3'
  
  def from_ode(self, ode: SquaredActivationODE):
    self.timestamp= str(datetime.datetime.now())
    self.p=int(ode.p)
    self.k=int(ode.k)
    self.noise=float(ode.noise)
    self.gamma=float(ode.gamma)
    self.dt=float(ode.dt)
    self.P=np.array(ode.P).tolist()
    self.simulated_time=float(ode._simulated_time)
    self.save_per_decade=int(len(ode.saved_times)/np.log10(ode._simulated_time/ode.dt)) if len(ode.saved_times)>0 else None
    self.times=np.array(ode.saved_times).tolist()
    self.risks=np.array(ode.saved_risks).tolist()
    self.Ms=np.array(ode.saved_Ms).tolist()
    self.Qs=np.array(ode.saved_Qs).tolist()
  
  def get_initial_condition_id(self):
    # Achtung!! Changing this function make all previous generated data unacessible!
    # Consider producing a script of conversion before apply modifications.
    ic_string = self.initial_condition
    if ic_string is None:
      ic_string = np.random.randint(int(1e9))

    datastring = '_'.join([
      str(ic_string),
      f"{self.p}",
      f"{self.k}",
      f"{self.noise:.6f}",
      f"{self.gamma:.6f}",
      f"{self.simulated_time:.6f}",
      f"{self.dt:.6f}",
      f"{self.id}"
    ])
    return hashlib.md5(datastring.encode('utf-8')).hexdigest()

  def from_file_or_run(self, ode, decades, path='',show_progress=True, force_run=False):
    self.from_ode(ode)
    self.simulated_time = float(10**decades)
    try:
      if force_run:
        raise FileNotFoundError
      self.from_file(path=path)
    except FileNotFoundError:
      ode.fit_logscale(decades, show_progress=show_progress)
      self.from_ode(ode)
      self.to_file(path=path)

  def _time_to_index(self,t):
    """
    Return the smallest saved index whose corresponding saved time is larger or 
    equal than the given time.
    It is based on binary search.
    """
    return upper_bound(t, self.times)