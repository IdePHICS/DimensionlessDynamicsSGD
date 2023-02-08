import yaml
import os

class BaseResult():
  """"
  This is the base abstract class for the results.
  """
  def __init__(self, initial_condition = None, id = 0, **kattributes):
    self.initial_condition = initial_condition
    self.id = id

    for attr, val in kattributes.items():
      setattr(self, attr, val)

  def from_file(self, filename=None, path = ''):
    if filename is None:
      filename = self.get_initial_condition_id()

    with open(path+filename+'.yaml', 'r') as file:
      data = yaml.safe_load(file)
      for att, val in data.items():
        setattr(self, att, val)

  def to_file(self, filename=None, path = ''):
    if filename is None:
      filename = self.get_initial_condition_id()

    data = {}
    for att, val in self.__dict__.items():
      if not att.startswith('__') and not callable(val):
        data[att] = val
    full_path_filename = path+filename+'.yaml'
    os.makedirs(path, exist_ok=True)
    with open(full_path_filename, 'w') as file:
      yaml.dump(data, file)
