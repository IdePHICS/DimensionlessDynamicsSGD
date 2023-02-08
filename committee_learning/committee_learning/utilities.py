import matplotlib.pyplot as plt
import seaborn
from matplotlib.colors import PowerNorm
import numpy as np

## Plotting
class plot_style():
  def __init__(self, fontsize=12, latex = True, style='seaborn'):
    # Some check on MatplotLib import
    self.fontsize = fontsize
    self.latex = latex
    self.style = style
  def __enter__(self):
    if self.latex:
      plt.rc('text', usetex=True)
      plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{siunitx} \usepackage{amsfonts}')
    plt.rc('font', size=self.fontsize)
    plt.style.use(self.style)

    TK_SILENCE_DEPRECATION=1
  
  def __exit__(self, *args):
    plt.close('all')
    plt.rcdefaults()
    plt.style.use('default')

    TK_SILENCE_DEPRECATION=0

def macroscopic_variable_plot(MV,figsize=(2.75,2.75),file=None):
  with plot_style():
    fig, ax = plt.subplots(figsize=figsize)
    seaborn.heatmap(
      abs(np.array(MV)),
      cmap='seismic',
      norm=PowerNorm(1.8,vmin=0.,vmax=1.),
      ax=ax
    )
    if file is not None:
      fig.savefig(file, bbox_inches = 'tight')
    plt.show()

## Algorithms
def upper_bound(value, array):
    """
    Implementation of binary search that finds the upper bound of value in array.
    """
    if value < array[0]:
      raise ValueError(f'{value} is smaller than the smmalest value in the array {array[0]}')
    if value > array[-1]:
      raise ValueError(f'{value} is larger than the largest value in the array {array[-1]}')

    if value == array[0]:
      return 0

    a = 0
    b = len(array)-1
    while a+1 != b:
      m = (a+b)//2
      if array[m] < value:
        a = m
      else:
        b = m
    print(m)
    return m