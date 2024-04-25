# FIRST PAPER (Keener, 1993)

from common import (
  Csv, PrefMethod, make_pref_matrix, 
  power_method, power_method_nonlinear, sort_by_rank,
  disp_ranks
)

import numpy as np
from matplotlib import colormaps as cmaps
import matplotlib.pyplot as plt

def main():
  csv = Csv('../data/games.csv')

  # config
  pref_method = PrefMethod.NONLINEAR
  power_iterations = 100

  # calculate ranking
  teams, R, M, A, E = make_pref_matrix(csv, pref_method)
  r = power_method(A, power_iterations)
  r_nonlinear = power_method_nonlinear(
    E, 
    M, 
    f = lambda x: (.05*x + x*x) / (2 + .05*x + x*x),
    num_iterations = power_iterations)
  
  team_ranks_linear = sort_by_rank(teams, A @ r)
  team_ranks_nonlinear = sort_by_rank(teams, A @ r_nonlinear)

  fig, ax = plt.subplots()
  ax.set_title('Preference Matrix')
  im = ax.imshow(R, cmap=cmaps['bone'])
  fig.colorbar(im, ax=ax)
  disp_ranks(team_ranks_linear, title='Eigenvector Ranking (Keener 1993)')
  disp_ranks(team_ranks_nonlinear, title='Nonlinear Eigenvector Ranking (Keener 1993)')
  plt.show()

  for team, rank in team_ranks[:20]:
    print(f'{team}: {rank}')

if __name__ == '__main__':
  main()