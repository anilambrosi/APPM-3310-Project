# SECOND PAPER (Jech, 1983)

from common import (
  Csv, PrefMethod, make_pref_matrix, 
  make_rank_matrix, fixed_point_approx, sort_by_rank,
  disp_ranks
)

import numpy as np
from matplotlib import colormaps as cmaps
import matplotlib.pyplot as plt

def main():
  csv = Csv('../data/games.csv')

  #config 
  pref_method = PrefMethod.DISCRETE
  fixed_iterations = 100

  # calculate ranking
  teams, R, M, _, _ = make_pref_matrix(csv, pref_method)

  v = fixed_point_approx(R, M, fixed_iterations)
  P = make_rank_matrix(v)
  mask = np.eye(P.shape[0]) == 0
  P_adj = np.multiply(P, mask)
  r = P_adj @ np.ones(P_adj.shape[1])
  r /= np.full_like(r, P_adj.shape[1] - 1)
  team_ranks = sort_by_rank(teams, r)
  
  fig, ax = plt.subplots()
  ax.set_title('Rank Matrix')
  im = ax.imshow(P, cmap=cmaps['viridis'])
  fig.colorbar(im, ax=ax)
  disp_ranks(team_ranks, title='Probability Ranking (Jech 1983)')
  plt.show()

  for team, rank in team_ranks[:20]:
    print(f'{team}: {rank}')

if __name__ == '__main__':
  main()