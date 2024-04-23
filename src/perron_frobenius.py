# code for 1st paper (perron frobenius theorem)

from common import Csv, power_method, sort_by_rank
from math import copysign, sqrt, fabs, pi
from dataclasses import dataclass
from enum import Enum
from collections.abc import Callable
import numpy as np
from matplotlib import colormaps as cmaps
import matplotlib.pyplot as plt

type mat_f32 = np.ndarray[np.float32]

class PrefMethod(Enum): 
  DISCRETE = 'discrete'
  INTERPOLATE = 'interpoate'
  INTERPOLATE_OFFSET = 'interpolate_offset'
  NONLINEAR = 'nonlinear'

@dataclass
class PerronFrobeniusSettings:
  pref_method: PrefMethod
  power_iterations: int

calc_pref_interpolate = lambda si, sj: si / (si + sj)
calc_pref_interpolate_offset = lambda si, sj: calc_pref_interpolate(si + 1, sj + 1)

def calc_pref_discrete(si: int, sj: int) -> float:
  if si == sj:
    return 0.5
  elif si > sj:
    return 1
  return 0

def calc_pref_nonlinear(si: int, sj: int) -> float:
  x = calc_pref_interpolate_offset(si, sj)
  return 0.5 + 0.5 * copysign(1, x - 0.5) * sqrt(fabs(2 * x - 1))

PREF_METHODS = {
  PrefMethod.DISCRETE: calc_pref_discrete,
  PrefMethod.INTERPOLATE: calc_pref_interpolate,
  PrefMethod.INTERPOLATE_OFFSET: calc_pref_interpolate_offset,
  PrefMethod.NONLINEAR: calc_pref_nonlinear
}

def make_pref_matrix(csv: Csv, settings: PerronFrobeniusSettings) -> tuple[list[str], mat_f32, mat_f32, mat_f32, mat_f32]:
  
  teams = list(set(line[0] for line in csv.data) | 
               set(line[2] for line in csv.data))
  num_teams = len(teams)

  # make preference matrix
  calc_a = PREF_METHODS[settings.pref_method]
  pref_matrix = np.zeros((num_teams, num_teams), dtype=np.float32)
  games_matrix = np.zeros((num_teams, num_teams), dtype=np.float32)
  pref_matrix_nonlinear = np.zeros((num_teams, num_teams), dtype=np.float32)
  calc_e = lambda si, sj: (5 + si + si**(2/3)) / (5 + sj + si**(2/3))

  for line in csv.data:
    iname, ipts, jname, jpts = line

    ipts = int(ipts)
    jpts = int(jpts)
    i = teams.index(iname)
    j = teams.index(jname)

    pref_matrix[i, j] += calc_a(ipts, jpts)
    pref_matrix[j, i] += calc_a(jpts, ipts)
    
    pref_matrix_nonlinear[i, j] += calc_e(ipts, jpts)
    pref_matrix_nonlinear[j ,i] += calc_e(jpts, ipts)
    
    games_matrix[i, j] += 1
    games_matrix[j, i] += 1

  norm_pref_matrix = np.copy(pref_matrix)
  games_played = games_matrix @ np.ones(games_matrix.shape[1])
  for i, _ in enumerate(teams):
    norm_pref_matrix[i, :] /= games_played[i]

  return (teams, pref_matrix, games_matrix, norm_pref_matrix, pref_matrix_nonlinear)

def power_method_nonlinear(E: mat_f32, games_matrix: mat_f32, f: Callable[[float], float], num_iterations: int) -> mat_f32:
  r = np.random.rand(E.shape[1])
  f_vec = np.vectorize(f)
  games_played = games_matrix @ np.ones(games_matrix.shape[1])
  for _ in range(num_iterations):
    r = np.divide(f_vec(E @ r), games_played)
  
  return r

def main():
  csv = Csv('../data/games.csv')

  settings = PerronFrobeniusSettings(
    pref_method=PrefMethod.NONLINEAR, 
    power_iterations=10)

  # calculate ranking
  teams, pref_matrix, games_matrix, norm_pref_matrix, pref_matrix_nonlinear = make_pref_matrix(csv, settings)
  rank_vec = power_method(norm_pref_matrix, settings.power_iterations)
  rank_vec_nonlinear = power_method_nonlinear(
    E=pref_matrix_nonlinear, 
    games_matrix=games_matrix, 
    f=lambda x: (.05*x + x*x) / (2 + .05*x + x*x),
    num_iterations=10)
  score_vec = norm_pref_matrix @ rank_vec_nonlinear
  team_ranks = sort_by_rank(teams, score_vec)

  plt.imshow(pref_matrix, cmap=cmaps['viridis'])
  plt.colorbar()
  plt.show()

  for team, rank in team_ranks[:20]:
    print(f'{team}: {rank}')

if __name__ == '__main__':
  main()