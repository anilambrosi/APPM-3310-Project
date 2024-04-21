# code for 1st paper (perron frobenius theorem)

from common import Csv, power_method
from math import copysign, sqrt, fabs, pi
from dataclasses import dataclass
from enum import Enum
from functools import cmp_to_key
import numpy as np
from matplotlib import colormaps as cmaps
import matplotlib.pyplot as plt

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

def make_pref_matrix(csv: Csv, settings: PerronFrobeniusSettings) -> tuple[list[str], np.ndarray[np.float32]]:
  teams = list(set(line[0] for line in csv.data) | 
               set(line[2] for line in csv.data))
  num_teams = len(teams)

  # make preference matrix
  calc_a = PREF_METHODS[settings.pref_method]
  pref_matrix = np.zeros((num_teams, num_teams), dtype=np.float32)
  
  for line in csv.data:
    iname, ipts, jname, jpts = line

    ipts = int(ipts)
    jpts = int(jpts)
    i = teams.index(iname)
    j = teams.index(jname)

    pref_matrix[i, j] = calc_a(ipts, jpts)
    pref_matrix[j, i] = calc_a(jpts, ipts)

  return (teams, pref_matrix)

def main():
  csv = Csv('../data/games.csv')

  settings = PerronFrobeniusSettings(
    pref_method=PrefMethod.NONLINEAR, 
    power_iterations=10)

  teams, pref_matrix = make_pref_matrix(csv, settings)
  num_teams = len(teams)
  rank_vec = power_method(pref_matrix, settings.power_iterations)

  team_ranks = sorted(zip(teams, rank_vec), key=cmp_to_key(lambda a, b: b[1] - a[1]))
  for team, rank in team_ranks[:20]:
    print(f'{team}: {rank}')

if __name__ == '__main__':
  main()