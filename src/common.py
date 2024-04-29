# utils for cleaning/loading data

import re
import numpy as np
import matplotlib.pyplot as plt
from csv import reader as csv_reader
from io import StringIO
from functools import cmp_to_key
from math import copysign, sqrt, fabs, pi
from enum import Enum
from collections.abc import Callable

"""
naming conventions:
  M => schedule matrix
  R => result matrix
  A => preference matrix
  E => (nonlinear) preference matrix
  P => rank matrix described in Jech, 1983
  r => rank vector
  s => score vector
  T => nonlinear transformation described in Jech, 1983
  v => probability vector described in Jech, 1983
  V => matrix whose column space spans the probability vector space described in Jech, 1983
"""

# TYPES

type mat_f32 = np.ndarray[np.float32]
type vec_f32 = np.ndarray[np.float32]

# CSV

class Csv:
  def __init__(self, path: str):
    with open(path, mode='r') as file:
      text = file.read() # remove seed from team names
      clean_text = re.sub(r'\([0-9]+\)\s*', '', text)
      csv_file = csv_reader(StringIO(clean_text))
      next(csv_file) # remove header
      self.data = [line for line in csv_file if all(entry != '' for entry in line)]

# PREFERENCE / RESULT MATRICES

calc_pref_interpolate = lambda si, sj: si / (si + sj) # find a_{ij}
calc_pref_interpolate_offset = lambda si, sj: calc_pref_interpolate(si + 1, sj + 1) # find a_{ij}

def calc_pref_discrete(si: int, sj: int) -> float: # find a_{ij}
  if si == sj:
    return 0.5
  elif si > sj:
    return 1
  return 0

def calc_pref_nonlinear(si: int, sj: int) -> float: # find a_{ij}
  x = calc_pref_interpolate_offset(si, sj)
  return 0.5 + 0.5 * copysign(1, x - 0.5) * sqrt(fabs(2 * x - 1))

class PrefMethod(Enum): 
  DISCRETE = 'discrete' # DISCRETE -> result matrix (R)
  INTERPOLATE = 'interpoate'
  INTERPOLATE_OFFSET = 'interpolate_offset'
  NONLINEAR = 'nonlinear'

PREF_METHODS = {
  PrefMethod.DISCRETE: calc_pref_discrete,
  PrefMethod.INTERPOLATE: calc_pref_interpolate,
  PrefMethod.INTERPOLATE_OFFSET: calc_pref_interpolate_offset,
  PrefMethod.NONLINEAR: calc_pref_nonlinear
}

def make_pref_matrix(csv: Csv, pref_method: PrefMethod) -> tuple[list[str], mat_f32, mat_f32, mat_f32, mat_f32]:
  teams = list(set(line[0] for line in csv.data) | 
               set(line[2] for line in csv.data))
  num_teams = len(teams)

  M = np.zeros((num_teams, num_teams), dtype=np.float32) # schedule matrix
  R = np.zeros((num_teams, num_teams), dtype=np.float32) # result matrix
  E = np.zeros((num_teams, num_teams), dtype=np.float32) # nonlinear preference matrix
  
  calc_a = PREF_METHODS[pref_method] # find each A_ij
  calc_e = lambda si, sj: (5 + si + si**(2/3)) / (5 + sj + si**(2/3)) # find each E_ij

  for line in csv.data:
    iname, si, jname, sj = line

    i = teams.index(iname)
    j = teams.index(jname)
    si = int(si)
    sj = int(sj)

    M[i, j] += 1
    M[j, i] += 1
    R[i, j] += calc_a(si, sj)
    R[j, i] += calc_a(sj, si)
    E[i, j] += calc_e(si, sj)
    E[j ,i] += calc_e(sj, si)

  A = np.copy(R) # (normalized) preference matrix
  games_played = M @ np.ones(M.shape[1])
  for i, _ in enumerate(teams):
    A[i, :] /= games_played[i]

  return (teams, R, M, A, E)

# MATRIX METHODS

# approximate eigenvector (rank vector) correlating with largest eigenvalue  of A
def power_method(A: mat_f32, num_ierations: int) -> vec_f32:
  r = np.random.rand(A.shape[1])
  
  for _ in range(num_ierations):
    r = A @ r # multiply r by A
    r /= np.linalg.norm(r) # normalize the result

  return r

def power_method_nonlinear(E: mat_f32, M: mat_f32, f: Callable[[float], float], num_iterations: int) -> vec_f32:
  r = np.random.rand(E.shape[1])
  f_vec = np.vectorize(f)
  games_played = M @ np.ones(M.shape[1])

  # repeatedly apply the F transformation to approximate its fixed point
  for _ in range(num_iterations):
    r = np.divide(f_vec(E @ r), games_played) 
  
  return r

def sort_by_rank(teams: list[str], r: vec_f32) -> list[tuple[str, float]]:
  couple = zip(teams, r)
  return sorted(couple, key=cmp_to_key(lambda a, b: b[1] - a[1]))

# find rank matrix P given a probability vector v
def make_rank_matrix(v: vec_f32) -> mat_f32: 
  V = np.repeat([v], repeats=[v.shape[0]], axis=0)
  return np.power(1 + np.exp(V - np.transpose(V)), -1)

# create a basis V for the probability vector space
def make_v_basis(M: mat_f32) -> mat_f32: 
  n = M.shape[0] - 1
  top = np.full(n, -1, dtype=np.float32)
  return np.vstack([top, np.eye(n)])

# apply the nonlinear T transformation to a probability vector v
def apply_T(v: vec_f32, s: vec_f32, M: mat_f32) -> vec_f32: 
  P = make_rank_matrix(v)
  F = np.dot(np.multiply(M, P), np.ones_like(v))
  return v + s - F

# approximate the fixed point of T
def fixed_point_approx(R: mat_f32, M: mat_f32, num_iterations: int) -> vec_f32: 
  V = make_v_basis(M)
  v = V @ np.random.rand(V.shape[1])
  s = R @ np.ones_like(v)
  
  for _ in range(num_iterations):
    v = apply_T(v, s, M)

  return v

# display teams and ranks in a table
def disp_ranks(team_ranks: list[tuple[str, float]], title: str): 
  fig, ax = plt.subplots()
  plt.rcParams['font.family'] = 'monospace'
  table = ax.table(
    cellText = np.asarray(team_ranks[:15]),
    loc = 'center',
    colLabels = ['Team', 'Rank'])
  table.scale(1, 1.5)
  table.set_fontsize(13)
  table.auto_set_column_width(col=[0,1])
  table.auto_set_font_size(False)
  ax.axis('off')
  ax.set_title(title, pad=20)
  fig.tight_layout()
  fig.set_size_inches(5.25, 4.5)