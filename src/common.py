# utils for cleaning/loading data

import re
import numpy as np
from csv import reader as csv_reader
from io import StringIO
from functools import cmp_to_key

class Csv:
  def __init__(self, path: str):
    with open(path, mode='r') as file:
      text = file.read() # remove seed from team names
      clean_text = re.sub(r'\([0-9]+\)\s*', '', text)
      csv_file = csv_reader(StringIO(clean_text))
      next(csv_file) # remove header
      self.data = [line for line in csv_file if all(entry != '' for entry in line)]

# find eigenvector correlating with largest eigenvalue of A
def power_method(A: np.ndarray[np.float32], num_ierations: int) -> np.ndarray[np.float32]:
  # find eigenvalue approximation
  r = np.random.rand(A.shape[1])
  for _ in range(num_ierations):
    r = A @ r
    r /= np.linalg.norm(r)

  return r

def sort_by_rank(teams: list[str], rank_vec: np.ndarray[np.float32]) -> list[tuple[str, float]]:
  couple = zip(teams, rank_vec)
  return sorted(couple, key=cmp_to_key(lambda a, b: b[1] - a[1]))