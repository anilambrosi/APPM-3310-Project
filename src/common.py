# utils for cleaning/loading data

from csv import reader as csv_reader
import re
import math
import numpy as np
from io import StringIO

sgn = lambda n: math.copysign(1, n)

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


