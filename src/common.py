# utils for cleaning/loading data

import numpy as np
import csv
import re
import matplotlib.pyplot as plt
from io import StringIO

class Csv:
  def __init__(self, path: str):
    with open(path, mode='r') as file:
      text = file.read() # remove seed from team names
      clean_text = re.sub(r'\([0-9]+\)\s*', '', text)
      csv_file = csv.reader(StringIO(clean_text))
      next(csv_file) # remove header
      self.data = [line for line in csv_file if all(entry != '' for entry in line)]

def build_team_matrix(csv_obj):
  teams = list(set(line[0] for line in csv_obj.data) | set(line[2] for line in csv_obj.data))
  num_teams = len(teams)
  team_matrix = np.zeros((num_teams, num_teams), dtype=float)
  for line in csv_obj.data:
    w, wpts, l, lpts = line
    wi = teams.index(w)
    li = teams.index(l)
    team_matrix[wi, li] += float(wpts)
    team_matrix[li, wi] += float(lpts)
  
  # testing
  plt.imshow(team_matrix)
  plt.colorbar()
  plt.show()