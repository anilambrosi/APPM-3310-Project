# code for 1st paper (perron frobenius theorem)

from common import Csv, build_team_matrix

def main():
  csv = Csv('../data/games.csv')
  build_team_matrix(csv)

if __name__ == '__main__':
  main()