from kuhn_poker_nfsp import train_nfsp
from open_spiel.python.algorithms import discounted_cfr
from open_spiel.python.algorithms import exploitability
from absl import app

import pyspiel
import matplotlib.pyplot as plt


def main(unused):
    training_episodes = 1000
    expl, conv = train_discounted_cfr("kuhn_poker", training_episodes)
    return expl, conv, training_episodes

def train_discounted_cfr(game, num_iterations):
  game = pyspiel.load_game(game)
  discounted_cfr_solver = discounted_cfr.DCFRSolver(game)

  expl = []
  conv = []
  for _ in range(num_iterations):
    discounted_cfr_solver.evaluate_and_update_policy()
    current_expl = exploitability.exploitability(game, discounted_cfr_solver.average_policy())
    current_nash_conv = exploitability.nash_conv(game, discounted_cfr_solver.average_policy())
    expl.append(current_expl)
    conv.append(current_nash_conv)
  return expl, conv




