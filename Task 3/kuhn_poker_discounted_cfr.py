from kuhn_poker_nfsp import train_nfsp
from open_spiel.python.algorithms import discounted_cfr
from open_spiel.python.algorithms import exploitability
from absl import app

import pyspiel
import time
import matplotlib.pyplot as plt


def main(unused):
    training_episodes = 1000
    expl, conv = train_discounted_cfr("kuhn_poker", training_episodes)
    return expl, conv, training_episodes


def train_discounted_cfr(game, num_iterations, eval_every, alpha=3/2, beta=0, gamma=2):
    game = pyspiel.load_game(game)
    discounted_cfr_solver = discounted_cfr.DCFRSolver(game, alpha=alpha, beta=beta, gamma=gamma)

    expl = []
    conv = []
    start_time = time.time()
    for i in range(num_iterations + 1):
        
        if i % eval_every == 0:
            current_expl = exploitability.exploitability(game, discounted_cfr_solver.average_policy())
            current_nash_conv = exploitability.nash_conv(game, discounted_cfr_solver.average_policy())
            expl.append(current_expl)
            conv.append(current_nash_conv)
            print(80 * "-")
            print("Training Episode: " + str(i))
            print("Exploitability: " + str(expl[-1]))
            print("Time elapsed: " + str(time.time() - start_time))
        
        discounted_cfr_solver.evaluate_and_update_policy()
            
    return expl, conv
