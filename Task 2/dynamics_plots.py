import logging
import pyspiel
import matplotlib.pyplot as plt
import numpy as np
from open_spiel.python.egt import dynamics
from open_spiel.python.egt import utils
from absl import flags, app
from open_spiel.python.egt.visualization import Dynamics2x2Axes
from open_spiel.python.egt.utils import game_payoffs_array

def plot_dynamics(game, temp, kappa):
    payoff_matrix = utils.game_payoffs_array(game)
    replicator = dynamics.MultiPopulationDynamics(payoff_matrix, dynamics.replicator)
    boltzmannq = dynamics.MultiPopulationDynamics(payoff_matrix, lambda state, fitness: dynamics.boltzmannq(state, fitness, temp))
    lenient = dynamics.MultiPopulationDynamics(
         payoff_matrix, lambda state, fitness: lenientboltzmannq(state, fitness, payoff_matrix[0], temp, kappa)
    )
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw=dict(projection="2x2"))
    ax1.quiver(replicator)
    ax1.set_title("replicator")
    ax2.quiver(boltzmannq)
    ax2.set_title("boltzmannq")
    ax3.quiver(lenient)
    ax3.set_title("lenient")
    plt.show()
    plt.savefig('plot_dynamics.png')


def plot_temperatures(game, temperatures, kappa):
    payoff_matrix = utils.game_payoffs_array(game)
    n = len(temperatures)
    _, axs = plt.subplots(1, n, subplot_kw=dict(projection="2x2"))
    for (i, t) in enumerate(temperatures):
        lenient = dynamics.MultiPopulationDynamics(
            payoff_matrix, lambda state, fitness: lenientboltzmannq(state, fitness, payoff_matrix[0], t, kappa)
        )
        axs[i].quiver(lenient)
    plt.show()
    plt.savefig('plot_temperatures.png')


def plot_kappas(game, temperature, kappas):
    payoff_matrix = utils.game_payoffs_array(game)
    n = len(kappas)
    _, axs = plt.subplots(1, n, subplot_kw=dict(projection="2x2"))
    for (i, k) in enumerate(kappas):
        lenient = dynamics.MultiPopulationDynamics(
            payoff_matrix, lambda state, fitness: lenientboltzmannq(state, fitness, payoff_matrix[0], temperature, k)
        )
        axs[i].quiver(lenient)
    plt.show()
    plt.savefig('plot_kappas.png')

def main(_):
    # game = pyspiel.load_game("matrix_pd")
    game = pyspiel.create_matrix_game("stag_hunt", "Stag Hunt Game",
                                            ["Stag", "Hare"], ["Stag", "Hare"],
                                            [[4, 1],
                                             [3, 3]], 
                                            [[4, 3], 
                                             [1, 3]])
    kappa = 5
    kappas = [1, 2, 5, 25]
    temp = 0.1
    temperatures = [float('inf'), 0.72877, 0.00001]
    plot_dynamics(game, temp, kappa)
    plot_temperatures(game, temperatures, kappa)
    plot_kappas(game, temp, kappas)
    

def lenientboltzmannq(state, fitness, A, temp, kappa):
    y = np.linalg.inv(A) @ fitness
    n = len(A)
    u = np.zeros(n)
    for i in range(n):
        for j in range(n):
            u[i] += A[i, j] * y[j] * (sum(y[A[i, :] <= A[i, j]])**kappa - sum(y[A[i, :] < A[i, j]])**kappa) / sum(y[A[i, :] == A[i, j]])
    return dynamics.boltzmannq(state, u, temp)



if __name__ == "__main__":
    app.run(main)
