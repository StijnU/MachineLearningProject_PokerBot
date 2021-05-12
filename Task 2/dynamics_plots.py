import logging
import pyspiel
import matplotlib.pyplot as plt
import numpy as np
from open_spiel.python.egt import dynamics
from open_spiel.python.egt import utils
from absl import flags, app
from open_spiel.python.egt.visualization import Dynamics2x2Axes, Dynamics3x3Axes
from open_spiel.python.egt.utils import game_payoffs_array


# Grafiek voor Replicator, Boltzmann Q en Lenient Boltzmann Q
def plot_dynamics(game, temp, kappa, nb_actions):
    payoff_matrix = utils.game_payoffs_array(game)
    if nb_actions == 2:
        replicator = dynamics.MultiPopulationDynamics(payoff_matrix, dynamics.replicator)
        lenient = dynamics.MultiPopulationDynamics(
            payoff_matrix, lambda state, fitness: lenientboltzmannq(state, fitness, payoff_matrix[0], temp, kappa)
        )
        _, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection="2x2"))
    else:
        replicator = dynamics.SinglePopulationDynamics(payoff_matrix, dynamics.replicator)
        lenient = dynamics.SinglePopulationDynamics(
            payoff_matrix, lambda state, fitness: lenientboltzmannq(state, fitness, payoff_matrix[0], temp, kappa)
        )
        _, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection="3x3"))
    ax1.quiver(replicator)
    ax2.quiver(lenient)
    plt.tight_layout()
    plt.savefig('plot_dynamics.png')
    plt.show()
    return ax1, ax2

def plot_LFAQ(game, temp, kappa, nb_actions):
    payoff_matrix = utils.game_payoffs_array(game)
    if nb_actions == 2:
        lenient = dynamics.MultiPopulationDynamics(
            payoff_matrix, lambda state, fitness: lenientboltzmannq(state, fitness, payoff_matrix[0], temp, kappa)
        )
        _, ax = plt.subplots(1, subplot_kw=dict(projection="2x2"))
    else:
        lenient = dynamics.SinglePopulationDynamics(
            payoff_matrix, lambda state, fitness: lenientboltzmannq(state, fitness, payoff_matrix[0], temp, kappa)
        )
        _, ax = plt.subplots(1, subplot_kw=dict(projection="3x3"))
    ax.quiver(lenient)
    plt.tight_layout()
    plt.savefig('plot_lfaq.png')
    plt.show()
    return ax


# Lenient Boltzmann Q: grafiek voor verschillende temperatures
def plot_temperatures(game, temperatures, kappa, nb_actions):
    payoff_matrix = utils.game_payoffs_array(game)
    n = len(temperatures)
    if nb_actions == 2:
        _, axs = plt.subplots(1, n, subplot_kw=dict(projection="2x2"))
    else:
        _, axs = plt.subplots(1, n, subplot_kw=dict(projection="3x3"))
    for (i, t) in enumerate(temperatures):
        if nb_actions == 2:
            lenient = dynamics.MultiPopulationDynamics(
                payoff_matrix, lambda state, fitness: lenientboltzmannq(state, fitness, payoff_matrix[0], t, kappa)
            )
        else:
            lenient = dynamics.SinglePopulationDynamics(
                payoff_matrix, lambda state, fitness: lenientboltzmannq(state, fitness, payoff_matrix[0], t, kappa)
            )
        axs[i].quiver(lenient)
    plt.tight_layout()
    plt.savefig('plot_temperatures.png', bbox_inches="tight")
    plt.show()

# Lenient Boltzmann Q: grafiek voor verschillende kappas
def plot_kappas(game, temperature, kappas, nb_actions):
    payoff_matrix = utils.game_payoffs_array(game)
    n = len(kappas)
    if nb_actions == 2:
        _, axs = plt.subplots(1, n, subplot_kw=dict(projection="2x2"))
    else:
        _, axs = plt.subplots(1, n, subplot_kw=dict(projection="3x3"))
    for (i, k) in enumerate(kappas):
        if nb_actions == 2:
            lenient = dynamics.MultiPopulationDynamics(
            payoff_matrix, lambda state, fitness: lenientboltzmannq(state, fitness, payoff_matrix[0], temperature, k)
            )
        else:
            lenient = dynamics.SinglePopulationDynamics(
                payoff_matrix, lambda state, fitness: lenientboltzmannq(state, fitness, payoff_matrix[0], temperature, k)
            )
        axs[i].tick_params(labelsize=6)
        axs[i].quiver(lenient)
    plt.tight_layout()
    plt.savefig('plot_kappas.png', bbox_inches="tight")
    plt.show()  


def main(_):
    game = pyspiel.load_game("matrix_rps")
    # game = pyspiel.create_matrix_game("battle_of_sexes", "Battle of the Sexes",
    #                                                ["B", "S"], ["B", "S"],
    #                                               [[2, 0], [0, 1]], [[1, 0], [0, 2]])
    # game = pyspiel.create_matrix_game("stag_hunt", "Stag Hunt Game",
    #                                         ["Stag", "Hare"], ["Stag", "Hare"],
    #                                         [[4, 1], [3, 3]], 
    #                                         [[4, 3], [1, 3]])
    # game = pyspiel.create_matrix_game("chicken_game", "Chicken Game",
    #                                       ["Swerve", "Straight"], ["Swerve", "Straight"],
    #                                       [[0, -1], [1, -10]], 
    #                                       [[0, 1], [-1, -10]])
    kappa = 5
    kappas = [1, 2, 5, 25]
    temp = 0.1
    temperatures = [float('inf'), 0.72877, 0.001]
    nb_actions = game.num_distinct_actions()
    plot_dynamics(game, temp, kappa, nb_actions)
    plot_LFAQ(game, temp, kappa, nb_actions)
    plot_temperatures(game, temperatures, kappa, nb_actions)
    plot_kappas(game, temp, kappas, nb_actions)

def lenientboltzmannq(state, fitness, A, temp, kappa):
    y = np.linalg.pinv(A) @ fitness
    if np.linalg.det(A) == 0:
        y += np.full(y.shape, 1/len(y))
    n = len(A)
    u = np.zeros(n)
    for i in range(n):
        for j in range(n):
            u[i] += A[i, j] * y[j] * (sum(y[A[i, :] <= A[i, j]])**kappa - sum(y[A[i, :] < A[i, j]])**kappa) / sum(y[A[i, :] == A[i, j]])
    return dynamics.boltzmannq(state, u, temp)

if __name__ == "__main__":
    app.run(main)
