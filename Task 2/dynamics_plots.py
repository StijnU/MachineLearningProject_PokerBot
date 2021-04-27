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
def plot_dynamics(game, temp, kappa):
    payoff_matrix = utils.game_payoffs_array(game)
    replicator = dynamics.MultiPopulationDynamics(payoff_matrix, dynamics.replicator)
    # replicator = dynamics.SinglePopulationDynamics(payoff_matrix, dynamics.replicator)
    boltzmannq = dynamics.MultiPopulationDynamics(payoff_matrix, lambda state, fitness: dynamics.boltzmannq(state, fitness, temp))
    # boltzmannq = dynamics.SinglePopulationDynamics(payoff_matrix, lambda state, fitness: dynamics.boltzmannq(state, fitness, temp))
    lenient = dynamics.MultiPopulationDynamics(
         payoff_matrix, lambda state, fitness: lenientboltzmannq(state, fitness, payoff_matrix[0], temp, kappa)
    )
    # lenient = dynamics.SinglePopulationDynamics(
    #     payoff_matrix, lambda state, fitness: lenientboltzmannq(state, fitness, payoff_matrix[0], temp, kappa)
    # )
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw=dict(projection="2x2"))
    # _, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw=dict(projection="3x3"))
    ax1.quiver(replicator)
    ax1.set_title("replicator")
    ax2.quiver(boltzmannq)
    ax2.set_title("boltzmannq")
    ax3.quiver(lenient)
    ax3.set_title("lenient")
    plt.tight_layout()
    plt.savefig('plot_dynamics.png')
    plt.show()
    return ax1, ax2, ax3


# Lenient Boltzmann Q: grafiek voor verschillende temperatures
def plot_temperatures(game, temperatures, kappa):
    payoff_matrix = utils.game_payoffs_array(game)
    n = len(temperatures)
    _, axs = plt.subplots(1, n, subplot_kw=dict(projection="2x2"))
    # _, axs = plt.subplots(1, n, subplot_kw=dict(projection="3x3"))
    for (i, t) in enumerate(temperatures):
        lenient = dynamics.MultiPopulationDynamics(
            payoff_matrix, lambda state, fitness: lenientboltzmannq(state, fitness, payoff_matrix[0], t, kappa)
        )
        # lenient = dynamics.SinglePopulationDynamics(
        #     payoff_matrix, lambda state, fitness: lenientboltzmannq(state, fitness, payoff_matrix[0], t, kappa)
        # )
        axs[i].quiver(lenient)
    plt.tight_layout()
    plt.savefig('plot_temperatures.png')
    plt.show()

# Lenient Boltzmann Q: grafiek voor verschillende temperatures
# Als kappas = [1, 2, 5, 25], en temperature = 0.1, komen deze overeen met de grafieken in
# http://www.flowermountains.nl/pub/Bloembergen2010lfaq.pdf, op de laatse pagina 
def plot_kappas(game, temperature, kappas):
    payoff_matrix = utils.game_payoffs_array(game)
    n = len(kappas)
    _, axs = plt.subplots(1, n, subplot_kw=dict(projection="2x2"))
    # _, axs = plt.subplots(1, n, subplot_kw=dict(projection="3x3"))
    for (i, k) in enumerate(kappas):
        lenient = dynamics.MultiPopulationDynamics(
           payoff_matrix, lambda state, fitness: lenientboltzmannq(state, fitness, payoff_matrix[0], temperature, k)
        )
        # lenient = dynamics.SinglePopulationDynamics(
        #     payoff_matrix, lambda state, fitness: lenientboltzmannq(state, fitness, payoff_matrix[0], temperature, k)
        # )
        axs[i].quiver(lenient)
    plt.tight_layout()
    plt.savefig('plot_kappas.png')
    plt.show()  


def main(_):
    game = pyspiel.load_game("matrix_mp")
    # game = pyspiel.create_matrix_game("matching_pennies", "Matching Pennies",
    #                                                ["Heads", "Tails"], ["Heads", "Tails"],
    #                                                [[1, -1], [-1, 1]], [[-1, 1], [1, -1]])
    # game = pyspiel.create_matrix_game("rps", "Rock Paper Scissors",
    #                                 ["Rock", "Paper", "Scissors"], ["Rock", "Paper", "Scissors"],
    #                                 [[0, -1, 1], [1, 0, -1], [-1, 1, 0]], [[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    # game = pyspiel.create_matrix_game("battle_of_sexes", "Battle of the Sexes",
    #                                                ["B", "S"], ["B", "S"],
    #                                               [[2, 0], [0, 1]], [[1, 0], [0, 2]])
    kappa = 5
    kappas = [1, 2, 5, 25]
    temp = 0.1
    temperatures = [float('inf'), 0.72877, 0.00001]
    plot_dynamics(game, temp, kappa)
    plot_temperatures(game, temperatures, kappa)
    plot_kappas(game, temp, kappas)
    # trajectories()

def lenientboltzmannq(state, fitness, A, temp, kappa):
    y = np.linalg.pinv(A) @ fitness
    if np.linalg.det(A) == 0:
        y += np.full(y.shape, 1/len(y))
    n = len(A)
    u = np.zeros(n)
    # print("FOUND y" + str(y))
    for i in range(n):
        for j in range(n):
            u[i] += A[i, j] * y[j] * (sum(y[A[i, :] <= A[i, j]])**kappa - sum(y[A[i, :] < A[i, j]])**kappa) / sum(y[A[i, :] == A[i, j]])
    return dynamics.boltzmannq(state, u, temp)


# def trajectories():
#     game = pyspiel.load_game("matrix_mp")
#     payoff_matrix = utils.game_payoffs_array(game)
#     dyn = dynamics.MultiPopulationDynamics(
#         payoff_matrix, lambda state, fitness: lenientboltzmannq(state, fitness, payoff_matrix[0], 0.1, 6)
#     )
#     start_list = np.array([[0.9,0.1,0.1,0.9],[0.75, 0.25, 0.25, 0.75], [0.6,0.4,0.4,0.6],[0.4,0.6,0.6,0.4],[0.25, 0.75, 0.75, 0.25],[0.1,0.9,0.9,0.1]])
#
#     for start in start_list:
#         x = [start[0]]
#         y = [start[2]]
#         end = False
#         while not end:
#             temp = start + 0.03 * dyn(start)
#             if np.allclose(temp, start):
#                 end = True
#             start = temp
#             for i in range(len(start)):
#                 start[i] = min(1, max(0, start[i]))
#                 if start[i] == 0 or start[i] == 0:
#                     end = True
#             x.append(start[0])
#             y.append(start[2])
#         plt.plot(x, y, 'k')
#     plt.show()


if __name__ == "__main__":
    app.run(main)
