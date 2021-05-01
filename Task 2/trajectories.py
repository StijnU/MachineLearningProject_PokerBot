import logging
import pyspiel
import matplotlib.pyplot as plt
import numpy as np
import random
from open_spiel.python import rl_environment, rl_tools
from open_spiel.python.algorithms import random_agent, tabular_qlearner, masked_softmax
from open_spiel.python.egt import dynamics
from open_spiel.python.egt import utils
from absl import flags, app
from open_spiel.python.egt.visualization import Dynamics2x2Axes
from open_spiel.python.egt.utils import game_payoffs_array
from open_spiel.python.egt.dynamics import time_average


FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", int(5e4), "Number of train episodes.")

def plot_trajectory(ax, game, probabilities):
    num_actions = int(probabilities.shape[1] / 2)
    if num_actions == 2:
        ax.plot(probabilities[:, 0], probabilities[:, 2], 'b')
        ax.plot(probabilities[0, 0], probabilities[0, 2], 'go')
        ax.plot(probabilities[-1, 0], probabilities[-1, 2], 'ro')
    else:
        ax.plot(probabilities[:, np.array(range(num_actions))])
        # ax.plot(probabilities[:, num_actions + np.array(range(num_actions))])


def plot_q_values(q_values):
    num_actions = int(q_values.shape[1] / 2)
    _, ax = plt.subplots(1, 1)
    ax.set_title("Boltzmann Q Learning: Q Values")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Q-Value")

    if num_actions == 2:
        ax.plot(range(len(q_values)), q_values[:, 0], 'b')
        ax.plot(range(len(q_values)), q_values[:, 1], '--b')
        ax.plot(range(len(q_values)), q_values[:, 2], 'r')
        ax.plot(range(len(q_values)), q_values[:, 3], '--r')
    else:
        ax.plot(range(len(q_values)), q_values[:, 0], 'b')
        ax.plot(range(len(q_values)), q_values[:, 1], ':b')
        ax.plot(range(len(q_values)), q_values[:, 2], '--b')
        ax.plot(range(len(q_values)), q_values[:, 3], 'r')
        ax.plot(range(len(q_values)), q_values[:, 4], ':r')
        ax.plot(range(len(q_values)), q_values[:, 5], '--r')
    plt.savefig('qvalues.png')
    plt.show()


def main(_):
    game = pyspiel.load_game("matrix_rps")
    # game = pyspiel.create_matrix_game("stag_hunt", "Stag Hunt Game",
    #                                        ["Stag", "Hare"], ["Stag", "Hare"],
    #                                        [[4, 1], [3, 3]], 
    #                                        [[4, 3], [1, 3]])
    # game = pyspiel.create_matrix_game("chicken_game", "Chicken Game",
    #                                        ["Swerve", "Straight"], ["Swerve", "Straight"],
    #                                        [[0, -1], [1, -10]], 
    #                                        [[0, 1], [-1, -10]])

    discount_factor = 0.9
    step_size = 0.001
    temp = 1

    env = rl_environment.Environment(game)
    num_actions = env.action_spec()["num_actions"]
    q_agents = [tabular_qlearner.QLearner(player_id=0, num_actions=num_actions, discount_factor=discount_factor, step_size=step_size),
                tabular_qlearner.QLearner(player_id=1, num_actions=num_actions, discount_factor=discount_factor, step_size=step_size)]
        
    training_episodes = 15000

    payoff_matrix = utils.game_payoffs_array(game)
    if num_actions == 2: 
        boltzmannq = dynamics.MultiPopulationDynamics(payoff_matrix, lambda state, fitness: dynamics.boltzmannq(state, fitness, temp))
        _, ax = plt.subplots(1, 1, subplot_kw=dict(projection="2x2"))
        init_q_values_0 = [[-0.08, 0.08], [0, 0], [0.08, -0.08]]
        init_q_values_1 = [[-0.08, 0.08], [0, 0], [0.08, -0.08]]
    else:
        boltzmannq = dynamics.SinglePopulationDynamics(payoff_matrix, lambda state, fitness: dynamics.boltzmannq(state, fitness, temp))
        _, ax = plt.subplots(1, 1, subplot_kw=dict(projection="3x3"))
        init_q_values_0 = [[1, -1, 0]]
        init_q_values_1 = [[0, -1, 1]]

    ax.set_title("Boltzmann Q Learning Dynamics and Trajectory")
    ax.quiver(boltzmannq)

    # intialisatie van Q-values
    for m in range(len(init_q_values_0)):
        for n in range(len(init_q_values_1)):
            print(m)
            print(n)
            #if not(init_q_values_0[m] == [0, 0] and init_q_values_1[n] == [0, 0]):

            time_step = env.reset()
            for k in range(2):
                state = str(time_step.observations['info_state'][k])
                actions = time_step.observations['legal_actions'][k]

                for a in actions:
                    if k == 0:
                        q_agents[k]._q_values[state][a] = init_q_values_0[m][a]
                    else:
                        q_agents[k]._q_values[state][a] = init_q_values_1[n][a]


            q_values = np.zeros((training_episodes, 2 * num_actions))
            probabilities = np.zeros((training_episodes, 2 * num_actions))

            for i in range(training_episodes):
                time_step = env.reset()

                chosen_actions = [-1, -1]
                for j in range(2):
                    state = str(time_step.observations['info_state'][j])
                    actions = time_step.observations['legal_actions'][j]

                    # extract probabilities and sample action
                    probs = masked_softmax.np_masked_softmax(np.array([q_agents[j]._q_values[state][a] for a in actions]) * 1/temp, np.ones(len(actions)))
                    chosen_action = random.choices(actions, weights=probs)[0]
                    chosen_actions[j] = chosen_action

                    probabilities[i, j * num_actions + np.array(range(num_actions))] = probs

                    # update agents
                    q_agents[j]._prev_action = chosen_action
                    q_agents[j]._prev_info_state = state

                    # extract Q values
                    for a in actions:
                        q_values[i, num_actions*j+a] = q_agents[j]._q_values[state][a]

                time_step = env.step(chosen_actions)

                # Episode is over, step all agents with final info state.
                for j in range(2):
                    q_agents[j].step(time_step)

            print(q_values)
            print(probabilities)
            
            # probabilities = time_average(probabilities)

            plot_trajectory(ax, game, probabilities)
            # plot_q_values(q_values)

    plt.savefig('traj.png')
    plt.show()
    plot_q_values(q_values)


if __name__ == "__main__":
    app.run(main)
